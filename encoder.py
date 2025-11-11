import torch 
import torch.nn as nn
from einops.layers.torch import Rearrange
import math
import torch.nn.functional as F
# This is the folder for Speech Encoder

def sinusoidal(T, dim,device="cuda"):
    position = torch.arange(T, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)).to(device)
    pe = torch.zeros(T, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

class PatchEmbedding(nn.Module):
    def __init__(self, patch_width=4, patch_height=4, embed_dim=16):
        super().__init__()
        patch_size = (patch_height, patch_width)
        self.proj = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B, 1, F, T]
        x  = self.proj(x)  # [B, C, F', T']
        #F' = F/patch_height, T' = T/patch_width
        return x  # [B, C, F', T']


class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.ffn_norm = nn.LayerNorm(embed_dim)

    def forward(self, latents, x):
        # latents: [B, N, C], x: [B, T*F, C]
        residual = latents
        latents, _ = self.attn(latents, x, x)         # Cross-attention
        latents = self.norm(latents + residual)       # Add & Norm
        residual = latents
        latents = self.ffn(latents)
        latents = self.ffn_norm(latents + residual)   # Add & Norm
        return latents

class CrossAtten(nn.Module):
    def __init__(self, embed_dim=64, num_tokens=12,depth=2, num_heads=4):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, num_tokens, embed_dim))  # [1, N, C]
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(embed_dim, num_heads) for _ in range(depth)
        ])

    def forward(self, x,T, F):
        # x: [B, T'*F', C]
        B, TF, C = x.shape
        assert TF == T * F, "Input shape does not match T and F"
        latents = self.latents.expand(B, -1, -1).contiguous()  # [B, N, C]
        for block in self.blocks:
            latents = block(latents, x)  # [B, N, C]
        return latents  # [B, N, C]
    
class QFormer(nn.Module):
    def __init__(self, embed_dim=64, patch_width=4, patch_height=4,num_tokens=12,depth=12):
        super().__init__()
        self.patch_embedding = PatchEmbedding(patch_width, patch_height, embed_dim)
        self.embed_dim = embed_dim
        self.transformers = CrossAtten(embed_dim=embed_dim,depth=depth,num_tokens=num_tokens)  # Transformer blocks
        self.fc = nn.Linear(embed_dim*num_tokens, embed_dim*num_tokens)  # Final linear layer for output

    def forward(self, x):
        x = x.unsqueeze(1)
        # x: [B, 1, F, T]
        x = self.patch_embedding(x)  # [B, C, F', T']
        B,C,F,T = x.shape
        pos_emb = sinusoidal(T, self.embed_dim, device=x.device)  # [T', C]
        pos_emb = pos_emb.unsqueeze(0).unsqueeze(2) # [1, T', 1, C]
        x = Rearrange('b c f t -> b t f c')(x)  # [B, T', F', C]
        x = x + pos_emb # [B, T', F', C] 
        x = Rearrange('b t f c -> b (t f) c')(x) # [B, T'*F', C]
        x = self.transformers(x, T, F)
        x = x.reshape(B, -1)  # Flatten to [B, N*C]
        x = self.fc(x)
        return x
    
class TextEncoder(nn.Module):
    def __init__(self, embed_dim=756):
        super().__init__()
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        x = self.fc(x)  # [B, C]
        return x
    
class Transformer(nn.Module):
    def __init__(self, embed_dim=64, depth=12):
        super().__init__()
        self.embed_dim = embed_dim
        self.cov = nn.Conv1d(80,embed_dim,4,stride=4)
        num_heads = 4
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        self.fc = nn.Linear(embed_dim, 768)

    def forward(self, x):
        # x: [B, F, T]
        x = self.cov(x)
        B,F,T = x.shape
        pos_emb = sinusoidal(T, self.embed_dim, device=x.device)  # [T, F]
        pos_emb = pos_emb.unsqueeze(0) # [1, T, F]
        x = Rearrange('b f t -> b t f')(x)  # [B, T, F]
        x = x + pos_emb # [B, T, F]
        for block in self.blocks:
            x = block(x, x)  # [B, T, F]
        x = torch.sum(x,dim=1)
        x = self.fc(x)
        return x
    
#wrapper for contrastive-based training
class Contrastive(nn.Module):
    """Some Information about Contrastive"""
    def __init__(self, speech_encoder):
        super().__init__()
        self.text_encoder = TextEncoder(768)
        self.speech_encoder = speech_encoder

    def forward(self, speech,text=None):
        if text is not None:
            text_embed = self.text_encoder(text)
            return speech_embed,text_embed
        else: 
            speech_embed = self.speech_encoder(speech)
            return speech_embed
        
class MultiCategoryClassifier(nn.Module):
    def __init__(self, embed_dim, num_classes_per_category):
        """
        embed_dim: int, dimension D of speech_embed
        num_classes_per_category: list[int], length C, each element is number of classes for that category
        """
        super().__init__()
        self.num_categories = len(num_classes_per_category)
        self.classifiers = nn.ModuleList([
            nn.Linear(embed_dim, num_classes_per_category[i])
            for i in range(self.num_categories)
        ])

    def forward(self, speech_embed):
        """
        speech_embed: [B, D]
        returns:
          logits_list: list of length C, each tensor shape [B, K_i]
        """
        logits_list = []
        for classifier in self.classifiers:
            logits_list.append(classifier(speech_embed))
        return logits_list
    

#wrapper for classification-based training
class Classifiaction(nn.Module):
    """Some Information about Contrastive"""
    def __init__(self, speech_encoder):
        super().__init__()
        self.speech_encoder = speech_encoder
        from parameta import para_category,category
        num_class_per_category = [len(para_category[c]) for c in category]
        self.classifier = MultiCategoryClassifier(768,num_class_per_category)

    def forward(self, speech):
        speech_embed = self.speech_encoder(speech)
        logits = self.classifier(speech_embed)
        return logits
    

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=True):
        super(ResBlock, self).__init__()
        stride = 2 if downsample else 1
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
            nn.BatchNorm2d(out_channels)
        ) if downsample or in_channels != out_channels else nn.Identity()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        identity = self.downsample(x)
        out = self.block(x)
        out += identity
        return F.relu(out)

class CNN(nn.Module):
    def __init__(self, embed_dim=768, in_channels=1):
        super(CNN, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),  # [B, 32, F, T]
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.encoder = nn.Sequential(
            ResBlock(32, 64),       # [B, 64, F/2, T/2]
            ResBlock(64, 64, False),
            ResBlock(64, 128),      # [B, 128, F/4, T/4]
            ResBlock(128, 128, False)
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, embed_dim)
        

    def forward(self, x):
        x = x.unsqueeze(1)  # [B, 1, F, T]
        x = self.stem(x)
        x = self.encoder(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)

 
class LSTM(nn.Module):
    def __init__(self, embed_dim=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.cov = nn.Conv1d(80,embed_dim,4,stride=4)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=768, batch_first=True)

    def forward(self, x):
        # x: [B, F, T]
        x = self.cov(x)
        B,F,T = x.shape
        x = Rearrange('b f t -> b t f')(x)  # [B, T, F]
        _, (h_n, _) = self.lstm(x)  # h_n: [1, B, 768]
        x = h_n.squeeze(0)  # [B, 768]
        return x