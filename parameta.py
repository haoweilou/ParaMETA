import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import Transformer
import numpy as np
#age: child 0-12, teenager: 13-19, youngadult: 20-30, adult: 31-50, senior: 50+
para_category = {
    "emotion":["happy","angry","sad","neutral","surprise","disgust","fear","unknown"],
    "age":["child","teenager","youngadult","adult","senior","unknown"],
    "nation":["chinese","english","unknown"],
    "gender":["male","female","unknown"]
}
category = ["emotion","age","nation","gender"]

def caption_to_idx(caption:str):
    #caption is in order: gender, age, emotion, language
    #label should be in order of: emotion,age,nation,gender
    c = caption.split(",")
    label = []
    label.append(para_category['emotion'].index(c[2]))
    label.append(para_category['age'].index(c[1]))
    label.append(para_category['nation'].index(c[3]))
    label.append(para_category['gender'].index(c[0]))
    return label

def idx_to_caption(idx):
    return f"{para_category['emotion'][idx[0]]},{para_category['age'][idx[1]]},{para_category['nation'][idx[2]]},{para_category['gender'][idx[3]]}"

class ParaMETAPrototypes(nn.Module):
    def __init__(self, para_category, embedding_dim=128):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Create a nn.ParameterDict to hold learnable prototypes for each category
        # Each category maps to an nn.Embedding of (num_classes, embedding_dim)
        self.prototypes = nn.ParameterDict()
        for category, classes in para_category.items():
            self.prototypes[category] = nn.Parameter(
                torch.randn(len(classes), embedding_dim)
            )

    def freeze(self,freeze=True):
        for param in self.prototypes.parameters():
            param.requires_grad = freeze
        
    def forward(self):
        # Return normalized prototypes per category (L2 norm)
        normalized = {}
        for category, emb in self.prototypes.items():
            normalized[category] = F.normalize(emb, dim=1)  # shape: [num_classes, embedding_dim]
        return normalized
    
class TextEncoder(nn.Module):
    """Some Information about SpeechEncoder"""
    def __init__(self,embed_dim=768):
        super().__init__()
        self.linear = nn.Linear(768,embed_dim)

    def forward(self, x):
        x = self.linear(x) #B,T
        return x

def supervised_contrastive_loss(embeddings, labels, temperature=0.07,ignore_label=-1):
    """
    Compute supervised contrastive loss for one category.

    Args:
        embeddings: Tensor [B, D], normalized or raw embeddings
        labels: Tensor [B], class indices for the category
        temperature: float, temperature scaling

    Returns:
        scalar loss
    """
    device = embeddings.device
    batch_size = embeddings.shape[0]

    # Normalize embeddings for cosine similarity
    embeddings = F.normalize(embeddings, p=2, dim=1)

    # Compute similarity matrix [B, B]
    sim_matrix = torch.matmul(embeddings, embeddings.T) / temperature
    
    # dist_matrix = torch.cdist(embeddings, embeddings, p=2)  # [B, B]

    # Create label mask: [B, B] where mask[i,j] = 1 if labels[i]==labels[j], else 0
    labels = labels.contiguous().view(-1, 1)
    
    # Create mask for valid (known) labels
    valid_mask = (labels != ignore_label).float()  # [B,1], 1 if known label else 0

    # Broadcast valid mask for pairwise mask
    valid_pair_mask = valid_mask @ valid_mask.T  
    
    # Create label equality mask (positive pairs)
    label_eq_mask = torch.eq(labels, labels.T).float()

    # Combine: positive pairs only if both known and labels match
    mask = label_eq_mask * valid_pair_mask

    # For each anchor, mask out self similarity (i == j)
    logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
    mask = mask * logits_mask

    # Compute log-softmax over rows
    exp_sim = torch.exp(sim_matrix) * logits_mask
    log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

    # Number of positive samples per anchor
    positives_per_row = mask.sum(dim=1)

    # Avoid division by zero
    positives_per_row = torch.clamp(positives_per_row, min=1.0)
    # Compute mean log-likelihood over positives for each anchor
    loss = -(mask * log_prob).sum(dim=1) / positives_per_row
    return loss.mean()

@torch.no_grad()
def momentum_update_prototypes(prototypes, embeddings, labels, categories, momentum=0.99):
    """
    Update prototypes using momentum (EMA) toward batch sample centroids.

    Args:
        prototypes: dict {category: Tensor[num_classes, D]} - prototype embeddings
        embeddings: Tensor [B, D] - normalized sample embeddings
        labels: Tensor [B, C] - class indices per category
        categories: list of category names
        momentum: float between 0 and 1
    """
    embeddings = F.normalize(embeddings, p=2, dim=1)

    for i, category in enumerate(categories):
        category_labels = labels[:, i]
        proto = prototypes[category]

        for c in torch.unique(category_labels):
            mask = (category_labels == c)
            if mask.sum() == 0:
                continue

            batch_mean = embeddings[mask].mean(dim=0)
            batch_mean = F.normalize(batch_mean.unsqueeze(0), p=2, dim=1).squeeze(0)

            # EMA update
            proto[c] = momentum * proto[c] + (1 - momentum) * batch_mean
            proto[c] = F.normalize(proto[c].unsqueeze(0), p=2, dim=1).squeeze(0)





@torch.no_grad()
def momentum_update_prototypes_mh(prototypes, speech_embeddings, labels, categories, momentum=0.99):
    """
    Update prototypes using momentum (EMA) toward batch sample centroids.

    Args:
        prototypes: dict {category: Tensor[num_classes, D]} - prototype embeddings
        embeddings: Tensor [num_category,B, D] - normalized sample embeddings
        labels: Tensor [B, C] - class indices per category
        categories: list of category names
        momentum: float between 0 and 1
    """

    for i, category in enumerate(categories):
        category_labels = labels[:, i]
        proto = prototypes[category]

        embeddings = speech_embeddings[i]
        embeddings = F.normalize(embeddings, p=2, dim=1)

        for c in torch.unique(category_labels):
            mask = (category_labels == c)
            if mask.sum() == 0:
                continue

            batch_mean = embeddings[mask].mean(dim=0)
            batch_mean = F.normalize(batch_mean.unsqueeze(0), p=2, dim=1).squeeze(0)

            # EMA update
            proto[c] = momentum * proto[c] + (1 - momentum) * batch_mean
            proto[c] = F.normalize(proto[c].unsqueeze(0), p=2, dim=1).squeeze(0)

#each part of embedding is responsibile for different tasks
def category_supervised_contrastive_loss(embeddings,labels,temperature=0.07):
    total_loss = 0
    for i in range(labels.shape[1]):
        category_labels = labels[:, i].clone()  # shape [B]
        start = i * 192
        end = start + 192
        if end == 768: end = None
        loss_cat = supervised_contrastive_loss(embeddings[:,start:end], category_labels,temperature) 
        total_loss += loss_cat
    return total_loss

def prototype_alignment_loss(embeddings, prototypes_dict, labels):
    """
    Compute prototype alignment loss by pulling each sample embedding 
    towards its corresponding class prototype embedding.

    Args:
        embeddings: Tensor [B, D] - normalized or raw sample embeddings
        prototypes_dict: dict {category: Tensor[num_classes, D]} - prototype embeddings
        labels: Tensor [B, C] - class indices per category

    Returns:
        scalar tensor loss
    """
    total_loss = 0
    embeddings = F.normalize(embeddings, p=2, dim=1)  # normalize embeddings
    categories = list(prototypes_dict.keys())

    for i, cat in enumerate(categories):
        category_labels = labels[:, i].to(embeddings.device)  # [B]
        prototypes = prototypes_dict[cat]                 # [num_classes, D]
        prototypes = F.normalize(prototypes, p=2, dim=1)       # normalize prototypes
        start = i * 192
        end = start + 192
        if end == 768: end = None

        # Select prototypes corresponding to each sample's class
        class_protos = prototypes[category_labels]             # [B, D]
        # Calculate cosine distance (1 - cosine similarity)
        loss_sample_to_proto = (1 - F.cosine_similarity(embeddings[:,start:end], class_protos, dim=1)).mean()
        total_loss += loss_sample_to_proto
    return total_loss

def meta_regularization_loss(z, labels, temperature=0.1, eps=1e-8):
    """
    z: (B, D) tensor of embeddings
    labels: (B, 4) tensor of multi-factor labels (int)
    temperature: scalar
    """
    B, D = z.shape
    z = F.normalize(z, p=2, dim=1)

    # Step 1: compute cosine similarity logits
    logits = torch.mm(z, z.T) / temperature  # (B, B)
    logits = logits - torch.max(logits, dim=1, keepdim=True).values  # numerical stability

    # Step 2: compute similarity weights based on label overlap
    labels_a = labels.unsqueeze(1).expand(B, B, 4)  # (B, B, 4)
    labels_b = labels.unsqueeze(0).expand(B, B, 4)  # (B, B, 4)

    matches = (labels_a == labels_b).float()  # (B, B, 4)
    sim_weights = matches.mean(dim=-1)        # (B, B), in [0,1]

    # Remove self-similarity
    mask = torch.eye(B, device=z.device).bool()
    sim_weights.masked_fill_(mask, 0.0)

    # Step 3: compute log-softmax over logits
    log_probs = F.log_softmax(logits, dim=1)  # (B, B)

    # Step 4: weighted loss
    weighted_log_probs = sim_weights * log_probs  # (B, B)
    pos_weights = sim_weights.sum(dim=1) + eps    # (B,)

    loss_per_sample = - (weighted_log_probs.sum(dim=1) / pos_weights)  # (B,)
    loss = loss_per_sample.mean()
    return loss

class ParaMETA(nn.Module):
    def __init__(self,embed_dim=768,speech_encoder=None,weight=0.1):
        super().__init__()
        self.speech_encoder = speech_encoder if speech_encoder is not None else Transformer()
        self.proto_embed = ParaMETAPrototypes(para_category,embedding_dim=192)
        self.text_encoder = TextEncoder(embed_dim)

        self.nation_encoder = nn.Linear(embed_dim,192)
        self.gender_encoder = nn.Linear(embed_dim,192)
        self.emotion_encoder = nn.Linear(embed_dim,192)
        self.age_encoder = nn.Linear(embed_dim,192)

        self.norm_emotion = nn.LayerNorm(192, elementwise_affine=False)
        self.norm_age = nn.LayerNorm(192, elementwise_affine=False)
        self.norm_nation = nn.LayerNorm(192, elementwise_affine=False)
        self.norm_gender = nn.LayerNorm(192, elementwise_affine=False)
                
        self.weight = weight

    def forward(self, speech,text):
        text_embed = self.text_encoder(text)
        #B,768
        meta_embed = self.speech_encoder(speech)
        #B,192
        nation_embed = self.nation_encoder(meta_embed)
        age_embed = self.age_encoder(meta_embed)
        emotion_embed = self.emotion_encoder(meta_embed)
        gender_embed = self.gender_encoder(meta_embed)
        #B,768, in order of emotion,age,nation,gender
        speech_embed = torch.cat([emotion_embed, age_embed, nation_embed, gender_embed], dim=1)
        prototypes = self.proto_embed() 
        return speech_embed,meta_embed,text_embed,prototypes
    
    def encode_text(self,text):
        #text is [B,L], is a batch of text of length L
        text_embed = self.text_encoder(text)

        emotion_embed = self.norm_emotion(text_embed[:, :192])
        age_embed = self.norm_age(text_embed[:, 192:192*2])
        nation_embed = self.norm_nation(text_embed[:, 192*2:192*3])
        gender_embed = self.norm_gender(text_embed[:, 192*3:])
        #B,768, in order of emotion,age,nation,gender
        text_embed = torch.cat([emotion_embed, age_embed, nation_embed, gender_embed], dim=1)
        return text_embed
    
    def encode(self,speech):
        meta_embed = self.speech_encoder(speech)
        #B,192
        nation_embed = self.norm_nation(self.nation_encoder(meta_embed))
        age_embed = self.norm_age(self.age_encoder(meta_embed))
        emotion_embed = self.norm_emotion(self.emotion_encoder(meta_embed))
        gender_embed = self.norm_gender(self.gender_encoder(meta_embed))
        #B,768, in order of emotion,age,nation,gender
        speech_embed = torch.cat([emotion_embed, age_embed, nation_embed, gender_embed], dim=1)
        return speech_embed
    
    def enc_loss(self,text_embed,speech_embed,meta_embed,labels):
        #speech embed in order of emotion,age,nation,gender
        #text is [B,L], is a batch of text of length L
        #speech is [B,F,T] is a batch of speech of length T
        #labels is [B,C], is a batch of label, C = number of category
        prototypes = self.proto_embed() #dict of  {category: tensor of shape [num_classes, emb_dim]}
        
        #ensure prototype follow the distribution of push different class away, and pull same class within each category together
        #ensure speech&text embedding follow the distribution of push different class away, and pull same class within each category together
        s_scl = category_supervised_contrastive_loss(speech_embed,labels)

        #prototype alignment loss
        s_pal = prototype_alignment_loss(speech_embed,prototypes,labels)
        t_pal = prototype_alignment_loss(text_embed,prototypes,labels)

        #meta space regularization
        meta_scl = meta_regularization_loss(meta_embed,labels)

        total_loss = s_scl + meta_scl + self.weight*(s_pal+t_pal)
        return total_loss,s_scl,meta_scl,s_pal,t_pal
    
    def embed_label(self,embed,prototypes):
        #get classfication label for the batch of embedding given
        #embed: [B,D]
        #prototypes: dict of embedding of shape [NUM_CLASS,D]
        #output: label: [B,C], C is number of category
        embed_norm = F.normalize(embed, dim=1)  # [B, D]
        preds = []
        i = 0
        for i,cat in enumerate(["emotion","age","nation","gender"]):
            proto_tensor = prototypes[cat]
            proto_norm = F.normalize(proto_tensor, dim=1)  # [num_classes, D]
            start = i * 192
            end = start + 192
            if end == 768: end = None
            sim = torch.matmul(embed_norm[:,start:end], proto_norm.T)   # [B, num_classes]
            pred_classes = torch.argmax(sim, dim=1)        # [B]
            preds.append(pred_classes.unsqueeze(1))        # [B,1]
        pred_labels = torch.cat(preds, dim=1)              # [B, C]
        return pred_labels
        
    def analysis(self, spec_batch):
        """
        Analyze a batch of spectrograms and return a list of classification strings.

        Args:
            spec_batch: Tensor of shape [B, ...], batch of spectrograms
        Returns:
            List[str]: list of strings
        """
        # Encode batch
        speech_embed = self.encode(spec_batch)
        # Get prototypes
        prototypes = self.proto_embed()  # dict or tensor depending on your implementation
        # Import the helper function
        # Predict labels
        result = self.embed_label(speech_embed, prototypes)  # assumed output shape: [B, 4]
        result = result.detach().cpu().numpy().astype(np.int32)

        # Build result strings
        results = []
        for res in result:  # res is a 4-element array: [emotion_idx, age_idx, nation_idx, gender_idx]
            emotion = para_category["emotion"][res[0]]
            age = para_category["age"][res[1]]
            nation = para_category["nation"][res[2]]
            gender = para_category["gender"][res[3]]
            results.append(f"{gender},{age},{emotion},{nation}")
        return results  # list of strings, length = batch size

@torch.no_grad()
def EMA(prototypes, embeddings, labels, categories, momentum=0.99):
    """
    Update prototypes using momentum (EMA) toward batch sample centroids.

    Args:
        prototypes: dict {category: Tensor[num_classes, D]} - prototype embeddings
        embeddings: Tensor [B, D] - normalized sample embeddings
        labels: Tensor [B, C] - class indices per category
        categories: list of category names
        momentum: float between 0 and 1
    """
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    for i, category in enumerate(categories):
        category_labels = labels[:, i]
        proto = prototypes[category]
        start = i * 192
        end = start + 192
        if end == 762: end = None

        for c in torch.unique(category_labels):
            mask = (category_labels == c)
            if mask.sum() == 0: continue

            batch_mean = embeddings[mask].mean(dim=0)[start:end]
            batch_mean = F.normalize(batch_mean.unsqueeze(0), p=2, dim=1).squeeze(0)

            # EMA update
            proto[c] = momentum * proto[c] + (1 - momentum) * batch_mean
            proto[c] = F.normalize(proto[c].unsqueeze(0), p=2, dim=1).squeeze(0)
    return 