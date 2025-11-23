import utils
import torch
from tts.model import ParaMETATTS
from parameta import ParaMETA
from encoder import Transformer
from sentence_transformers import SentenceTransformer

from g2p import all_ipa_phoneme
from parameta import para_category

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hps = utils.get_hparams()
model = ParaMETATTS.from_pretrained("haoweilou/ParaMETA").to(device)
model.eval()

parameta = ParaMETA(768,Transformer()).cuda()
text_encoder = SentenceTransformer('all-mpnet-base-v2',trust_remote_code=True)
ckp  = torch.load(f"./ckp/ParaMETA_Transformer.pt")

parameta.load_state_dict(ckp['net'])
print(ckp.keys(),ckp['epoch'])
# load reference wav
from utils import load_wav_to_mel

mel = load_wav_to_mel("./sample/12.wav")
mel = mel.unsqueeze(0).cuda()
if mel.shape[-1] % 4 != 0: 
    mel = mel[:, :, :mel.shape[-1] - (mel.shape[-1] % 4)]
model.eval()

from g2p import mix_to_ipa,ipa_to_idx
from utils import save_audio

speech_embed = parameta.encode(mel).to(device)  
prediction = parameta.analysis(mel)
print("Speaking Style Recognition:",prediction)

en = "how are you doing today? I hope you have a great day!"
ipa,tone = mix_to_ipa(en)
ipa_index = torch.tensor([ipa_to_idx(ipa)]).to(device)  
tone = torch.tensor([tone]).to(device)
src_lens = torch.tensor([ipa_index.shape[-1]]).to(device)
wave,_,_,_ = model.infer(ipa_index,tone,x_lengths=src_lens,length_scale=1,noise_scale=0.5,noise_scale_w=0,ab_style=speech_embed)
# this is the manipulation with original speaking style from reference speech
save_audio(wave[0].cpu().detach(), 22050, f"manipulate_ori","./manipulation/")

# now manipulate the emotion 
protoytype = parameta.proto_embed()
for emotion_idx in range(len(protoytype['emotion'])-1): # exclude unknow class
    emotion = para_category['emotion'][emotion_idx]
    print("Manipulate speech with emotion:",emotion)
    emo_embed = parameta.norm_emotion(protoytype['emotion'][emotion_idx])
    style_embed = speech_embed.clone()
    style_embed[:,0:192] = emo_embed
    wave,_,_,_ = model.infer(ipa_index,tone,x_lengths=src_lens,length_scale=1,noise_scale=0.5,noise_scale_w=0,ab_style=style_embed)
    save_audio(wave[0].cpu().detach(), 22050, f"manipulate_{emotion}","./manipulation/emotion/")

# emotion, age, language, gender
for age_idx in range(len(protoytype['age'])-1): # exclude unknow class
    age = para_category['age'][age_idx]
    print("Manipulate speech with age:",age)
    age_embed = parameta.norm_emotion(protoytype['age'][age_idx])
    style_embed = speech_embed.clone()
    style_embed[:,192:384] = age_embed
    wave,_,_,_ = model.infer(ipa_index,tone,x_lengths=src_lens,length_scale=1,noise_scale=0.5,noise_scale_w=0,ab_style=style_embed)
    save_audio(wave[0].cpu().detach(), 22050, f"manipulate_{age}","./manipulation/age/")

for language_idx in range(len(protoytype['nation'])-1): # exclude unknow class
    language = para_category['nation'][language_idx]
    print("Manipulate speech with language:",language)
    lang_embed = parameta.norm_emotion(protoytype['nation'][language_idx])
    style_embed = speech_embed.clone()
    style_embed[:,384:576] = lang_embed
    wave,_,_,_ = model.infer(ipa_index,tone,x_lengths=src_lens,length_scale=1,noise_scale=0.5,noise_scale_w=0,ab_style=style_embed)
    save_audio(wave[0].cpu().detach(), 22050, f"manipulate_{language}","./manipulation/language/")


for gender_index in range(len(protoytype['gender'])-1): # exclude unknow class
    gender = para_category['gender'][gender_index]
    print("Manipulate speech with gender:",gender)
    gender_embed = parameta.norm_emotion(protoytype['gender'][gender_index])
    style_embed = speech_embed.clone()
    style_embed[:,576:] = gender_embed
    wave,_,_,_ = model.infer(ipa_index,tone,x_lengths=src_lens,length_scale=1,noise_scale=0.5,noise_scale_w=0,ab_style=style_embed)
    save_audio(wave[0].cpu().detach(), 22050, f"manipulate_{gender}","./manipulation/gender/")