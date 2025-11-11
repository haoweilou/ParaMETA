import utils
import torch
from tts.model import ParaMETATTS
from parameta import ParaMETA
from encoder import Transformer
from sentence_transformers import SentenceTransformer

from g2p import all_ipa_phoneme
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hps = utils.get_hparams()
model = ParaMETATTS(len(all_ipa_phoneme),8,
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).to(device)
model.load_state_dict(torch.load(f"./ckp/parameta_tts.pt"))
# from huggingface_hub import upload_folder
# upload_folder(
#     repo_id="haoweilou/ParaMETA",
#     folder_path="./tts/ParaMETA_TTS",
#     commit_message="Initial upload"
# )
# model.push_to_hub("haoweilou/ParaMETA", repo_path_or_name="./tts/ParaMETA_TTS")
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

# This is to generate speech with speaking style controlled by speech prompt
speech_embed = parameta.encode(mel).to(device)  
prediction = parameta.analysis(mel)
print("Speaking Style Recognition:",prediction)

zh = "今天天气真好，我们一起去公园玩吧".lower()
ipa,tone = mix_to_ipa(zh)
ipa_index = torch.tensor([ipa_to_idx(ipa)]).to(device)  
tone = torch.tensor([tone]).to(device)
src_lens = torch.tensor([ipa_index.shape[-1]]).to(device)
wave,_,_,_ = model.infer(ipa_index,tone,x_lengths=src_lens,length_scale=1,noise_scale=0.5,noise_scale_w=0,ab_style=speech_embed)
save_audio(wave[0].cpu().detach(), 22050, f"speech_ch","./generation/")

text_embed = text_encoder.encode(["female,adult,sad,chinese"])
text_embed = parameta.encode_text(torch.tensor(text_embed).to(device))
wave,_,_,_ = model.infer(ipa_index,tone,x_lengths=src_lens,length_scale=1,noise_scale=0.5,noise_scale_w=0,ab_style=text_embed)
save_audio(wave[0].cpu().detach(), 22050, f"text_ch","./generation/")

en = "how are you doing today? I hope you have a great day!"
ipa,tone = mix_to_ipa(en)
ipa_index = torch.tensor([ipa_to_idx(ipa)]).to(device)  
tone = torch.tensor([tone]).to(device)
src_lens = torch.tensor([ipa_index.shape[-1]]).to(device)
wave,_,_,_ = model.infer(ipa_index,tone,x_lengths=src_lens,length_scale=1,noise_scale=0.5,noise_scale_w=0,ab_style=speech_embed)
save_audio(wave[0].cpu().detach(), 22050, f"speech_en","./generation/")

text_embed = text_encoder.encode(["female,adult,angry,english"])
text_embed = parameta.encode_text(torch.tensor(text_embed).to(device))
wave,_,_,_ = model.infer(ipa_index,tone,x_lengths=src_lens,length_scale=1,noise_scale=0.5,noise_scale_w=0,ab_style=text_embed)
save_audio(wave[0].cpu().detach(), 22050, f"text_en","./generation/")