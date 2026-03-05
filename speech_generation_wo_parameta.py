from tts.model import ParaStyleTTS2
from encoder import Transformer
from g2p import all_ipa_phoneme,mix_to_ipa,ipa_to_idx
import torch
import utils
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

path = hf_hub_download("haoweilou/ParaMETA", "speech_only/model.safetensors")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# this model is the tts model + speaking style encoder traiend in end-to-end manner.
# speech-only model in the paper
hps = utils.get_hparams()
model = ParaStyleTTS2(len(all_ipa_phoneme),8, hps.data.filter_length // 2 + 1, hps.train.segment_size // hps.data.hop_length, **hps.model)
model.eval()
model.speech_encoder = Transformer()
model.to(device)
state = load_file(path)
model.load_state_dict(state, strict=False)

text = "今天天气真好,我们一起出去玩吧"
ipa,tone = mix_to_ipa(text)
mel1 = utils.load_wav_to_mel("./generation/parameta_wo/reference1.wav").unsqueeze(0).to(device)
if mel1.shape[-1] % 4 != 0:  mel1 = mel1[:, :, :mel1.shape[-1] - (mel1.shape[-1] % 4)]
mel2 = utils.load_wav_to_mel("./generation/parameta_wo/reference2.wav").unsqueeze(0).to(device)
if mel2.shape[-1] % 4 != 0:  mel2 = mel2[:, :, :mel2.shape[-1] - (mel2.shape[-1] % 4)]

ipa_index = torch.tensor([ipa_to_idx(ipa)]).to(device)  
tone = torch.tensor([tone]).to(device)
src_lens = torch.tensor([ipa_index.shape[-1]]).to(device)
wave,_,_,_ = model.infer(ipa_index,tone,x_lengths=src_lens,length_scale=1,noise_scale=0.5,noise_scale_w=0,mel=mel1)
utils.save_audio(wave[0].cpu().detach(), 22050,"speech_ch_female.wav", "./generation/parameta_wo/")
wave,_,_,_ = model.infer(ipa_index,tone,x_lengths=src_lens,length_scale=1,noise_scale=0.5,noise_scale_w=0,mel=mel2)
utils.save_audio(wave[0].cpu().detach(), 22050,"speech_ch_male.wav", "./generation/parameta_wo/")


text = "today's weather is good, let us go play"
ipa,tone = mix_to_ipa(text)
ipa_index = torch.tensor([ipa_to_idx(ipa)]).to(device)  
tone = torch.tensor([tone]).to(device)
src_lens = torch.tensor([ipa_index.shape[-1]]).to(device)
wave,_,_,_ = model.infer(ipa_index,tone,x_lengths=src_lens,length_scale=1,noise_scale=0.5,noise_scale_w=0,mel=mel1)
utils.save_audio(wave[0].cpu().detach(), 22050,"speech_en_female.wav", "./generation/parameta_wo/")
wave,_,_,_ = model.infer(ipa_index,tone,x_lengths=src_lens,length_scale=1,noise_scale=0.5,noise_scale_w=0,mel=mel2)
utils.save_audio(wave[0].cpu().detach(), 22050,"speech_en_male.wav", "./generation/parameta_wo/")