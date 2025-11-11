from encoder import Transformer,QFormer,CNN,LSTM
from parameta import ParaMETA
import argparse
import torch
from utils import load_wav_to_mel
import pandas as pd

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
samples = pd.read_csv("./sample/label.csv")

parser = argparse.ArgumentParser()
parser.add_argument('--unknow_exist', type=bool, default=False, help='whether the prediction contains unknow class, recommend to set False for inference')
parser.add_argument('--model', type=str, default="Transformer", help='choose the backbone model from Transformer, QFormer, CNN, LSTM')
args = parser.parse_args()

if args.model not in ["Transformer","QFormer","CNN","LSTM"]:
    raise ValueError("model must be one of Transformer, QFormer, CNN, LSTM")

model = args.model
speech_encoder = None
if model == "Transformer":
    speech_encoder = Transformer().to(device)
elif model == "QFormer":
    speech_encoder = QFormer().to(device)
elif model == "CNN":
    speech_encoder = CNN().to(device)
elif model == "LSTM":
    speech_encoder = LSTM().to(device)

parameta = ParaMETA(768, speech_encoder).to(device) 
ckp  = torch.load(f"./ckp/ParaMETA_{model}.pt")
parameta.load_state_dict(ckp['net'])
print(f"Loaded ParaMETA_{model} checkpoint:", ckp.keys(), ckp['epoch'])

# classification
for idx, row in samples.iterrows():
    file_path = f"./sample/{row['file']}"
    mel = load_wav_to_mel(file_path).to(device).unsqueeze(0)
    if mel.shape[-1] % 4 != 0: mel = mel[:, :, :mel.shape[-1] - (mel.shape[-1] % 4)]
    result = parameta.analysis(mel)
    print(f"Speech: {row['file']} Predict: [{result[0]}] Real: [{row['gender']},{row['age']},{row['emotion']},{row['language']}]")