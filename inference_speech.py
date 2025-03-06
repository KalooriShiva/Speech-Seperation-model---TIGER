import yaml
import os
import look2hear.models
import argparse
import torch
import torchaudio


# audio path
parser = argparse.ArgumentParser()
parser.add_argument("--audio_path", default="test/mix.wav", help="Path to audio file")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load model

model = look2hear.models.TIGER.from_pretrained("JusperLee/TIGER-speech", cache_dir="cache")
model.to(device)
model.eval()

audio = torchaudio.load(parser.parse_args().audio_path)[0].to(device)
with torch.no_grad():
    ests_speech = model(audio[None])

torchaudio.save(
    f"test/spk1.wav", ests_speech[:,0].cpu(), 16000
)
torchaudio.save(
    f"test/spk2.wav", ests_speech[:,1].cpu(), 16000
)
