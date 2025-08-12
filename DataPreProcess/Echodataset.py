from datasets import load_dataset
import os
import soundfile as sf
from tqdm import tqdm
from rich import print
import itertools

output_dir = "/home/kaloori.shiva/Thesis/TIGER/DataPreProcess/EchoSet"
os.makedirs(output_dir, exist_ok=True)

try:
    print("Loading JusperLee/EchoSet...")
    ds = load_dataset("JusperLee/EchoSet", streaming=True)

    for split in ds.keys():
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)

        iterator = iter(ds[split])
        for i, group in enumerate(tqdm(itertools.zip_longest(*[iterator]*3), desc=f"Saving {split}")):
            group = [item for item in group if item is not None]
            if len(group) != 3:
                print(f"Warning: Incomplete group {i} in {split}, skipping")
                continue

            # EchoSet stores only 'wav' field, we must infer file type from __key__
            for item in group:
                key = item["__key__"]
                if "spk1_reverb" in key:
                    tag = "spk1_reverb"
                elif "spk2_reverb" in key:
                    tag = "spk2_reverb"
                elif "mix" in key:
                    tag = "mix"
                else:
                    print(f"Unrecognized key: {key}, skipping")
                    continue

                try:
                    audio = item["wav"]["array"]
                    sr = item["wav"]["sampling_rate"]
                    audio_path = os.path.join(split_dir, f"{tag}_{i:05d}.wav")

                    if sr != 16000:
                        import librosa
                        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                        sr = 16000

                    sf.write(audio_path, audio, samplerate=sr)
                except Exception as e:
                    print(f"Error saving {tag}_{i:05d}.wav: {e}")

except Exception as e:
    print(f"Dataset loading error: {e}")
