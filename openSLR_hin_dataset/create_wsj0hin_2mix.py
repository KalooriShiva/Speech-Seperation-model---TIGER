#!/usr/bin/env python3
import os
import random
import argparse
from pathlib import Path
import soundfile as sf
import numpy as np

def read_audio(path, sr):
    audio, fs = sf.read(path)
    if fs != sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=fs, target_sr=sr)
    return audio

def write_audio(path, audio, sr):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sr)

def rms(x):
    return np.sqrt(np.mean(x**2))

def set_relative_level(sig1, sig2, target_db):
    """Set sig2 relative to sig1 by target_db (positive = sig1 louder)."""
    rms1, rms2 = rms(sig1), rms(sig2)
    target_ratio = 10 ** (target_db / 20)
    sig2 = sig2 * (rms1 / (rms2 * target_ratio))
    return sig1, sig2

def mix_signals(sig1, sig2, mode):
    if mode == 'min':
        length = min(len(sig1), len(sig2))
        return sig1[:length], sig2[:length], sig1[:length] + sig2[:length]
    else:  # max
        length = max(len(sig1), len(sig2))
        s1 = np.pad(sig1, (0, length - len(sig1)), 'constant')
        s2 = np.pad(sig2, (0, length - len(sig2)), 'constant')
        return s1, s2, s1 + s2

def group_by_speaker(files):
    from collections import defaultdict
    spk_files = defaultdict(list)
    for f in files:
        spk_id = f.parent.name  # Speaker folder name
        spk_files[spk_id].append(f)
    return spk_files

def create_mixes(speech_path, output_path, n_train, n_valid, n_test, sr, mode):
    random.seed(42)

    split_map = {
        'tr': ('train', n_train),
        'cv': ('valid', n_valid),
        'tt': ('test', n_test)
    }

    def gather_files(split_folder):
        files = []
        for spk_dir in Path(split_folder).iterdir():
            if spk_dir.is_dir():
                files.extend(sorted(spk_dir.glob("*.wav")))
        return files

    for in_split, (out_split, n_mixes) in split_map.items():
        in_dir = Path(speech_path) / in_split
        files = gather_files(in_dir)
        spk_files = group_by_speaker(files)
        speakers = list(spk_files.keys())

        if len(speakers) < 2:
            raise ValueError(f"Not enough speakers in {in_dir} to create mixtures.")

        metadata_file = Path(output_path) / f"{out_split}_mixinfo.txt"
        metadata_file.parent.mkdir(parents=True, exist_ok=True)

        with open(metadata_file, "w") as mf:
            for i in range(n_mixes):
                spk1, spk2 = random.sample(speakers, 2)  # pick two different speakers
                f1 = random.choice(spk_files[spk1])
                f2 = random.choice(spk_files[spk2])

                s1 = read_audio(f1, sr)
                s2 = read_audio(f2, sr)

                rel_db = random.uniform(0, 5)
                s1, s2 = set_relative_level(s1, s2, rel_db)

                s1, s2, mix = mix_signals(s1, s2, mode)

                # Extract sentence ID (before underscore from first file)
                base_sentence_id = f1.stem.split("_")[0]
                spk1_id = spk1
                spk2_id = spk2

                # Construct filenames
                mix_filename = f"{base_sentence_id}_{spk1_id}_{spk2_id}.wav"
                s1_filename = f"{base_sentence_id}_{spk1_id}.wav"
                s2_filename = f"{base_sentence_id}_{spk2_id}.wav"

                # Output paths
                mix_path = Path(output_path) / out_split / "mix_clean" / mix_filename
                s1_path = Path(output_path) / out_split / "s1" / s1_filename
                s2_path = Path(output_path) / out_split / "s2" / s2_filename

                # Write files
                write_audio(s1_path, s1, sr)
                write_audio(s2_path, s2, sr)
                write_audio(mix_path, mix, sr)

                # Save metadata line
                mf.write(f"{mix_path} {s1_path} {s2_path} {spk1_id} {spk2_id} {rel_db:.2f}\n")

        print(f"Saved {n_mixes} mixtures and metadata to {metadata_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--speech_path", type=str, required=True,
                        help="Path to Hindi WSJ0 corpus (speaker folders with wav files)")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Where to save wsj0hin-2mix dataset")
    parser.add_argument("--n_train", type=int, default=20000)
    parser.add_argument("--n_valid", type=int, default=5000)
    parser.add_argument("--n_test", type=int, default=3000)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--mode", type=str, choices=['min', 'max'], default='min')
    args = parser.parse_args()

    create_mixes(args.speech_path, args.output_path,
                 args.n_train, args.n_valid, args.n_test,
                 args.sample_rate, args.mode)
