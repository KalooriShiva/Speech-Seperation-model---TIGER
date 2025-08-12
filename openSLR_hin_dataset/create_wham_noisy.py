#!/usr/bin/env python3
import argparse
import random
from pathlib import Path
import numpy as np
import soundfile as sf


def read_audio(path, sr):
    audio, fs = sf.read(path)
    if audio.ndim > 1:  # convert to mono
        audio = np.mean(audio, axis=1)
    if fs != sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=fs, target_sr=sr)
    return audio


def write_audio(path, audio, sr):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sr)


def rms(x):
    return np.sqrt(np.mean(x ** 2))


def match_length(sig, target_len):
    """Crop or pad signal to match target length."""
    if len(sig) > target_len:
        return sig[:target_len]
    elif len(sig) < target_len:
        return np.pad(sig, (0, target_len - len(sig)), 'constant')
    return sig


def create_noisy_mixes(clean_root, noise_root, output_root, sr):
    random.seed(42)

    # WHAM! official SNR range (in dB)
    snr_min, snr_max = -6, 3

    for split in ["tr", "cv", "tt"]:
        clean_mix_dir = Path(clean_root) / split / "mix_clean"
        noise_dir = Path(noise_root) / split

        clean_files = sorted(clean_mix_dir.glob("*.wav"))
        noise_files = sorted(noise_dir.glob("*.wav"))

        if not clean_files:
            raise RuntimeError(f"No clean mixture files found in {clean_mix_dir}")
        if not noise_files:
            raise RuntimeError(f"No noise files found in {noise_dir}")

        for i, clean_path in enumerate(clean_files):
            clean = read_audio(clean_path, sr)
            noise_path = noise_files[i % len(noise_files)]
            noise = read_audio(noise_path, sr)

            # Random noise offset: up to 2s pre-padding, 2s post-padding
            max_offset = int(2 * sr)
            pre_pad = random.randint(0, max_offset)
            post_pad = random.randint(0, max_offset)
            noise = np.pad(noise, (pre_pad, post_pad), 'constant')

            # Match noise length to clean mixture
            noise = match_length(noise, len(clean))

            # Random SNR between -6 and +3 dB
            snr_db = random.uniform(snr_min, snr_max)
            clean_rms = rms(clean)
            noise_rms = rms(noise)
            if noise_rms > 0:
                noise = noise * (clean_rms / (10 ** (snr_db / 20) * noise_rms))

            # Create noisy mixture
            noisy_mix = clean + noise

            base_name = clean_path.name
            write_audio(Path(output_root) / split / "mix_noisy" / base_name, noisy_mix, sr)
            write_audio(Path(output_root) / split / "noise" / base_name, noise, sr)

            if i % 500 == 0:
                print(f"[{split}] Processed {i}/{len(clean_files)} mixtures")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_root", type=str, required=True,
                        help="Path to clean 2mix dataset root (with tr/cv/tt/mix_clean)")
    parser.add_argument("--noise_root", type=str, required=True,
                        help="Path to WHAM! noise root (with tr/cv/tt noise wavs)")
    parser.add_argument("--output_root", type=str, required=True,
                        help="Where to save WHAM!-style noisy mixtures")
    parser.add_argument("--sample_rate", type=int, default=16000)
    args = parser.parse_args()

    create_noisy_mixes(args.clean_root, args.noise_root, args.output_root, args.sample_rate)
