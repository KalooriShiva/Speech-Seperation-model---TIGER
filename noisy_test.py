import os
import yaml
import torch
import torchaudio
import look2hear.models
import subprocess
import tempfile
import numpy as np
from glob import glob
from pypesq import pesq
from pystoi import stoi

def compute_snr(clean, denoised):
    """Compute Signal-to-Noise Ratio (SNR) in dB."""
    noise = clean - denoised
    signal_power = (clean ** 2).mean()
    noise_power = (noise ** 2).mean()
    if noise_power == 0:
        return float('inf')
    return 10 * np.log10(signal_power / noise_power)

def separate_audio(config_path, checkpoint_path, input_dir, output_dir, clean_dir=None):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    print("Configuration loaded successfully.")

    exp_dir = os.path.join(os.getcwd(), "Experiments", "checkpoint", config["exp"]["exp_name"])
    model_path = os.path.join(exp_dir, checkpoint_path)
    model_class = getattr(look2hear.models, config["audionet"]["audionet_name"])
    model = model_class(
        sample_rate=config["datamodule"]["data_config"]["sample_rate"],
        **config["audionet"]["audionet_config"]
    )
    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict = checkpoint.get('state_dict', checkpoint)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("audio_model."):
            new_state_dict[k[len("audio_model."):]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print("Model loaded and set to evaluation mode.")

    os.makedirs(output_dir, exist_ok=True)
    input_files = glob(os.path.join(input_dir, "*.wav")) + glob(os.path.join(input_dir, "*.mp3"))
    if not input_files:
        raise ValueError(f"No audio files found in {input_dir}")

    snr_results = []
    pesq_results = []
    stoi_results = []
    for input_audio in input_files:
        print(f"\nProcessing: {input_audio}")
        temp_wav = None
        input_wav = input_audio
        if input_audio.lower().endswith('.mp3'):
            temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
            subprocess.run(['ffmpeg', '-y', '-i', input_audio, '-ac', '1', '-ar', '16000', temp_wav], check=True)
            input_wav = temp_wav

        waveform, sample_rate = torchaudio.load(input_wav)
        target_sr = config["datamodule"]["data_config"]["sample_rate"]
        if sample_rate != target_sr:
            waveform = torchaudio.functional.resample(waveform, sample_rate, target_sr)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        waveform = waveform.to(device)

        with torch.no_grad():
            est_sources = model(waveform.unsqueeze(0))
        while est_sources.dim() > 2:
            est_sources = est_sources.squeeze(0)
        est_sources = est_sources.cpu()

        best_source = None
        best_snr = -float('inf')
        best_idx = 0
        if clean_dir:
            clean_file = os.path.join(clean_dir, os.path.basename(input_audio))
            if os.path.exists(clean_file):
                clean_wave, clean_sr = torchaudio.load(clean_file)
                if clean_sr != target_sr:
                    clean_wave = torchaudio.functional.resample(clean_wave, clean_sr, target_sr)
                if clean_wave.shape[0] > 1:
                    clean_wave = torch.mean(clean_wave, dim=0, keepdim=True)
                for i in range(est_sources.shape[0]):  # Check both sources
                    source = est_sources[i].unsqueeze(0) if est_sources[i].dim() == 1 else est_sources[i]
                    if clean_wave.shape[1] > source.shape[1]:
                        clean_wave_trunc = clean_wave[:, :source.shape[1]]
                    else:
                        clean_wave_trunc = clean_wave
                        source = source[:, :clean_wave.shape[1]]
                    snr = compute_snr(clean_wave_trunc, source)
                    if snr > best_snr:
                        best_snr = snr
                        best_source = source
                        best_idx = i
                print(f"Selected source {best_idx} with SNR: {best_snr:.2f} dB")
            else:
                print(f"No clean reference found for {input_audio}, using source 0.")
                best_source = est_sources[0].unsqueeze(0) if est_sources[0].dim() == 1 else est_sources[0]
                best_idx = 0
        else:
            best_source = est_sources[0].unsqueeze(0) if est_sources[0].dim() == 1 else est_sources[0]
            best_idx = 0

        output_file = os.path.join(output_dir, f"denoised_{os.path.basename(input_audio)}")
        torchaudio.save(output_file, best_source, target_sr)
        print(f"Saved denoised source: {output_file}, shape={best_source.shape}")

        if clean_dir and os.path.exists(clean_file):
            pesq_score = pesq(clean_wave_trunc.numpy().flatten(), best_source.numpy().flatten(), target_sr)
            stoi_score = stoi(clean_wave_trunc.numpy().flatten(), best_source.numpy().flatten(), target_sr)
            snr_results.append(best_snr)
            pesq_results.append(pesq_score)
            stoi_results.append(stoi_score)
            print(f"Metrics for {output_file}: SNR={best_snr:.2f} dB, PESQ={pesq_score:.2f}, STOI={stoi_score:.2f}")

        if temp_wav and os.path.exists(temp_wav):
            os.remove(temp_wav)

    if snr_results:
        print(f"\nAverage metrics across {len(snr_results)} files:")
        print(f"SNR: {np.mean(snr_results):.2f} dB")
        print(f"PESQ: {np.mean(pesq_results):.2f}")
        print(f"STOI: {np.mean(stoi_results):.2f}")

    print(f"\nAll denoising completed. Outputs saved to {output_dir}")

if __name__ == "__main__":
    separate_audio(
        config_path="Experiments/checkpoint/TIGER-EchoSet/conf.yml",
        checkpoint_path="epoch=110.ckpt",
        input_dir="/home/kaloori.shiva/Thesis/TIGER/Voicebank_DEMAND_test/noisy_testset_wav",
        output_dir="/home/kaloori.shiva/Thesis/TIGER/Voicebank_DEMAND_test/denoised_wav",
        clean_dir="/home/kaloori.shiva/Thesis/TIGER/Voicebank_DEMAND_test/clean_testset_wav"
    )