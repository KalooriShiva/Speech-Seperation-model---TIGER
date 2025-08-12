import os
import yaml
import torch
import torchaudio
import look2hear.models
import subprocess
import tempfile

def separate_audio(config_path, checkpoint_path, input_audio, output_dir):
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    print("Configuration loaded successfully.")

    # Handle MP3 to WAV conversion if necessary
    input_wav = input_audio
    temp_wav = None
    if input_audio.lower().endswith('.mp3'):
        print("Input is MP3, converting to WAV.")
        # Create a temporary WAV file
        temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
        try:
            # Convert MP3 to WAV using ffmpeg
            subprocess.run([
                'ffmpeg', '-y', '-i', input_audio, '-ac', '1', '-ar', '16000', temp_wav
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            input_wav = temp_wav
            print(f"Converted MP3 to WAV: {input_wav}")
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg error: {e.stderr.decode()}")
            raise RuntimeError("Failed to convert MP3 to WAV using ffmpeg.")
        except FileNotFoundError:
            raise RuntimeError("FFmpeg not found. Please install ffmpeg to process MP3 files.")

    # Model setup
    exp_dir = os.path.join(
        os.getcwd(), "Experiments", "checkpoint", config["exp"]["exp_name"]
    )
    print(f"Experiment directory: {exp_dir}")
    model_path = os.path.join(exp_dir, checkpoint_path)
    model_class = getattr(look2hear.models, config["audionet"]["audionet_name"])
    model = model_class(
        sample_rate=config["datamodule"]["data_config"]["sample_rate"],
        **config["audionet"]["audionet_config"]
    )
    print("Model initialized successfully.")

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

    # Load audio
    waveform, sample_rate = torchaudio.load(input_wav)
    print(f"Loaded input audio: shape={waveform.shape}, dtype={waveform.dtype}, sample_rate={sample_rate}")
    target_sr = config["datamodule"]["data_config"]["sample_rate"]
    if sample_rate != target_sr:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sr)
        print(f"Resampled audio to {target_sr} Hz.")

    # Convert to mono if necessary
    if waveform.shape[0] > 1:
        print("Input is stereo, converting to mono.")
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    waveform = waveform.to(device)
    print(f"Input audio tensor shape after resampling and mono conversion (if any): {waveform.shape}")

    # Separate sources
    with torch.no_grad():
        est_sources = model(waveform.unsqueeze(0))  # (batch, sources, samples) or (batch, channels, samples)
    print(f"Raw model output shape: {est_sources.shape}")

    # Remove all batch dimensions; ensure output is (sources, samples)
    while est_sources.dim() > 2:
        est_sources = est_sources.squeeze(0)
    est_sources = est_sources.cpu()
    print(f"Separated sources shape after squeeze: {est_sources.shape}")

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    for i in range(est_sources.shape[0]):
        source = est_sources[i]
        if source.dim() == 1:
            source = source.unsqueeze(0)  # (1, samples)
        print(f"Saving source {i+1}: shape={source.shape}")
        torchaudio.save(
            os.path.join(output_dir, f"s{i+1}.wav"),
            source,
            target_sr
        )

    # Clean up temporary WAV file if created
    if temp_wav and os.path.exists(temp_wav):
        os.remove(temp_wav)
        print(f"Temporary WAV file {temp_wav} deleted.")

    print(f"Separated sources saved to {output_dir}.")

if __name__ == "__main__":
    separate_audio(
        config_path="Experiments/checkpoint/TIGER-EchoSet/conf.yml",
        checkpoint_path="epoch=110.ckpt",  # Ensure this checkpoint is trained with num_sources=1
        input_audio="/home/kaloori.shiva/Thesis/TIGER/custom_results/bbc_news_debate.mp3",  # Replace with noisy audio
        output_dir="/home/kaloori.shiva/Thesis/TIGER/custom_results/bbc_news_debate"  # Replace with desired output directory
    )