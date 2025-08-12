import os
import soundfile as sf

audio_folder = "/home/kaloori.shiva/Thesis/TIGER/openSLR_hin_dataset/train"

durations = []

for root, _, files in os.walk(audio_folder):
    for file in files:
        if file.lower().endswith(('.wav', '.flac', '.ogg')):
            filepath = os.path.join(root, file)
            try:
                with sf.SoundFile(filepath) as f:
                    duration = len(f) / f.samplerate
                    durations.append((duration, filepath))
            except Exception as e:
                print(f'Could not process {filepath}: {e}')

if durations:
    # Count files close to each integer second from 1 to 15 (±0.5s)
    counts = {i: 0 for i in range(1, 16)}
    for duration, _ in durations:
        for i in range(1, 16):
            if abs(duration - i) < 0.5:
                counts[i] += 1
    for i in range(1, 16):
        print(f"Number of files with duration close to {i} second(s): {counts[i]}")
    min_duration, min_file = min(durations, key=lambda x: x[0])
    max_duration, max_file = max(durations, key=lambda x: x[0])
    print(f"\nMinimum duration: {min_duration:.2f} seconds ({min_file})")
    print(f"Maximum duration: {max_duration:.2f} seconds ({max_file})")
else:
    print("No audio files found.")