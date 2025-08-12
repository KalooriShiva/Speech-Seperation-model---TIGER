import os
import shutil
import random
from tqdm import tqdm

random.seed(42)  # for reproducibility

# Your current full train directory
original_dir = "/home/kaloori.shiva/Thesis/TIGER/DataPreProcess/EchoSet/train"
base_dir = "/home/kaloori.shiva/Thesis/TIGER/DataPreProcess/EchoSet"

# Create val/ and test/ directories
val_dir = os.path.join(base_dir, "val")
test_dir = os.path.join(base_dir, "test")
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Collect all mix files
mix_files = sorted([f for f in os.listdir(original_dir) if f.startswith("mix") and f.endswith(".wav")])
total = len(mix_files)
val_count = int(0.10 * total)
test_count = int(0.10 * total)

# Shuffle and split
random.shuffle(mix_files)
val_files = mix_files[:val_count]
test_files = mix_files[val_count:val_count + test_count]

def move_files(file_list, destination):
    for mix_file in tqdm(file_list, desc=f"Moving files to {destination}"):
        index = mix_file.split("_")[1].split(".")[0]  # e.g., '00001'
        for tag in ["mix", "spk1_reverb", "spk2_reverb"]:
            src_file = os.path.join(original_dir, f"{tag}_{index}.wav")
            dst_file = os.path.join(destination, f"{tag}_{index}.wav")
            if os.path.exists(src_file):
                shutil.move(src_file, dst_file)
            else:
                print(f"Warning: {src_file} not found!")

# Move files to val/ and test/
move_files(val_files, val_dir)
move_files(test_files, test_dir)

print(f"Splitting complete. Total: {total}, Train left: {total - val_count - test_count}, Val: {val_count}, Test: {test_count}")
