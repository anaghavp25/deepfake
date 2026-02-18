import os
import librosa
import soundfile as sf
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time

# --- CONFIGURATION ---
TARGET_SR = 16000
CHUNK_DURATION = 4 
SAMPLES_PER_CHUNK = TARGET_SR * CHUNK_DURATION

def preprocess(input_dir, output_dir):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    metadata = []

    for label in ["REAL", "FAKE"]:
        source_folder = input_path / "AUDIO" / label
        if not source_folder.exists(): continue
            
        dest_folder = output_path / label
        dest_folder.mkdir(exist_ok=True)
        
        print(f"\n--- Processing {label} ---")
        files = list(source_folder.glob("*.wav"))
        
        for idx, audio_file in enumerate(tqdm(files)):
            try:
                # Load and Resample
                y, _ = librosa.load(audio_file, sr=TARGET_SR, mono=True)
                y = librosa.util.normalize(y)
                
                for i, start in enumerate(range(0, len(y), SAMPLES_PER_CHUNK)):
                    chunk = y[start : start + SAMPLES_PER_CHUNK]
                    if len(chunk) < SAMPLES_PER_CHUNK:
                        chunk = np.pad(chunk, (0, SAMPLES_PER_CHUNK - len(chunk)))
                    
                    name = f"{audio_file.stem}_c{i}.flac"
                    sf.write(dest_folder / name, chunk, TARGET_SR)
                    metadata.append({"path": f"{label}/{name}", "label": 0 if label == "REAL" else 1})

                # THERMAL SAFETY: Every 100 files, rest for 5 seconds
                if idx % 100 == 0 and idx > 0:
                    time.sleep(5) 
                    
            except Exception as e:
                print(f"Error: {e}")

    pd.DataFrame(metadata).to_csv(output_path / "metadata.csv", index=False)
    print("Done!")

if __name__ == "__main__":
    # Point these to your ThinkPad folders
    preprocess("./KAGGLE", "./processed_data")