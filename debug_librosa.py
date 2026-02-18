import librosa
import os

try:
    files = os.listdir("KAGGLE/AUDIO/REAL")
    if not files:
        print("No files found in KAGGLE/AUDIO/REAL")
        exit()
        
    f = os.path.join("KAGGLE/AUDIO/REAL", files[0])
    print(f"Trying to load: {f}")
    y, sr = librosa.load(f, sr=16000, duration=10) # Load only 10 seconds
    print(f"Loaded successfully! Shape: {y.shape}, SR: {sr}")
except Exception as e:
    print(f"Error: {e}")
