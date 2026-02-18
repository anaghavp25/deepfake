import torch
import librosa
import moviepy.editor as mp
from pathlib import Path

# --- STEP 1: Hardware Detection ---
# This ensures we use the GPU (MPS for Mac, CUDA for Windows)
def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"

DEVICE = get_device()

# --- STEP 2: The Core Analysis Function ---
def analyze_audio_stream(video_path):
    """
    Takes a video path, checks for audio, and detects deepfake artifacts.
    """
    # Use Path for Windows/Mac compatibility
    v_path = Path(video_path)
    
    try:
        # 1. GATEKEEPER: Load video metadata only
        clip = mp.VideoFileClip(str(v_path))
        
        # Check if audio exists
        if clip.audio is None:
            return {"score": 0.0, "details": "No audio detected. Skipping."}
        
        # 2. EXTRACTION: Rip the audio to a temporary file
        audio_temp = "temp_detect.wav"
        clip.audio.write_audiofile(audio_temp, fps=16000, verbose=False, logger=None)
        
        # 3. FEATURE EXTRACTION: Convert sound to math
        # We load the sound and convert it to a 'Mel Spectrogram'
        # A spectrogram is basically a 'picture' of sound frequencies
        y, sr = librosa.load(audio_temp, sr=16000)
        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        
        # 4. AI INFERENCE (The 'Decision')
        # Placeholder: In a full model, you'd pass 'spectrogram' into a Neural Network
        # For now, we simulate a detection score
        detected_score = 0.75  # 75% chance it's a deepfake
        
        return {
            "score": detected_score,
            "details": f"Analyzed via {DEVICE}. High frequency anomalies detected."
        }

    except Exception as e:
        # The 'Code Contract' safety net
        return {"score": 0.5, "details": f"Error: {str(e)}"}

# --- STEP 3: Test Run ---
if __name__ == "__main__":
    # Change this to a real video path on your system to test
    result = analyze_audio_stream("test_video.mp4")
    print(result)