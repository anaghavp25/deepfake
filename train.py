import os
import torch
import pandas as pd
from datasets import Dataset, Audio
from transformers import (
    Wav2Vec2ForSequenceClassification, 
    Wav2Vec2FeatureExtractor, 
    TrainingArguments, 
    Trainer
)

# --- MORPHEUS AI PATH CONFIG ---
# DO NOT USE DATASET-balanced.csv (it only has MFCC math, not audio paths)
BASE_DIR = r"C:\Users\LENOVO\Desktop\MiniProj"
CSV_FILE = os.path.join(BASE_DIR, "processed_data", "metadata_thinkpad.csv")
AUDIO_DIR = os.path.join(BASE_DIR, "processed_data")

# 1. Load the map we made during preprocessing
if not os.path.exists(CSV_FILE):
    print(f"❌ ERROR: {CSV_FILE} not found! Did you run preprocess.py first?")
else:
    df = pd.read_csv(CSV_FILE)
    # The 'relative_path' column exists in the metadata_thinkpad.csv we built
    # Let's map it to the full address on your ThinkPad
    df['path'] = df['relative_path'].apply(lambda x: os.path.join(AUDIO_DIR, x))
    print("✅ SUCCESS: Found your audio chunks. Mapping complete.")

# 2. Now cast to Audio (this will work because 'path' now exists in df)
dataset = Dataset.from_pandas(df).cast_column("path", Audio(sampling_rate=16000))
# Print columns so you can see them in the terminal (helpful for debugging!)
print(f"Detected columns: {df.columns.tolist()}")

# Identify which column has the audio paths. 
# Usually, in Kaggle datasets, it's 'filename' or 'path'.
# Let's rename it to 'path' to stay consistent.
if 'filename' in df.columns:
    df = df.rename(columns={'filename': 'path'})
elif 'audio_path' in df.columns:
    df = df.rename(columns={'audio_path': 'path'})

# Now ensure the path is the FULL Windows path
# This assumes your audio files are inside a folder named 'AUDIO' in your KAGGLE folder
df['path'] = df['path'].apply(lambda x: os.path.join(r"C:\Users\LENOVO\Desktop\MiniProj\KAGGLE", x))

# NOW this line will work without the pyarrow error!
dataset = Dataset.from_pandas(df).cast_column("path", Audio(sampling_rate=16000))
dataset = dataset.train_test_split(test_size=0.2) # 80/20 Train-Test Split

# 3. PREPROCESSING FUNCTION (Required for Wav2Vec2)
# This converts raw audio arrays into the tensors the model needs
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")

def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["path"]]
    inputs = feature_extractor(
        audio_arrays, 
        sampling_rate=feature_extractor.sampling_rate, 
        max_length=16000 * 4, # 4 seconds
        truncation=True,
        padding="max_length"
    )
    return inputs

# Apply the mapping
encoded_dataset = dataset.map(preprocess_function, remove_columns=["path"], batched=True)

# 4. LOAD MORPHEUS AI MODEL
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "facebook/wav2vec2-base", 
    num_labels=2
)

# 5. TRAINING SETTINGS (ThinkPad Optimized)
training_args = TrainingArguments(
    output_dir="./sentinel_model_checkpoints",
    per_device_train_batch_size=2,   # Keeping it low for ThinkPad RAM
    gradient_accumulation_steps=8,  # Simulates a batch of 16
    num_train_epochs=3,
    learning_rate=3e-5,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    # Use CPU if you don't have an NVIDIA GPU set up
    use_cpu=False if torch.cuda.is_available() else True 
)

# 6. INITIALIZE TRAINER
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    tokenizer=feature_extractor,
)

# 7. EXECUTE TRAINING
print("Starting Morpheus AI Training...")
trainer.train()