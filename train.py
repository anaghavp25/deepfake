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

# --- 1. PREVENT CRASHES ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# --- 2. PATHS ---
BASE_DIR = r"C:\Users\LENOVO\Desktop\MiniProj"
DATA_DIR = os.path.join(BASE_DIR, "processed_data")
CSV_FILE = os.path.join(DATA_DIR, "metadata.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "morpheus_model_results")

def train_morpheus():
    # --- 3. DATA SERIALIZATION (PHASE 1) ---
    if not os.path.exists(CSV_FILE):
        print(f"❌ ERROR: Missing {CSV_FILE}")
        return

    print("✅ Loading metadata...")
    df = pd.read_csv(CSV_FILE)
    
    # Clean column detection
    path_col = next((c for c in ['path', 'relative_path', 'file_name'] if c in df.columns), None)
    label_col = next((c for c in ['label', 'LABEL', 'class'] if c in df.columns), None)

    # Convert to Python Native Lists (The "Large_String" Fix)
    # We force them into basic strings and integers here
    audio_list = [os.path.join(DATA_DIR, str(p)) for p in df[path_col].tolist()]
    
    label_names = sorted(df[label_col].unique())
    label_to_id = {label: i for i, label in enumerate(label_names)}
    print(f"🔎 Mapping: {label_to_id}")
    labels_list = [int(label_to_id[l]) for l in df[label_col].tolist()]

    # Create Dataset from Dictionary to bypass PyArrow errors
    raw_dataset = Dataset.from_dict({"audio": audio_list, "label": labels_list})
    raw_dataset = raw_dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    dataset = raw_dataset.train_test_split(test_size=0.2)

    # --- 4. FEATURE EXTRACTION (PHASE 2) ---
    print("Step 2/4: Initializing Feature Extractor...")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")

    def preprocess_function(examples):
        # Turns sound waves into normalized tensors
        audio_arrays = [x["array"] for x in examples["audio"]]
        return feature_extractor(
            audio_arrays, 
            sampling_rate=16000, 
            max_length=80000, # 5-second window to save Vostro RAM
            truncation=True,
            padding="max_length"
        )

    print("Step 3/4: Mapping audio to tensors...")
    encoded_dataset = dataset.map(
        preprocess_function, 
        remove_columns=["audio"], 
        batched=True, 
        batch_size=2 
    )

    # --- 5. MODEL & TRAINING (PHASES 3, 4, 5) ---
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        "facebook/wav2vec2-base", 
        num_labels=len(label_to_id),
        id2label={i: l for l, i in label_to_id.items()},
        label2id=label_to_id
    )

    # Vostro-Optimized Training Args
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,    # One sample at a time to prevent RAM spikes
        gradient_accumulation_steps=8,   # Backpropagation happens every 8 steps
        num_train_epochs=3,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        learning_rate=3e-5,
        fp16=False,                      # Disable half-precision for CPU stability
        save_total_limit=2,              # Only keep last 2 checkpoints to save disk space
        report_to="none"
    )

    # The Trainer handles Forward pass, Loss calculation, and Optimization
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["test"],
        tokenizer=feature_extractor,
    )

    print("Step 4/4: Commencing Training Engine (This will be slow on CPU)...")
    trainer.train()
    
    # Save the Final Weightage
    final_path = os.path.join(BASE_DIR, "morpheus_final_model")
    trainer.save_model(final_path)
    print(f"🎉 SUCCESS! Morpheus AI weights saved at: {final_path}")

if __name__ == "__main__":
    train_morpheus()