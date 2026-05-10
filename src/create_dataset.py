from datasets import load_dataset
import pandas as pd
from huggingface_hub import HfApi

# Load the famous IMDB dataset
print("Loading IMDB dataset...")
dataset = load_dataset("imdb")

# Convert to pandas to process it
train_df = pd.DataFrame(dataset["train"])
test_df = pd.DataFrame(dataset["test"])

# --- YOUR CUSTOM PROCESSING (this is what makes it "your" dataset) ---
# Add text length as a feature column
train_df["text_length"] = train_df["text"].apply(len)
test_df["text_length"] = test_df["text"].apply(len)

# Filter: only keep reviews under 1000 characters (short, punchy reviews)
# This is your creative processing step for VG!
train_df = train_df[train_df["text_length"] <= 1000].reset_index(drop=True)
test_df = test_df[test_df["text_length"] <= 1000].reset_index(drop=True)

# Rename label column to be clear (0=negative, 1=positive)
train_df["sentiment"] = train_df["label"].map({0: "negative", 1: "positive"})
test_df["sentiment"] = test_df["label"].map({0: "negative", 1: "positive"})

print(f"Train size: {len(train_df)} reviews")
print(f"Test size: {len(test_df)} reviews")
print(train_df["sentiment"].value_counts())

# Save locally first
train_df.to_csv("./data/train.csv", index=False)
test_df.to_csv("./data/test.csv", index=False)
print("Saved locally!")

# Upload to Hugging Face
from datasets import DatasetDict, Dataset

hf_dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df),
    "test": Dataset.from_pandas(test_df)
})

# CHANGE "medhuuu" to your HF username if different
hf_dataset.push_to_hub("medhuuu/movie-sentiment-dataset")
print("✅ Dataset uploaded to Hugging Face!")