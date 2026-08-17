
import os

# Paths relative to the project root (one level up from src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
input_file = os.path.join(DATA_DIR, "cleaned_transcripts.txt")
output_dir = os.path.join(DATA_DIR, "processed_data")

os.makedirs(output_dir, exist_ok=True)

# Read all text
with open(input_file, 'r', encoding='utf-8') as f:
    text = f.read()

print(f"Total characters: {len(text):,}")
print(f"Total words: {len(text.split()):,}")

# Split into train/val (90/10)
split_idx = int(len(text) * 0.9)
train_text = text[:split_idx]
val_text = text[split_idx:]

# Save splits
with open(os.path.join(output_dir, "train.txt"), 'w', encoding='utf-8') as f:
    f.write(train_text)

with open(os.path.join(output_dir, "val.txt"), 'w', encoding='utf-8') as f:
    f.write(val_text)

print(f"Train characters: {len(train_text):,}")
print(f"Val characters: {len(val_text):,}")

# Also create a simple character-level vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Vocabulary size (characters): {vocab_size}")

# Save vocabulary
with open(os.path.join(output_dir, "vocab.txt"), 'w', encoding='utf-8') as f:
    for ch in chars:
        f.write(ch + '\n')

# Character to index mappings
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

import json
with open(os.path.join(output_dir, "stoi.json"), 'w') as f:
    json.dump(stoi, f, ensure_ascii=False)

with open(os.path.join(output_dir, "itos.json"), 'w') as f:
    json.dump(itos, f, ensure_ascii=False)

print(f"\nData prepared in {output_dir}/")
print("Files created:")
print("  - train.txt (training data)")
print("  - val.txt (validation data)")
print("  - vocab.txt (character vocabulary)")
print("  - stoi.json (char to index mapping)")
print("  - itos.json (index to char mapping)")
