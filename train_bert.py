# ==============================================================================
# STEP 4: BERT TEXT CLASSIFICATION (TRANSFORMER / DEEP LEARNING)
# ==============================================================================
# This script:
# 1. Tokenizes news headlines with BERT Tokenizer (DistilBERT).
# 2. Fine-tunes pre-trained 'distilbert-base-uncased' on our 4 categories.
# 3. Evaluates test set performance (Accuracy, Macro F1-Score).
# 4. Saves classification report and Confusion Matrix heatmap.
# ==============================================================================

import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

OUTPUT_DIR = "bert_model_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "model")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------------------
# 1. LOAD DATA AND MAP LABELS TO IDS
# ------------------------------------------------------------------------------
print("Loading data...")
train_df = pd.read_csv("train_data.csv")
test_df = pd.read_csv("test_data.csv")

categories = sorted(train_df['label'].unique())
label2id = {cat: i for i, cat in enumerate(categories)}
id2label = {i: cat for i, cat in enumerate(categories)}

print(f"Categories: {label2id}")
print(f"Training Samples: {len(train_df)} | Test Samples: {len(test_df)}")

# ------------------------------------------------------------------------------
# 2. TOKENIZATION AND DATASET PREPARATION
# ------------------------------------------------------------------------------
MODEL_NAME = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def prepare_data(df):
    """
    Converts raw text headlines and labels into PyTorch TensorDataset.
    - padding=True    : Pads short sequences up to max_length=64.
    - truncation=True : Truncates sequences exceeding 64 tokens.
    """
    encodings = tokenizer(list(df['text']), padding=True, truncation=True, max_length=64, return_tensors="pt")
    labels = torch.tensor(df['label'].map(label2id).values, dtype=torch.long)
    return TensorDataset(encodings['input_ids'], encodings['attention_mask'], labels)

train_loader = DataLoader(prepare_data(train_df), batch_size=32, shuffle=True)
test_loader = DataLoader(prepare_data(test_df), batch_size=32)

# ------------------------------------------------------------------------------
# 3. LOAD OR TRAIN BERT MODEL
# ------------------------------------------------------------------------------
if os.path.exists(MODEL_SAVE_PATH):
    print(f"\n[+] Pre-trained model found at ('{MODEL_SAVE_PATH}'). Loading for evaluation...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_SAVE_PATH).to(device)
else:
    print("\nTraining model from scratch (2 Epochs)...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(categories), id2label=id2label, label2id=label2id
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=3e-5)

    for epoch in range(2):
        model.train()
        for step, (ids, mask, y) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(ids.to(device), attention_mask=mask.to(device), labels=y.to(device))
            outputs.loss.backward()
            optimizer.step()

            if (step + 1) % 25 == 0 or (step + 1) == len(train_loader):
                print(f"Epoch {epoch+1}/2 - Step {step+1}/{len(train_loader)} - Loss: {outputs.loss.item():.4f}")

    # Save fine-tuned model and tokenizer
    model.save_pretrained(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)

# ------------------------------------------------------------------------------
# 4. EVALUATE ON TEST SET
# ------------------------------------------------------------------------------
print("\n" + "="*50)
print("             EVALUATING BERT MODEL")
print("="*50)
model.eval()
predictions, true_labels = [], []

with torch.no_grad():
    for ids, mask, y in test_loader:
        outputs = model(ids.to(device), attention_mask=mask.to(device))
        preds = torch.argmax(outputs.logits, dim=1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(y.numpy())

acc = accuracy_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions, average='macro')
print(f"BERT RESULTS -> Accuracy: {acc*100:.2f}% | F1-Score: {f1*100:.2f}%")

# ------------------------------------------------------------------------------
# 5. SAVE REPORTS AND CONFUSION MATRIX PLOT
# ------------------------------------------------------------------------------
# Detailed Text Report
report_text = f"""==================================================
DEEP LEARNING / TRANSFORMER MODEL: BERT (DISTILBERT)
==================================================
Categories : {', '.join(categories)}
Accuracy   : {acc*100:.2f}%
F1-Score   : {f1*100:.2f}%

Classification Report:
{classification_report(true_labels, predictions, target_names=categories)}

Confusion Matrix (Numerical):
{confusion_matrix(true_labels, predictions)}
"""
with open(os.path.join(OUTPUT_DIR, "bert_report.txt"), "w", encoding="utf-8") as f:
    f.write(report_text)

# Summary CSV
pd.DataFrame([{
    "Model": "BERT (DistilBERT)",
    "Method": "Contextual Embedding (Transformer)",
    "Accuracy": round(acc, 4),
    "F1_Score": round(f1, 4)
}]).to_csv(os.path.join(OUTPUT_DIR, "bert_summary.csv"), index=False, encoding="utf-8-sig")

# Confusion Matrix Heatmap
cm = confusion_matrix(true_labels, predictions)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=categories, yticklabels=categories)
plt.title(f"Confusion Matrix - BERT (F1: {f1*100:.2f}%)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_bert.png"), dpi=300)
plt.close()

print("\n" + "="*50)
print(f" All outputs saved to '{OUTPUT_DIR}':")
print(f"  * {os.path.join(OUTPUT_DIR, 'bert_report.txt')}")
print(f"  * {os.path.join(OUTPUT_DIR, 'confusion_matrix_bert.png')}")
print("="*50)
