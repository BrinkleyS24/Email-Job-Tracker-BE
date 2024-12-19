# Updated train_model.py

import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, get_scheduler
from torch.utils.data import DataLoader, Dataset
import re
from html import unescape

# Paths
DATASET_PATH = '../email_dataset.json'
MODEL_DIR = '../model/distilbert_email_classifier'
os.makedirs(MODEL_DIR, exist_ok=True)

# Load Dataset
df = pd.read_json(DATASET_PATH)

# Preprocessing Function
def preprocess_email(subject, body):
    content = f"{subject} {body}".lower()
    content = unescape(content)
    content = re.sub(r"<[^>]+>", "", content)  # Remove HTML tags
    content = re.sub(r"http\S+|www\S+", "", content)  # Remove URLs
    content = re.sub(r"[^a-zA-Z0-9\s.,!?]", "", content)  # Remove special characters
    content = re.sub(r"\s+", " ", content).strip()  # Normalize whitespace
    return content

# Apply Preprocessing
df['content'] = df.apply(lambda x: preprocess_email(x['subject'], x['body']), axis=1)

# Label Mapping
label_mapping = {'Applied': 0, 'Interviewed': 1, 'Offers': 2, 'Rejected': 3, 'Irrelevant': 4}
df['label'] = df['category'].map(label_mapping)

# Train/Test Split (Stratified)
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['content'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

# Tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Dataset Class
class EmailDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = tokenizer(
            self.texts.iloc[idx],
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        label = torch.tensor(self.labels.iloc[idx], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": label}

# Prepare Datasets and Dataloaders
train_dataset = EmailDataset(train_texts, train_labels)
test_dataset = EmailDataset(test_texts, test_labels)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# Model Initialization
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(label_mapping))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Optimizer and Scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
num_training_steps = len(train_loader) * 5  # Assuming 5 epochs
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# Class Weights
class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

# Training Loop
def train_model(model, train_loader):
    model.train()
    for epoch in range(5):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/5 - Loss: {total_loss:.4f}")

# Evaluation Function
def evaluate_model(model, test_loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    report = classification_report(all_labels, all_preds, target_names=list(label_mapping.keys()))
    print("\nClassification Report:\n", report)

# Train and Evaluate
train_model(model, train_loader)
evaluate_model(model, test_loader)

# Save Model and Tokenizer
model.save_pretrained(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)
print("Model and tokenizer saved successfully.")
