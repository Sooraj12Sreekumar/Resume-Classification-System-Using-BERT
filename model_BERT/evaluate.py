import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from torch.utils.data import DataLoader
from model_BERT import config, utils
import joblib
from tqdm import tqdm
import os
import pandas as pd

LABEL_ENCODER_PATH = r"D:\Resume Classification System\model_BERT\checkpoints\label_encoder.pkl" 
label_encoder = joblib.load(LABEL_ENCODER_PATH)

def load_data_with_encoder(file_path, encoder):
    df = pd.read_csv(file_path)
    labels = encoder.transform(df['Category'])
    texts = df['Cleaned_Text'].values
    return texts, labels

X_val, y_val = load_data_with_encoder(config.VAL_FILE, label_encoder)

tokenizer = DistilBertTokenizer.from_pretrained(config.TOKENIZER_SAVE_PATH)
model = DistilBertForSequenceClassification.from_pretrained(config.MODEL_SAVE_PATH)

val_dataset = utils.ResumeDataset(list(X_val), y_val)
val_loader = DataLoader(val_dataset, batch_size=16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

all_preds, all_labels = [], []

with torch.no_grad():
    for batch in tqdm(val_loader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

predicted_classes = label_encoder.inverse_transform(all_preds)
true_classes = label_encoder.inverse_transform(all_labels)

print("\nEvaluation Results:")
print(f"Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
print(f"F1-score: {f1_score(all_labels, all_preds, average='weighted'):.4f}")
print("\nClassification Report:\n", 
      classification_report(true_classes, predicted_classes, target_names=label_encoder.classes_))