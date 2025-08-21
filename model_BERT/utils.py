import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from model_BERT import config
from transformers import DistilBertTokenizer  

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
label_encoder = LabelEncoder()

def load_data(file_path, fit_label_encoder=False):
    """
    Load and preprocess data
    Args:
        file_path: Path to CSV file
        fit_label_encoder: Whether to fit new label encoder
    Returns:
        texts: List of cleaned text samples
        labels: Encoded labels
    """
    try:
        df = pd.read_csv(file_path)
        
        required_columns = ['Category', 'Cleaned_Text']
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            raise ValueError(f"Dataset missing required columns: {missing}")

        if fit_label_encoder:
            labels = label_encoder.fit_transform(df['Category'])
        else:
            labels = label_encoder.transform(df['Category'])
            
        return df['Cleaned_Text'].tolist(), labels
        
    except Exception as e:
        raise ValueError(f"Error loading data: {str(e)}")

class ResumeDataset(Dataset):
    def __init__(self, texts, labels):
        """
        PyTorch Dataset for resume classification
        Args:
            texts: List of text samples
            labels: List of corresponding labels
        """
        self.texts = texts
        self.labels = labels
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=config.MAX_LEN,
            return_tensors="pt"
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

    @staticmethod
    def get_label_mapping():
        """Get mapping of encoded labels to original categories"""
        return {i: label for i, label in enumerate(label_encoder.classes_)}