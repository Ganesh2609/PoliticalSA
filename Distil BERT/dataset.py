import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import re
import emoji


class PoliticalSentimentAnalysis(Dataset):

    def __init__(self, root:str):
        self.data = pd.read_csv(root)
        self.classes = ['Opinionated', 'Substantiated', 'Neutral', 'Negative', 'Sarcastic', 'None of the above', 'Positive']
        self.class_to_idx = {label: i for i, label in enumerate(self.classes)}
        self.tokenizer = AutoTokenizer.from_pretrained("ai4bharat/IndicBERTv2-MLM-Sam-TLM")
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        tokenized = self.tokenizer(self.preprocess(self.data.loc[idx, 'content']), max_length=512, truncation=True, padding="max_length", return_tensors="pt")
        #tokenized = self.tokenizer(self.data.loc[idx, 'content'], max_length=512, truncation=True, padding="max_length", return_tensors="pt")
        text = tokenized['input_ids'].squeeze(dim=0)
        mask = tokenized['attention_mask'].squeeze(dim=0)
        label = self.class_to_idx[self.data.loc[idx, 'labels']]
        return {
            'text': text,
            'mask' : mask,
            'label': label
        }
    
    def preprocess(self, text):
        text = re.sub(r"#\S+|@\S+|https?:\/\/\S+", "", text)
        text = re.sub(r"[^அ-ஹ0-9\s]", "", text)
        text = re.sub(r"[\n\r\t]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"(.)\1{2,}", r"\1\1", text)
        return text


def get_data_loaders(train_dir:str, test_dir:str, batch_size:int, num_workers:int=12, prefetch_factor:int=2, seed:int=42):
    
    torch.manual_seed(seed)

    train_dataset = PoliticalSentimentAnalysis(train_dir)
    val_dataset = PoliticalSentimentAnalysis(test_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=prefetch_factor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=prefetch_factor)

    return train_loader, val_loader