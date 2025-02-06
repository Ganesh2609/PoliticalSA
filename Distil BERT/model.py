import torch 
from torch import nn 
from transformers import AutoModelForSequenceClassification


class PSAModel(nn.Module):

    def __init__(self, num_classes:int=7):
        super(PSAModel, self).__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_classes)

    def forward(self, x, mask=None):
        x = self.bert(x, attention_mask=mask).logits
        return x