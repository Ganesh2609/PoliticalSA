import torch 
from torch import nn 
from transformers import AutoTokenizer, AutoModelForMaskedLM


class PSAModel(nn.Module):

    def __init__(self, num_classes:int=7):
        super(PSAModel, self).__init__()
        self.bert = AutoModelForMaskedLM.from_pretrained("l3cube-pune/tamil-bert")
        self.final_fc = nn.Linear(in_features=197285, out_features=num_classes, bias=True)

    def forward(self, x, mask=None):
        if mask is not None:
            x = self.bert(x, attention_mask=mask).logits
        else:
            x = self.bert(x).logits
        x = torch.mean(x, dim=1)
        x = self.final_fc(x)
        return x