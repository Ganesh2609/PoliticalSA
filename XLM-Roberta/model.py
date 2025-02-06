import torch 
from torch import nn 
from transformers import XLMRobertaModel


class PSAModel(nn.Module):

    def __init__(self, num_classes:int=7):
        super(PSAModel, self).__init__()
        self.bert = XLMRobertaModel.from_pretrained("FacebookAI/xlm-roberta-base")
        self.final_fc = nn.Linear(in_features=768, out_features=num_classes, bias=True)

    def forward(self, x, mask=None):
        if mask is not None:
            x = self.bert(x, attention_mask=mask).last_hidden_state
        else:
            x = self.bert(x).last_hidden_state
        x = torch.mean(x, dim=1)
        x = self.final_fc(x)
        return x