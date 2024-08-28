from transformers import BertForSequenceClassification, RobertaForSequenceClassification

import torch.nn as nn 

#https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py
class BERTClassification(nn.Module):
    def __init__(self,
        output_dim: int):
        super(BERTClassification, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased',num_labels=output_dim)
            
    def forward(self, ids, mask, token_type_ids):
        output = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        return output[0]

class RoBERTaClassification(nn.Module):
    def __init__(self,
        output_dim: int):
        super(RoBERTaClassification, self).__init__()
        self.bert = RobertaForSequenceClassification.from_pretrained('FacebookAI/roberta-base',num_labels=output_dim)
            
    def forward(self, ids, mask, token_type_ids):
        output = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        return output[0]
