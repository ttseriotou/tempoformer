import torch
import torch.nn as nn
import torch.nn.functional as F
from models.rope_mha import MHARoPE

class GatedConnection(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.w = nn.Linear(d_model*2, d_model, True)
    
    def forward(self, t1, t2):
        g = F.sigmoid(self.w(torch.cat([t1, t2], -1)))
        return g*t1 + (1-g)*t2
