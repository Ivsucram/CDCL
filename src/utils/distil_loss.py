import torch
import torch.nn as nn
import torch.nn.functional as F

class DistilLoss(nn.Module):
    def __init__(self):
        super(DistilLoss, self).__init__()
    
    def forward(self, x, target):
        assert x.shape[0] == target.shape[0]
        target = F.softmax(target, dim=-1)    
        loss = torch.sum( -target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()