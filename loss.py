import torch.nn as nn
import torch.nn.functional as F
from utils.utils import sequence_mask


class L1LossMasked(nn.Module):

    def forward(self, x, target, length):
        target.requires_grad = False
        mask = sequence_mask(sequence_length=length,
                             maxlen=target.size(1)).unsqueeze(2).float()
        mask = mask.expand_as(x)
        loss = F.l1_loss(x * mask, target * mask, reduction='sum')
        loss = loss / mask.sum()
        return loss
