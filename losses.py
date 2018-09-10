import torch
import torch.nn as nn


class ImportanceWeightedBCELoss(nn.Module):

    def __init__(self):
        super(ImportanceWeightedBCELoss, self).__init__()
        self.loss = 0.0

    def forward(self, reconstructed, original):
        batch_size = original.size(0)
        I = torch.abs(original - torch.mean(original, 0))
        BCE = original * torch.log(reconstructed + 1e-10) + (1-original) * torch.log(1-reconstructed + 1e-10)
        self.loss = -torch.sum(I * BCE) / batch_size
        return self.loss
