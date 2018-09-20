import torch
import torch.nn as nn

# ---------------------------------------------
# RECOSTRUCTION LOSSES

class ImportanceWeightedBCELoss(nn.Module):
    '''
    This loss is good for binary images.
    Importance weights are calculated as the deviations from 
    the average of the images in the batch axes.
    '''
    def __init__(self):
        super(ImportanceWeightedBCELoss, self).__init__()
        self.loss = 0.0

    def forward(self, reconstructed, original):
        batch_size = original.size(0)
        I = torch.abs(original - torch.mean(original, 0))
        BCE = original * torch.log(reconstructed + 1e-10) + (1-original) * torch.log(1-reconstructed + 1e-10)
        self.loss = -torch.sum(I * BCE) / batch_size
        return self.loss


class ImportanceWeightedBernoulliKLdivLoss(nn.Module):
    '''
    This method can be good for handling images 
    with values between 0 and 1.
    Importance weights are calculated as the deviations from 
    the average of the images in the batch axes.
    '''
    def __init__(self, importance):
        super(ImportanceWeightedBernoulliKLdivLoss, self).__init__()
        self.importance = importance
        self.loss = 0.0
    
    def forward(self, reconstructed, original):
        eps = 1e-15
        batch_size = original.size(0)
        I = self.importance(original)
        KL = (original * torch.log(original/(reconstructed + eps) + eps) 
            + (1-original) * torch.log((1 - original)/(1-reconstructed + eps) + eps))
        self.loss = torch.sum(I * KL) / batch_size
        return self.loss


# ---------------------------------------------
# FUNCTOINS for IMPORTANCE 

def batch_importance(original):
    return torch.abs(original - torch.mean(original, 0)) # average over the batches

def channel_importance(original):
    return torch.abs(original - torch.mean(original, 1)) # average over the channels

def preception_field_importance(features):
    pass


# ---------------------------------------------
# LOSSES for REGULARIZATION

def sparsity_loss(latent, rho, beta):
    eps = 1e-15
    rho_ = latent.mean(0) # average activations for each node
    kl_div = rho * torch.log(rho/(rho_ + eps) + eps) + (1-rho) * torch.log((1-rho)/(1-rho_ + eps) + eps) # KL divergence for avg. activation
    return beta * kl_div.sum()/rho_.size(0)
