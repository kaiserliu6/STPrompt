import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def masked_mae_torch(preds, labels, threshold):
    if threshold is None:
        mask = 1
    else:
        mask = labels > threshold
        mask = mask.float()
        mask /= mask.mean()
    loss = torch.abs(preds - labels)
    loss = loss * mask
    return loss.mean()

def masked_mape_torch(preds, labels, threshold):
    if threshold is None:
        mask = labels > 0
    else:
        mask = labels > threshold
    mask = mask.float()
    mask /= mask.mean()
    loss = torch.abs(preds - labels) / labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return loss.mean()

def masked_smape_torch(preds, labels, threshold):
    if threshold is None:
        mask = labels > 0
    else:
        mask = labels > threshold
    mask = mask.float()
    mask /= mask.mean()
    loss = torch.abs(preds - labels) / (preds + labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return loss.mean()

def masked_mse_torch(preds, labels, threshold):
    if threshold is None:
        mask = 1
    else:
        mask = labels > threshold
        mask = mask.float()
        mask /= mask.mean()
    loss = torch.pow(preds - labels, 2)
    loss = loss * mask
    return loss.mean()

def masked_rmse_torch(preds, labels, threshold):
    return masked_mse_torch(preds, labels, threshold).sqrt()

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07, dim=0, reduction='mean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.dim = dim

        if self.reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f"Unknown reduction type: {self.reduction}. Choose from 'none', 'mean', or 'sum'.")

    def forward(self, query, key=None) -> torch.Tensor:
        query = self.normalize(query)
        if key is None:
            key = query
        else:
            key = self.normalize(key)

        cos_sim = (query @ key.transpose(-1, -2)) / self.temperature
        pos_sim = torch.diagonal(cos_sim, 0, -2, -1)
        neg_sim = torch.logsumexp(cos_sim, -1)
        losses = -pos_sim + neg_sim

        if self.reduction == 'mean':
            return torch.mean(losses)
        elif self.reduction == 'sum':
            return torch.sum(losses)
        elif self.reduction == 'none':
            return losses
        else:
            # This case is already handled in __init__, but good practice to include.
            raise ValueError(f"Unknown reduction type: {self.reduction}")

    def normalize(self, embedding):
        if self.dim not in [-2, embedding.ndim - 2]:
            embedding = embedding.transpose(-2, self.dim)
        embedding = F.normalize(embedding.flatten(0, -3), dim=-1)
        return embedding