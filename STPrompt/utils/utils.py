import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Unfold(nn.Module):
    def __init__(self, lenth, stride, dialation=1, padding=0):
        super().__init__()

        self.lenth = lenth
        self.stride = stride
        self.dialation = dialation
        self.padding = padding

    def forward(self, x):
        if self.padding > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.padding), 'replicate')
        if self.dialation > 1:
            x = x.unfold(1, (self.lenth - 1) * self.dialation, self.stride)
            x = x[:, : self.lenth * self.dialation: self.dialation]
        else:
            x = x.unfold(1, self.lenth, self.stride)
        return x

def ensure_dir(dir_path):
    """Make sure the directory exists, if it does not exist, create it.

    Args:
        dir_path (str): directory path
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path

def get_laplacian(adj: torch.Tensor):
    d = adj.sum(-1)
    isolated_point_num = (d == 0).sum().item()
    d_inv_sqrt = d.pow(-0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    identity = torch.eye(adj.shape[-1], device=adj.device, dtype=d_mat_inv_sqrt.dtype)
    normalized_laplacian = identity - (d_mat_inv_sqrt @ adj.to(d_mat_inv_sqrt.dtype) @ d_mat_inv_sqrt)
    return normalized_laplacian, isolated_point_num

def get_laplacian_emb(adj: torch.Tensor, lape_dim=32):
    # adj: N, N
    L, isolated_point_num = get_laplacian(adj)
    eig_vals, eig_vecs = torch.linalg.eigh(L)

    laplacian_pe = eig_vecs[:, isolated_point_num + 1: lape_dim + isolated_point_num + 1]
    return laplacian_pe

def graph_row_wise_torch(adj):
    if isinstance(adj, np.ndarray):
        adj = torch.tensor(adj)
    adj = adj.to(torch.float32)
    d_inv = adj.sum(-1, keepdim=True)
    d_inv = d_inv.pow(-1)
    d_inv.nan_to_num_(0)
    return d_inv * adj