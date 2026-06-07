import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TemporalEmbedding(nn.Module):
    def __init__(self, emb_dim, num_times):
        super().__init__()
        self.emb1 = nn.Embedding(num_times, emb_dim)
        self.emb2 = nn.Embedding(7, emb_dim)

    def forward(self, x):
        emb1 = self.emb1(x[...,0])
        emb2 = self.emb2(x[...,-1])
        return emb1 + emb2

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # B, L, d
        if isinstance(x, int):
            l = x
        else:
            l = x.size(1)
        return self.pe[:, :l]

class Patch_Embedding(nn.Module):
    def __init__(self, patch_len, stride, emb_dim, padding=True):
        super().__init__()

        self.patch_len = patch_len
        self.stride = stride
        self.emb_dim = emb_dim
        self.padding = padding
        self.lin = nn.Linear(patch_len, emb_dim)

    def forward(self, x):
        if self.padding:
            x = F.pad(x, (0, self.stride), 'replicate')
        x = x.unfold(-1, self.patch_len, self.stride)
        x = self.lin(x)
        return x

class TokenEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim, norm_layer=None):
        super().__init__()
        self.token_embed = nn.Linear(input_dim, embed_dim, bias=True)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

    def forward(self, x):
        x = self.token_embed(x)
        x = self.norm(x)
        return x

class LaplacianPE(nn.Module):
    def __init__(self, lape_dim, embed_dim):
        super().__init__()
        self.embedding_lap_pos_enc = nn.Linear(lape_dim, embed_dim)

    def forward(self, lap_mx):
        lap_pos_enc = self.embedding_lap_pos_enc(lap_mx).unsqueeze(0).unsqueeze(0)
        return lap_pos_enc

class DataEmbedding(nn.Module):
    def __init__(self, feature_dim, embed_dim, lape_dim, num_times,
                 drop=0., add_time_in_day=False, add_day_in_week=False, device=torch.device('cuda')):
        super().__init__()

        self.add_time_in_day = add_time_in_day
        self.add_day_in_week = add_day_in_week

        self.device = device
        self.embed_dim = embed_dim
        self.feature_dim = feature_dim
        self.value_embedding = TokenEmbedding(feature_dim, embed_dim)

        self.position_encoding = PositionalEmbedding(embed_dim, 100)
        if self.add_time_in_day:
            self.minute_size = num_times
            self.daytime_embedding = nn.Embedding(self.minute_size, embed_dim)
        if self.add_day_in_week:
            weekday_size = 7
            self.weekday_embedding = nn.Embedding(weekday_size, embed_dim)
        self.spatial_embedding = LaplacianPE(lape_dim, embed_dim)
        self.dropout = nn.Dropout(drop)

    def forward(self, x, p, g):
        """
        x: B, T, N, d_fea
        p: B, T, 2
        g: N, d_lap
        Returns: B, T, N, d_emb
        """
        x = self.value_embedding(x)
        x += self.position_encoding(x).unsqueeze(2)
        if self.add_time_in_day:
            x += self.daytime_embedding(p[..., 0]).unsqueeze(2)
        if self.add_day_in_week:
            x += self.weekday_embedding(p[..., -1]).unsqueeze(2)
        x += self.spatial_embedding(g)
        x = self.dropout(x)
        return x