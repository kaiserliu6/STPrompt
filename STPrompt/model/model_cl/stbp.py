import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import Model_cl


class Model(Model_cl):
    def __init__(self, config, data):
        super().__init__(config)
        config.input_dim = self.dim_in
        config.output_dim = self.dim_out
        self.data = data
        self.graph = self.get_graph(config.start_year)
        self.model = STBP_Model(config)

    def prepare_for_new_task(self, **kwargs):
        for name, param in self.model.named_parameters():
            param.requires_grad = "pattern_bank" in name
        self.graph = self.get_graph(self.year)
        self.model.expand_adaptive_params(self.config.num_nodes_year[self.year])

    def forward(self, *data):
        x = self.norm.norm(data[0])
        graph = data[-1] if len(data) > 1 and data[-1] is not None else self.graph
        output = self.model(x, graph)
        output = self.norm.inorm(output)
        return output


class LayerNorm(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, input):
        mean = input.mean(dim=(1, 2), keepdim=True)
        variance = input.var(dim=(1, 2), unbiased=False, keepdim=True)
        input = (input - mean) / torch.sqrt(variance + self.eps)
        return input


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.2):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=(1, 1))
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=(1, 1))
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Conv(nn.Module):
    def __init__(self, features, dropout=0.2):
        super().__init__()
        self.conv = nn.Conv2d(features, features, (1, 1))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class DLGA(nn.Module):
    def __init__(self, d_model, head, seq_length=1):
        super().__init__()
        assert d_model % head == 0
        self.d_k = d_model // head
        self.head = head
        self.q = Conv(d_model)
        self.k = Conv(d_model)
        self.v = Conv(d_model)
        self.concat = Conv(d_model)

    def forward(self, input, node_emb):
        _, d_model, num_nodes, seq_length = input.shape
        query, key, value = self.q(input), self.k(input), self.v(input)
        query = query.view(query.shape[0], -1, self.d_k, query.shape[2], seq_length).permute(0, 1, 4, 3, 2)
        key = key.view(key.shape[0], -1, self.d_k, key.shape[2], seq_length).permute(0, 1, 4, 3, 2)
        value = value.view(value.shape[0], -1, self.d_k, value.shape[2], seq_length).permute(0, 1, 4, 3, 2)
        node_emb = node_emb.view(node_emb.shape[0], -1, self.d_k, node_emb.shape[2], seq_length).permute(0, 1, 4, 3, 2)

        key = torch.softmax(key / math.sqrt(self.d_k), dim=-1)
        query = torch.softmax(query / math.sqrt(self.d_k), dim=-1)
        node_emb = torch.softmax(node_emb / math.sqrt(self.d_k), dim=-1)

        kv = torch.einsum("bhlnx,bhlny->bhlxy", key, value)
        attn_qkv1 = torch.einsum("bhlnx,bhlxy->bhlny", query, kv)

        node_emb_v = torch.einsum("bhlnx,bhlny->bhlxy", node_emb, value)
        attn_qkv2 = torch.einsum("bhlnx,bhlxy->bhlny", query, node_emb_v)

        attn_qkv = attn_qkv1 + attn_qkv2
        x = attn_qkv.permute(0, 1, 4, 3, 2).contiguous().view(attn_qkv.shape[0], d_model, num_nodes, seq_length)
        x = self.concat(x)
        return x


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class Encoder(nn.Module):
    def __init__(self, d_model, head, seq_length=1):
        super().__init__()
        assert d_model % head == 0
        self.d_k = d_model // head
        self.head = head
        self.seq_length = seq_length
        self.d_model = d_model

        self.attn = DLGA(d_model, head, seq_length)
        self.FFW = Mlp(d_model)
        self.norm1 = LayerNorm()
        self.norm2 = LayerNorm()

    def forward(self, x, node_emb):
        gate, scale, memory = torch.chunk(node_emb, 3, dim=1)
        x = x + gate * self.attn(modulate(self.norm1(x), 0, scale), memory)
        x = x + gate * self.FFW(modulate(self.norm2(x), 0, scale))
        return x


class FConv(nn.Module):
    def __init__(self, in_features, out_features, bias=True, fc=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_neigh = nn.Linear(in_features, out_features, bias=bias)
        self.fre_network = FreNetwork(out_features)
        if not fc:
            self.weight_self = nn.Linear(in_features, out_features, bias=False)
        else:
            self.register_parameter("weight_self", None)
        self.reset_parameters()

    def reset_parameters(self):
        self.weight_neigh.reset_parameters()
        if self.weight_self is not None:
            self.weight_self.reset_parameters()

    def forward(self, x):
        output = self.fre_network(self.weight_neigh(x))
        if self.weight_self is not None:
            output += self.weight_self(x)
        return output


class FreNetwork(nn.Module):
    def __init__(self, time_dim=64):
        super().__init__()
        self.time_dim = time_dim
        self.complex_weight = nn.Parameter(torch.randn(1, 1, time_dim // 2 + 1, 2, dtype=torch.float32) * 0.02)
        nn.init.xavier_uniform_(self.complex_weight)

    def forward(self, x):
        x = torch.fft.rfft(x, dim=2, norm="ortho")
        t_bank = torch.view_as_complex(self.complex_weight)
        x = x * t_bank
        x = torch.fft.irfft(x, n=self.time_dim, dim=2, norm="ortho")
        return x


class STBP_Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dropout = config.dropout
        self.input_window = config.input_window
        self.output_window = config.output_window
        self.dim_in = config.dim_in
        self.dim_out = config.dim_out

        in_channel = self.input_window * self.dim_in
        out_channel = self.input_window * self.dim_in
        hidden_channel = config.get("hidden_channel", 192)
        head = config.get("head", 1)

        self.in_channel = in_channel
        self.hidden_channel = hidden_channel
        self.out_channel = out_channel

        self.fconv1 = FConv(in_channel, hidden_channel, bias=True, fc=False)
        self.stmodule = Encoder(d_model=hidden_channel, head=head, seq_length=1)
        self.fconv2 = FConv(hidden_channel, out_channel, bias=True, fc=False)
        self.residual_proj = nn.Identity() if in_channel == out_channel else nn.Linear(in_channel, out_channel, bias=False)
        self.fc = nn.Linear(out_channel, self.output_window * self.dim_out)
        self.activation = nn.GELU()

        self.pattern_bank = nn.Parameter(torch.empty(config.num_nodes, hidden_channel * 3).uniform_(-0.1, 0.1))
        self.num_nodes = config.num_nodes

    def expand_adaptive_params(self, new_num_nodes):
        if new_num_nodes > self.num_nodes:
            new_params = nn.Parameter(
                torch.empty(
                    new_num_nodes - self.num_nodes,
                    self.hidden_channel * 3,
                    dtype=self.pattern_bank.dtype,
                    device=self.pattern_bank.device,
                ).uniform_(-0.1, 0.1)
            )
            self.pattern_bank = nn.Parameter(torch.cat([self.pattern_bank, new_params], dim=0))
            self.num_nodes = new_num_nodes

    def forward(self, x, adj=None):
        batch_size, _, num_nodes, _ = x.shape
        x = x.transpose(1, 2).flatten(2)
        if x.size(-1) != self.in_channel:
            raise ValueError(f"STBP expected flattened input dim {self.in_channel}, got {x.size(-1)}.")
        if num_nodes > self.pattern_bank.size(0):
            raise ValueError(f"STBP pattern_bank has {self.pattern_bank.size(0)} nodes, got input with {num_nodes}.")

        residual = self.residual_proj(x)
        x = F.relu(self.fconv1(x))

        adaptive_params = self.pattern_bank[:num_nodes]
        node_emb = adaptive_params.unsqueeze(0).expand(batch_size, -1, -1).permute(0, 2, 1).unsqueeze(-1)
        x = x.permute(0, 2, 1).unsqueeze(-1)

        x = self.stmodule(x, node_emb).squeeze(-1)
        x = x.permute(0, 2, 1)
        x = self.fconv2(x)
        x = x + residual
        x = self.fc(self.activation(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.unflatten(-1, (self.output_window, self.dim_out)).transpose(1, 2)
        return x
