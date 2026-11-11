import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from copy import deepcopy as cp

def get_activation_function(af):
    af = af.lower()
    if af == 'relu':
        return nn.ReLU()
    elif af == 'gelu':
        return nn.GELU()
    elif af == 'silu':
        return nn.SiLU()

def get_normalization(norm):
    norm = norm.lower()
    if norm == 'rms' or norm == 'rmsnorm':
        return nn.RMSNorm
    else:
        return nn.LayerNorm

class Encoder(nn.Module):
    def __init__(self, d_model, num_layer=1, num_head=8, dropout=0.1, d_ff=None, attention=None, activation="relu", dim=-2, norm_first=True):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        attn = attention or Attention(d_model, num_head, dropout, norm_first=norm_first)
        self.attn = nn.ModuleList([cp(attn) for i in range(num_layer)])
        ffn = FFN(d_model, d_ff, dropout, activation, norm_first)
        self.ffn = nn.ModuleList([cp(ffn) for i in range(num_layer)])
        self.dim = dim
        self.num_layer = num_layer

    def forward(self, x):
        x = x.transpose(-2, self.dim).contiguous()
        shape = x.shape
        x = x.flatten(0, -3)
        # x = x.view(-1, shape[-2], shape[-1])
        for i in range(self.num_layer):
            x = self.attn[i](x)
            x = self.ffn[i](x)
        x = x.unflatten(0, shape[:-2]).transpose(-2, self.dim)
        # x = x.view(shape).transpose(-2, self.dim).contiguous()
        return x

class Encoder_new(nn.Module):
    def __init__(self, d_model, num_layer=1, num_head=8, dropout=0.1, d_ff=None, attention=None, activation="silu", dim=-2, norm_first=True):
        super().__init__()
        d_ff = d_ff or int(8 / 3 * d_model)
        attn = attention or Attention(d_model, num_head, dropout, normalization='rms', norm_first=norm_first)
        self.attn = nn.ModuleList([cp(attn) for i in range(num_layer)])
        ffn = SwiGLU(d_model, d_ff, dropout, activation, 'rms', norm_first)
        self.ffn = nn.ModuleList([cp(ffn) for i in range(num_layer)])
        self.dim = dim
        self.num_layer = num_layer

    def forward(self, x):
        x = x.transpose(-2, self.dim)
        shape = x.shape
        x = x.flatten(0, -3)
        for i in range(self.num_layer):
            x = self.attn[i](x)
            x = self.ffn[i](x)
        x = x.unflatten(0, shape[:-2]).transpose(-2, self.dim)
        return x

class Encoder_rope(nn.Module):
    def __init__(self, d_model, num_layer=1, num_head=8, dropout=0.1, d_ff=None, attention=None, activation="silu", dim=-2, norm_first=True):
        super().__init__()
        d_ff = d_ff or int(8 / 3 * d_model)
        attn = Attention_RoPE(d_model, num_head, dropout)
        self.attn = nn.ModuleList([cp(attn) for i in range(num_layer)])
        ffn = SwiGLU(d_model, d_ff, dropout, activation, 'rms', norm_first)
        self.ffn = nn.ModuleList([cp(ffn) for i in range(num_layer)])
        self.dim = dim
        self.num_layer = num_layer

    def forward(self, x):
        x = x.transpose(-2, self.dim)
        shape = x.shape
        x = x.flatten(0, -3)
        for i in range(self.num_layer):
            x = self.attn[i](x)
            x = self.ffn[i](x)
        x = x.unflatten(0, shape[:-2]).transpose(-2, self.dim)
        return x

class Decoder(nn.Module):
    def __init__(self, d_model, num_layer=1, num_head=8, dropout=0.1, d_ff=None, attention=None, activation="relu", dim=-2, norm_first=True):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        attn = attention or Attention(d_model, num_head, dropout, norm_first=norm_first)
        self.attn = nn.ModuleList([cp(attn) for i in range(num_layer)])
        ffn = FFN(d_model, d_ff, dropout, activation, norm_first)
        self.ffn = nn.ModuleList([cp(ffn) for i in range(num_layer)])
        self.dim = dim
        self.num_layer = num_layer

    def forward(self, trg, src):
        """
        :param src: b,t,n,d
        :param trg: b,h,n,d
        :return: b,h,n,d
        """
        if src.ndim < trg.ndim:
            src = src.flatten(0, -2).unsqueeze(-2)
        else:
            src = src.transpose(-2, self.dim)
            src = src.flatten(0, -3)
        trg = trg.transpose(-2, self.dim)
        shape = trg.shape
        trg = trg.flatten(0, -3)
        for i in range(self.num_layer):
            trg = self.attn[i](trg, src)
            trg = self.ffn[i](trg)
        out = trg.unflatten(0, shape[:-2]).transpose(-2, self.dim)
        return out
    
class FFN(nn.Module):
    def __init__(self, emb_dim, hid_dim=None, dropout=0.1, activation="gelu", norm_first=False):
        super().__init__()
        hid_dim = hid_dim or emb_dim * 4
        self.lin1 = nn.Linear(emb_dim, hid_dim)
        self.lin2 = nn.Linear(hid_dim, emb_dim)
        self.norm = nn.LayerNorm(emb_dim)
        self.dp = nn.Dropout(dropout)
        self.af = get_activation_function(activation)
        self.nf = norm_first

    def forward(self, x):
        if self.nf:
            x_ = self.dp(self.af(self.lin1(self.norm(x))))
            x_ = self.dp(self.lin2(x_))
            return x + x_
        else:
            x_ = self.dp(self.af(self.lin1(x)))
            x_ = self.dp(self.lin2(x_))
            return self.norm(x + x_)

class SwiGLU(nn.Module):
    def __init__(self, emb_dim, hid_dim=None, dropout=0.0, activation="silu", normalization='rms', norm_first=True, bias=False):
        super().__init__()
        hid_dim = hid_dim or emb_dim * 4
        self.lin1 = nn.Linear(emb_dim, hid_dim, bias=bias)
        self.lin2 = nn.Linear(emb_dim, hid_dim, bias=bias)
        self.lin3 = nn.Linear(hid_dim, emb_dim, bias=bias)
        self.norm = get_normalization(normalization)(emb_dim)
        self.dp = nn.Dropout(dropout)
        self.af = get_activation_function(activation)
        self.nf = norm_first

    def forward(self, x):
        if self.nf:
            x_ = self.norm(x)
            x_ = self.dp(self.af(self.lin1(x_)) * self.lin2(x_))
            x_ = self.dp(self.lin3(x_))
            return x + x_
        else:
            x_ = self.dp(self.af(self.lin1(x)) * self.lin2(x))
            x_ = self.dp(self.lin3(x_))
            return self.norm(x + x_)

class RoPE(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 4096, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, freqs)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(self, xq, xk):
        seq_len = xq.shape[1]
        freqs_cis = self.freqs_cis[:seq_len]
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.view(1, seq_len, 1, -1)
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
        return xq_out.type_as(xq), xk_out.type_as(xk)

class Attention(nn.Module):
    def __init__(self, emb_dim, num_heads=8, dropout=0.1, kv_dim=None, normalization='layer', norm_first=True):
        super().__init__()
        self.attn = nn.MultiheadAttention(emb_dim, num_heads, dropout, kdim=kv_dim, vdim=kv_dim, batch_first=True)
        self.norm = get_normalization(normalization)(emb_dim)
        self.nf = norm_first

    def forward(self, x, kv=None, attn_mask=None, kp_mask=None):
        kv = x if kv is None else kv
        use_flash_attn = (x.size(0) <= 65536) and (x.size(1) >= 128)
        # use_flash_attn = True
        with torch.backends.cuda.sdp_kernel(enable_flash=use_flash_attn):
            if self.nf:
                x_, _ = self.attn(self.norm(x), self.norm(kv), self.norm(kv), attn_mask=attn_mask, key_padding_mask=kp_mask, need_weights=False)
                return x + x_
            else:
                x_, _ = self.attn(x, kv, kv, attn_mask=attn_mask, key_padding_mask=kp_mask, need_weights=False)
                return self.norm(x + x_)

class Attention_RoPE(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout=0.1, max_seq_len=1024):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        self.dp = dropout
        self.norm = nn.RMSNorm(emb_dim)

        self.proj = nn.Linear(emb_dim, 3 * emb_dim, bias=False)
        self.out_proj = nn.Linear(emb_dim, emb_dim, bias=False)
        self.rope = RoPE(dim=self.head_dim, max_seq_len=max_seq_len)

    def forward(self, x):
        B, L, D = x.shape
        q, k, v = self.proj(self.norm(x)).view(B, L, self.num_heads * 3, self.head_dim).chunk(3, -2)
        q, k = self.rope(q, k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn_output = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dp)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(attn_output) + x

class Bottleneck_Attention(nn.Module):
    def __init__(self, emb_dim, ib_num, num_heads=8, dropout=0.1, dim=-2):
        super().__init__()

        self.attn1 = Attention(emb_dim, num_heads, dropout)
        self.ffn1 = FFN(emb_dim)
        self.attn2 = Attention(emb_dim, num_heads, dropout)
        self.ffn2 = FFN(emb_dim)
        self.ib = nn.Parameter(nn.init.xavier_uniform(torch.empty(ib_num, emb_dim)).unsqueeze(0))
        self.dim = dim

    def forward(self, x):
        x = x.transpose(-2, self.dim)
        shape = x.shape
        x = x.flatten(0, -3)
        ib = self.ffn1(self.attn1(self.ib.expand(x.size(0), -1, -1), x))
        out = self.ffn2(self.attn2(x, ib))
        out = out.unflatten(0, shape[:-2]).transpose(-2, self.dim)
        return out