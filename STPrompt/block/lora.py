import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, overload
from torch import Tensor


class LoRA_Lin(nn.Module):
    @overload
    def __init__(self, original_linear: nn.Linear, rank: int = 8, alpha: int = 16):
        ...

    @overload
    def __init__(self, weight: Tensor, bias: Optional[Tensor] = None, rank: int = 8, alpha: int = 16):
        ...

    @overload
    def __init__(self, weight: Tensor, rank: int = 8, alpha: int = 16):
        ...

    def __init__(self, *args, rank: int = 8, alpha: int = 16):
        super().__init__()
        valid_para = 1
        if args and isinstance(args[0], nn.Linear):
            original_linear = args[0]
            in_features = original_linear.in_features
            out_features = original_linear.out_features
            weight = original_linear.weight.data
            bias = original_linear.bias.data if original_linear.bias is not None else None
        else:
            weight = args[0]
            out_features, in_features = weight.shape
            if len(args) > 1 and isinstance(args[1], Tensor):
                bias = args[1]
                valid_para = 2
            else:
                bias = None

        if len(args) > valid_para:
            rank = args[valid_para]
            if len(args) > valid_para + 1:
                alpha = args[valid_para + 1]

        self.rank = rank
        self.alpha = alpha
        self.in_features = in_features
        self.out_features = out_features
        has_bias = bias is not None
        self.device = weight.device

        self.register_buffer('original_weight', weight.clone())
        if has_bias:
            self.register_buffer('original_bias', bias.clone())
        else:
            self.register_buffer('original_bias', None)

        self.lora_A = nn.Parameter(torch.empty(rank, in_features, device=self.device))
        self.lora_B = nn.Parameter(torch.empty(out_features, rank, device=self.device))
        self.reset_lora_parameters()

    def reset_lora_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    @torch.no_grad()
    def merge_lora_parameters(self):
        self.original_weight = self.original_weight + self.lora_B @ self.lora_A * (self.alpha / self.rank)
        self.original_weight.requires_grad = False
        # self.original_weight = self.original_weight + (self.lora_B @ self.lora_A)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_output = F.linear(x, self.original_weight, self.original_bias)

        lora_intermediate = F.linear(x, self.lora_A)
        lora_output = F.linear(lora_intermediate, self.lora_B)

        return original_output + lora_output * (self.alpha / self.rank)

    def __repr__(self):
        return (f'{self.__class__.__name__}(in_features={self.in_features}, '
                f'out_features={self.out_features}, '
                f'rank={self.rank}, alpha={self.alpha}, '
                f'has_bias={self.original_bias is not None})')

class LoRA_attn(nn.Module):
    def __init__(self, attn: nn.MultiheadAttention, rank: int = 8, alpha: int = 16):
        super().__init__()
        if not isinstance(attn, nn.MultiheadAttention):
            raise TypeError(f"Expect nn.MultiheadAttention, but get {type(attn).__name__} instead")

        self.rank = rank
        self.alpha = alpha

        self.embed_dim = attn.embed_dim
        self.num_heads = attn.num_heads
        self.dropout = attn.dropout
        self.head_dim = attn.head_dim
        self.batch_first = attn.batch_first

        if attn._qkv_same_embed_dim:
            q_weight, k_weight, v_weight = attn.in_proj_weight.chunk(3, dim=0)
        else:
            q_weight, k_weight, v_weight = attn.q_proj_weight, attn.k_proj_weight, attn.v_proj_weight

        if attn.in_proj_bias is not None:
            q_bias, k_bias, v_bias = attn.in_proj_bias.chunk(3)
            self.q_proj = LoRA_Lin(q_weight, q_bias, rank, alpha)
            self.k_proj = LoRA_Lin(k_weight, k_bias, rank, alpha)
            self.v_proj = LoRA_Lin(v_weight, v_bias, rank, alpha)
        else:
            self.q_proj = LoRA_Lin(q_weight, rank, alpha)
            self.k_proj = LoRA_Lin(k_weight, rank, alpha)
            self.v_proj = LoRA_Lin(v_weight, rank, alpha)

        self.out_proj = LoRA_Lin(attn.out_proj, rank, alpha)

    def forward(self,
                query: Tensor,
                key: Tensor,
                value: Tensor,
                key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = False,
                attn_mask: Optional[Tensor] = None,
                average_attn_weights: bool = True) -> Tuple[Tensor, Optional[Tensor]]:

        b, l, d = query.shape

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        # (B, L, E) -> (B, L, H, D) -> (B, H, L, D)
        q = q.view(b, l, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(b, l, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(b, l, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if need_weights:
            q_scaled = q * math.sqrt(1.0 / float(d))

            if attn_mask is not None:
                attn_output_weights = torch.baddbmm(
                    attn_mask, q_scaled, k.transpose(-2, -1)
                )
            else:
                attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
            attn_output_weights = F.softmax(attn_output_weights, dim=-1)
            if self.dropout > 0.0:
                attn_output_weights = F.dropout(attn_output_weights, p=self.dropout)

            attn_output = torch.bmm(attn_output_weights, v)

            attn_output = (
                attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
            )
            attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
            attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

            # optionally average attention weights over heads
            attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
            if average_attn_weights:
                attn_output_weights = attn_output_weights.mean(dim=1)

            if not is_batched:
                # squeeze the output if input was unbatched
                attn_output = attn_output.squeeze(1)
                attn_output_weights = attn_output_weights.squeeze(0)
            return attn_output, attn_output_weights
        else:
            if attn_mask is not None:
                if attn_mask.size(0) == 1 and attn_mask.dim() == 3:
                    attn_mask = attn_mask.unsqueeze(0)
                else:
                    attn_mask = attn_mask.view(b, self.num_heads, -1, l)

            dropout = self.dropout if self.training else 0.0
            attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask, dropout)
            attn_output = attn_output.permute(0, 2, 1, 3).reshape(b, l, d)
            attn_output = self.out_proj(attn_output)
            return attn_output, None

    def reset_lora_parameters(self):
        self.q_proj.reset_lora_parameters()
        self.k_proj.reset_lora_parameters()
        self.v_proj.reset_lora_parameters()
        self.out_proj.reset_lora_parameters()

    def merge_lora_parameters(self):
        self.q_proj.merge_lora_parameters()
        self.k_proj.merge_lora_parameters()
        self.v_proj.merge_lora_parameters()
        self.out_proj.merge_lora_parameters()

if __name__ == '__main__':
    attn = nn.MultiheadAttention(16,2, 0.1, batch_first=True)
    attn.eval()
    lora_attn = LoRA_attn(attn)
    lora_attn.eval()
    x = torch.randn(8, 10, 16)
    o1, _ = attn(x, x, x, need_weights=False)
    o2, _ = lora_attn(x, x, x)
    print(torch.allclose(o1, o2))
    print((o1-o2).max(), (o1-o2).min(), (o1-o2).mean())
    # l1 = nn.Linear(16, 16)
    # l2 = LoRA_Lin(l1)
    # x = torch.randn(2, 16)
    # y1 = l1(x)
    # y2 = l2(x)
    # print(torch.allclose(y1, y2))