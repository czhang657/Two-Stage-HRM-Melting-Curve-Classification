import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt

class EncoderLayer(nn.Module):
    def __init__(self, d_model, attention, d_ff=None, dropout=0.1, activation="relu", num_head=8):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.d_model = d_model
        self.num_head = num_head

    def forward(self, queries, x, attn_mask=None):
        new_x, attn = self.attention(
            queries, x, x,
            attn_mask=attn_mask
        )
        queries = queries + self.dropout(new_x)

        y = self.norm1(queries)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(queries + y), attn


# class CrossAttention(nn.Module):
#     def __init__(self, mask_flag=False, attention_dropout=0.1, output_attention=False, num_head=8, d_model=768):
#         super(CrossAttention, self).__init__()
#         self.mask_flag = mask_flag
#         self.output_attention = output_attention
#         self.dropout = nn.Dropout(attention_dropout)
#         self.num_head = num_head
#         self.d_model = d_model
#         self.Wq = nn.Linear(self.d_model, self.d_model)
#         self.Wk = nn.Linear(self.d_model, self.d_model)
#         self.Wv = nn.Linear(self.d_model, self.d_model)
#         self.dk = d_model // self.num_head

#     def forward(self, queries, keys, values, attn_mask=None):
#         B, Lq, D = queries.shape
#         B, Lk, D = keys.shape
#         queries = self.Wq(queries).reshape(B, Lq, self.num_head, self.dk)
#         keys = self.Wk(keys).reshape(B, Lk, self.num_head, self.dk)
#         values = self.Wv(values).reshape(B, Lk, self.num_head, self.dk)

#         queries = queries.permute(0, 2, 1, 3)
#         keys = keys.permute(0, 2, 3, 1)
#         values = values.permute(0, 2, 1, 3)

#         scores = torch.matmul(queries, keys) / sqrt(self.dk) 
#         if self.mask_flag:
#             if attn_mask is None: 
#                 attn_mask = TriangularCausalMask(B, Lq, device=queries.device)
#             scores.masked_fill_(attn_mask.mask, float('-inf'))
#         attn = torch.softmax(scores, dim=-1)
#         attn = self.dropout(attn)
#         context = torch.matmul(attn, values)

#         context = context.permute(0, 2, 1, 3).reshape(B, Lq, -1)

#         if self.output_attention:
#             return context, attn
#         else:
#             return context, None

class CrossAttention(nn.Module):
    def __init__(self, mask_flag=False, attention_dropout=0.1, output_attention=False,
                 num_head=8, d_model=768, use_gate=True):
        super(CrossAttention, self).__init__()
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

        self.num_head = num_head
        self.d_model = d_model
        self.dk = d_model // num_head

        # q, k, v projections
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)

        # 🔥 Qwen 2025 Gating Unit
        self.use_gate = use_gate
        if use_gate:
            # gate per head
            self.gate = nn.Linear(d_model, num_head)

        # output projection unchanged
        # (你之后可能会加 FFN，这里按你的原架构不动)

    def forward(self, queries, keys, values, attn_mask=None):
        B, Lq, D = queries.shape
        B, Lk, D = keys.shape

        # Project
        q = self.Wq(queries).reshape(B, Lq, self.num_head, self.dk)
        k = self.Wk(keys).reshape(B, Lk, self.num_head, self.dk)
        v = self.Wv(values).reshape(B, Lk, self.num_head, self.dk)

        q = q.permute(0, 2, 1, 3)   # (B, H, Lq, dk)
        k = k.permute(0, 2, 3, 1)   # (B, H, dk, Lk)
        v = v.permute(0, 2, 1, 3)   # (B, H, Lk, dk)

        # Attention scores
        scores = torch.matmul(q, k) / sqrt(self.dk)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, Lq, device=q.device)
            scores.masked_fill_(attn_mask.mask, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention
        context = torch.matmul(attn, v)  # (B, H, Lq, dk)

        # 🔥 Qwen 2025 Gating: gate = sigmoid(W_g * q_original)
        if self.use_gate:
            # q_original: (B, Lq, D)
            gate = torch.sigmoid(self.gate(queries))  # (B, Lq, H)

            # reshape gate to (B, H, Lq, 1)
            gate = gate.permute(0, 2, 1).unsqueeze(-1)

            # gated context
            context = context * gate

        # Combine heads
        context = context.permute(0, 2, 1, 3).reshape(B, Lq, -1)

        if self.output_attention:
            return context, attn
        return context, None