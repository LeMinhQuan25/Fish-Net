# SwinBlock đơn giản: WindowAttention + MLP + LayerNorm
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, dim, mlp_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, dim)
        self.act = nn.GELU()
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.ws = window_size
        self.qkv = nn.Linear(dim, dim*3)
        self.proj = nn.Linear(dim, dim)
    def forward(self, x):
        B, H, W, C = x.shape
        ws = self.ws
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        x = F.pad(x, (0,0,0,pad_w,0,pad_h))
        Hp, Wp = x.shape[1], x.shape[2]
        x = x.view(B, Hp//ws, ws, Wp//ws, ws, C).permute(0,1,3,2,4,5).contiguous().view(-1, ws*ws, C)
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(t.shape[0], t.shape[1], self.num_heads, C//self.num_heads).transpose(1,2) for t in qkv]
        attn = (q @ k.transpose(-2, -1)) * (1.0 / (C//self.num_heads)**0.5)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1,2).contiguous().view(x.shape[0], x.shape[1], C)
        x = self.proj(x)
        x = x.view(B, Hp//ws, Wp//ws, ws, ws, C).permute(0,1,3,2,4,5).contiguous().view(B, Hp, Wp, C)
        return x[:, :H, :W, :]

class SwinBlock(nn.Module):
    def __init__(self, dim, num_heads=8, window_size=7, mlp_dim=128):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads, window_size)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_dim)
    def forward(self, x):
        # x: B,C,H,W -> B,H,W,C
        x = x.permute(0,2,3,1)
        h = self.norm1(x)
        h = self.attn(h)
        x = x + h
        h = self.norm2(x)
        h = self.mlp(h)
        x = x + h
        return x.permute(0,3,1,2)
