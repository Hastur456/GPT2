import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()
    assert config.n_embd % config.n_head == 0
    self.n_embd = config.n_embd
    self.n_head = config.n_head
    self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd)
    self.c_proj = nn.Linear(self.n_embd, self.n_embd)
    self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)
                                            .view(1, 1, config.block_size, config.block_size)))


  def forward(self, x):
    B, T, C = x.size()
    qkv = self.c_attn(x)
    q, k, v = qkv.split(self.n_embd, dim=2)

    q = q.view(B, T, self.n_head, self.n_embd // self.n_head).transpose(1, 2) # (B, T, n_hd, n_e) --> (B, n_hd, T, n_e)
    k = k.view(B, T, self.n_head, self.n_embd // self.n_head).transpose(1, 2) # (B, T, n_hd, n_e) --> (B, n_hd, T, n_e)
    v = v.view(B, T, self.n_head, self.n_embd // self.n_head).transpose(1, 2) # (B, T, n_hd, n_e) --> (B, n_hd, T, n_e)
    attention = (q @ k.transpose(-2, -1))  * (1.0 / (k.size(-1)**0.5)) # (B, n_hd, T, T)
    attention = attention.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf")) # (B, n_hd, T, T)
    out = F.softmax(attention, dim=-1) @ v # (B, n_hd, T, T) @ (B, n_hd, T, n_e) --> (B, n_hd, T, n_e)
    out = out.transpose(1, 2).contiguous().view(B, T, -1) # (B, T, C)
    out = self.c_proj(out)
    return out


class MLP(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
    self.gelu = nn.GELU(approximate="tanh")
    self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

  def forward(self, x):
    x = self.c_fc(x)
    x = self.gelu(x)
    x = self.c_proj(x)
    return x


class Block(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.attn = SelfAttention(config)
    self.mlp = MLP(config)
    self.ln_1 = nn.LayerNorm(config.n_embd)
    self.ln_2 = nn.LayerNorm(config.n_embd)

  def forward(self, x):
    x = x + self.attn(self.ln_1(x))
    x = x + self.mlp(self.ln_2(x))
    return x