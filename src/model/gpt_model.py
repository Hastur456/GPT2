import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from .modules import Block
from src.model.config import GPTConfig
import os


def load_model(path_to_model: str, model_type: str):
    if path_to_model and os.path.isfile(path_to_model):
        checkpoint = torch.load(path_to_model, map_location='cpu')
        model = GPT(config=checkpoint['config'])
        model.load_state_dict(checkpoint['model'])
        print(f"Model loaded from checkpoint: {path_to_model}")
    else:
        model = GPT.from_pretrained(model_type)
        print(f"Model loaded from pretrained HF model: {model_type}")
    return model


class GPT(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.ModelOutput = namedtuple("ModelOutput", ["logits", "loss"])
    self.config = config
    self.transformer = nn.ModuleDict(dict(
         wte = nn.Embedding(config.vocab_size, config.n_embd),
         wpe = nn.Embedding(config.block_size, config.n_embd),
         h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
         ln_f = nn.LayerNorm(config.n_embd)
    ))
    self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    self.apply(self._init_weights)


  def _init_weights(self, module):
      if isinstance(module, nn.Linear):
          torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
          if module.bias is not None:
              torch.nn.init.zeros_(module.bias)
      elif isinstance(module, nn.Embedding):
          torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


  def forward(self, idx, target=None):
    B, T = idx.size()
    assert T <= self.config.block_size, print("The length of the sequence greater then size of the block")
    pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
    x = self.transformer.wte(idx) + self.transformer.wpe(pos)

    for block in self.transformer.h:
      x = block(x)

    x = self.transformer.ln_f(x)
    logits = self.lm_head(x) # (B, T, C) where C - vocab size

    loss = None
    if target is not None:
      target = target.to(idx.device)
      loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))

    return self.ModelOutput(
        logits=logits,
        loss=loss
    )


  @classmethod
  def from_pretrained(self, model_type):
    assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
    from transformers import GPT2LMHeadModel
    print("loading weights from pretrained gpt: {}".format(model_type))

    config_args = {
        'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
        'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
        'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
        'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
    }[model_type]

    config_args["vocab_size"] = 50257
    config_args["block_size"] = 1024

    config = GPTConfig(**config_args)

    model = GPT(config)
    sd = model.state_dict()

    sd_keys = sd.keys()
    sd_keys = [k for k in sd_keys if not k.endswith(".attn.bias")]

    hf_model = GPT2LMHeadModel.from_pretrained(model_type)
    hf_sd = hf_model.state_dict()

    hf_sd_keys = hf_sd.keys()
    hf_sd_keys = [k for k in hf_sd_keys if not k.endswith(".attn.masked_bias")]
    hf_sd_keys = [k for k in hf_sd_keys if not k.endswith(".attn.bias")]

    transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
    assert len(hf_sd_keys) == len(sd_keys), print(f"mistake of keys lengths: {len(hf_sd_keys.keys())} != {len(sd_keys.keys())}")

    for k in hf_sd_keys:
      if any(k.endswith(s) for s in transposed):
        assert sd[k].shape[:] == hf_sd[k].shape[::-1], print("Mistake of shapes nontransposed and transposed")
        print(f"Key: {k} successfully transposed and upload")
        with torch.no_grad():
          sd[k].copy_(hf_sd[k].t())
      else:
        assert sd[k].shape[:] == hf_sd[k].shape[:], print("Mistake of shapes between dimentions")
        print(f"Key: {k} successfully upload")
        with torch.no_grad():
          sd[k].copy_(hf_sd[k])
    return model