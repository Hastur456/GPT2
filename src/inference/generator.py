import torch
from transformers import GPT2Tokenizer
from types import SimpleNamespace


class GPT2Inference:
    def __init__(self, model, tokenizer, device):
        self.device = device
        self.model = model.to(device)
        self.tokenizer = tokenizer

    @torch.no_grad()
    def generate_sequence(self, prompt, max_new_tokens=50, top_k=50, temperature=1.0):
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        idx = torch.LongTensor(input_ids).unsqueeze(0).to(self.device)

        self.model.eval()
        for _ in range(max_new_tokens):
            outputs = self.model(idx)
            logits = outputs.logits
            logits = logits[:, -1, :]
            if temperature > 0:
                logits = logits / temperature

            probs = torch.softmax(logits, dim=-1)

            if top_k is not None and top_k > 0:
                topk_vals, topk_idx = torch.topk(probs, top_k, dim=-1)
                next_in_topk = torch.multinomial(topk_vals, num_samples=1)
                next_token = torch.gather(topk_idx, -1, next_in_topk)
            else:
                next_token = torch.multinomial(probs, num_samples=1) 

            idx = torch.cat((idx, next_token), dim=1)

        return idx