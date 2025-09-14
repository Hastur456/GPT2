from torch.utils.data import Dataset
import torch


class GPT2Dataset(Dataset):
    def __init__(self, text, tokenizer, block_size):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        print(f"Dataset size: {len(self.tokens)} tokens")

    def __len__(self):
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx):
        chunk = self.tokens[idx : idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y