from torch.utils.data import DataLoader, random_split
from .dataset import GPT2Dataset


class GPT2DataModule():
    def __init__(self, text_path, tokenizer, block_size=1024, batch_size=1, train_split=0.9):
        self.text_path = text_path
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.batch_size = batch_size
        self.train_split = train_split
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None

    def setup(self):
        with open(self.text_path, "r", encoding="utf-8") as f:
            text = f.read()

        self.dataset = GPT2Dataset(text, self.tokenizer, self.block_size)

        train_size = int(self.train_split * len(self.dataset))
        val_size = len(self.dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(self.dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
        