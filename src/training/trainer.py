import torch
from torch import nn
import torch.nn.functional as F
from tqdm.autonotebook import tqdm
from src.model.gpt_model import GPT
import os


class GPT2Trainer():
    def __init__(self, tokenizer, model: GPT, train_dataloader, val_dataloader=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = tokenizer
        self.model = model.to(self.device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.optimizer = None
        self.lr_scheduler = None
        self.loss_fn = nn.CrossEntropyLoss()


    def compile(self, optimizer=None, learning_rate=5e-5, weight_decay=0.01, lr_scheduler=None):
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        else:
            self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler


    def train_step(self, batch):
        self.model.train()
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)

        self.optimizer.zero_grad()

        with torch.amp.autocast(self.device.type, dtype=torch.bfloat16):
            outputs = self.model(x, target=y)
            loss = outputs.loss if getattr(outputs, "loss", None) is not None else \
                self.loss_fn(outputs.logits.view(-1, outputs.logits.size(-1)), y.view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return loss.item()

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        iters_count = 0
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, desc="Validation")):
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)

                with torch.amp.autocast(self.device.type, dtype=torch.bfloat16):
                    outputs = self.model(x, target=y)
                    loss = outputs.loss if getattr(outputs, "loss", None) is not None else \
                        self.loss_fn(outputs.logits.view(-1, outputs.logits.size(-1)), y.view(-1))

                total_loss += loss.item()
                iters_count += 1

        return total_loss / max(1, iters_count)


    def save_model(self, directory: str, name: str, epoch=None, train_losses=None, val_losses=None):
        if not directory:
            directory = "weights"
        os.makedirs(directory, exist_ok=True)


    def fit(self, epochs=10, learning_rate=5e-5, iter_step=10000, save_dir=None, save_every=None):
        if self.optimizer is None:
            self.compile(learning_rate=learning_rate)

        train_losses = []
        val_losses = []
        best_val_loss = float("inf")

        for epoch in range(1, epochs + 1):
            print(f"Epoch {epoch}/{epochs}")
            progress_bar = tqdm(self.train_dataloader, desc="Training")
            for iter_idx, batch in enumerate(progress_bar, start=1):
                loss = self.train_step(batch)
                train_losses.append(loss)
                progress_bar.set_postfix({"loss": loss})

                if save_every and (iter_idx % iter_step == 0):
                    self.save_model(save_every, f"gpt2-iter-{iter_idx}-loss-{loss:.4f}",
                                    epoch=epoch, train_losses=train_losses, val_losses=val_losses)

            if self.val_dataloader:
                val_loss = self.evaluate(self.val_dataloader)
                val_losses.append(val_loss)
                print(f"Epoch {epoch} validation loss: {val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if save_dir:
                        self.save_model(save_dir, "best_model",
                                        epoch=epoch, train_losses=train_losses, val_losses=val_losses)