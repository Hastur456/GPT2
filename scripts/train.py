import argparse
import os
import torch
from transformers import GPT2Tokenizer
from src.model.gpt_model import GPT, load_model
from src.data.datamodule import GPT2DataModule
from src.training.trainer import GPT2Trainer
from src.model.config import GPTConfig


def args_parse():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--max_tokens', type=int, default=50)
    parser.add_argument('--temperature', type=float, default=1.)
    parser.add_argument('--model_path', type=str, default="")
    parser.add_argument('--model_type', type=str, default='gpt2')
    parser.add_argument('--weights_path', type=str, default='')
    parser.add_argument('--use_val', type=bool, default=True)
    parser.add_argument('--dataset_path', type=str, default='input.txt')
    args = parser.parse_args()
    return args


def train(args=None):
    if args is None:
        args = args_parse()

    model = load_model(args.model_path, args.model_type)

    tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")

    dataset = GPT2DataModule(args.dataset_path, tokenizer)
    dataset.setup()
    train_dataloader = dataset.train_dataloader()
    val_dataloader = dataset.val_dataloader() if args.use_val else None

    config = GPTConfig()
    model = GPT(config)

    trainer = GPT2Trainer(tokenizer, model, train_dataloader, val_dataloader=val_dataloader)

    trainer.fit(
        save_dir = args.weights_path + "/best_weights",
        save_every = args.weights_path + "/all_weights"
        )


if __name__ == "__main__":
    train()