import argparse
import torch
from types import SimpleNamespace
from transformers import GPT2Tokenizer
from src.model.gpt_model import GPT, load_model
from src.inference.generator import GPT2Inference


def args_parse():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, default="Hello, I am a language model,")
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--max_tokens', type=int, default=50)
    parser.add_argument('--temperature', type=float, default=1.)
    parser.add_argument('--model_path', type=str, default="")
    parser.add_argument('--model_type', type=str, default='gpt2')
    args = parser.parse_args()
    return args


def inference(args=None):
    if args is None:
        args = args_parse()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")

    model = load_model(args.model_path, args.model_type)
    
    model = model.to(device)
    inferencer = GPT2Inference(model, tokenizer, device)

    output = inferencer.generate_sequence(
                    args.prompt, 
                    max_new_tokens=args.max_tokens,
                    top_k=args.top_k,
                    temperature=args.temperature)

    output_sentences = tokenizer.decode(output[0])
    print(output_sentences)


if __name__ == "__main__":
    inference()