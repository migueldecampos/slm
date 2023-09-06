import time
import torch

from gpt import GPTLanguageModel, Block, FeedFoward, MultiHeadAttention, Head
from utils import get_device
from constants import TINYSTORIES_DATASOURCE
from data_loaders import get_tinystories, get_shakespeare


def infer(model, encode, decode, prompt, max_new_tokens, device):
    tic = time.time()
    context = torch.tensor([encode(prompt)], device=device)
    generation = decode(
        model.generate(context, max_new_tokens=max_new_tokens)[0].tolist()
    )
    print("Generation took", time.time() - tic)
    return generation


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 4:
        print("We need model checkpoint and prompt.")
        exit()

    device = get_device()
    with open(sys.argv[1], "rb") as f:
        checkpoint = torch.load(f)
    hyperparameters = checkpoint["hyperparameters"]
    model = checkpoint["model"]
    model.device = device

    if hyperparameters["datasource"] == TINYSTORIES_DATASOURCE:
        _, _, _, encode, decode = get_tinystories()
    else:
        _, _, _, encode, decode = get_shakespeare()

    max_new_tokens = int(sys.argv[2])
    prompt = sys.argv[3]
    print("Prompt:", prompt)
    print("Max new tokens:", max_new_tokens)
    print()

    generation = infer(
        model,
        encode,
        decode,
        max_new_tokens=max_new_tokens,
        prompt=prompt,
        device=device,
    )
    print(generation)
    print()
