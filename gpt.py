# adapted from https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py
import time
import torch
import torch.nn as nn
from torch.nn import functional as F

from constants import (
    DATASOURCES,
    TINYSTORIES_DATASOURCE,
    SHAKESPEARE_DATASOURCE,
    PREV_TOKEN,
    NEXT_TOKEN,
    PREDICTION_MODES,
    WRONG_PREDICTION_MODE,
)
from utils import plot_loss, save_checkpoint
from data_loaders import get_shakespeare, get_tinystories


@torch.no_grad()
def estimate_loss(batch_iterators_dict, eval_iters):
    """
    receives a dict containing data splits and their names
    e.g. {'train': train_batch_iterator, 'val: val_batch_iterator'}
    """
    out = {}
    model.eval()
    for split in batch_iterators_dict:
        losses = torch.zeros(eval_iters, 2)
        for k in range(eval_iters):
            x, y = next(batch_iterators_dict[split])
            logits, loss = model(x, y)
            losses[k, 0] = loss.item()
        out[split] = losses[:, 0].mean()
    model.train()
    return out


class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, head_size, n_embd, block_size, dropout, prediction_mode):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.block_size = block_size
        if prediction_mode == PREV_TOKEN:
            self.register_buffer("triu", torch.triu(torch.ones(block_size, block_size)))
            self.register_buffer("tri", self.triu)
        if prediction_mode == NEXT_TOKEN:
            self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
            self.register_buffer("tri", self.tril)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)  # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        # compute attention scores ("affinities")
        wei = (
            q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        )  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tri[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(
        self, n_embd, num_heads, head_size, block_size, dropout, prediction_mode
    ):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                Head(head_size, n_embd, block_size, dropout, prediction_mode)
                for _ in range(num_heads)
            ]
        )
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head, block_size, dropout, prediction_mode):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(
            n_embd, n_head, head_size, block_size, dropout, prediction_mode
        )
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, inp):
        x = inp
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        n_embd,
        block_size,
        n_head,
        n_layer,
        dropout,
        prediction_mode,
        device,
    ):
        super().__init__()
        self.device = device
        self.block_size = block_size
        self.prediction_mode = prediction_mode

        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[
                Block(
                    n_embd,
                    n_head=n_head,
                    block_size=block_size,
                    dropout=dropout,
                    prediction_mode=self.prediction_mode,
                )
                for _ in range(n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        if self.prediction_mode in (PREV_TOKEN, NEXT_TOKEN):
            self.lm_head = nn.Linear(n_embd, vocab_size)
        else:
            self.lm_head = nn.Linear(n_embd, vocab_size * 2)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=self.device)
        )  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            if self.prediction_mode == PREV_TOKEN:
                # crop idx to the first block_size tokens
                idx_cond = idx[:, : self.block_size]
            elif self.prediction_mode == NEXT_TOKEN:
                # crop idx to the last block_size tokens
                idx_cond = idx[:, -self.block_size :]
            # get the predictions
            logits, loss = self(idx_cond)
            B, T, C = logits.shape
            idx_prev = None
            idx_next = None
            if self.prediction_mode == PREV_TOKEN:
                # focus only on first "time step"
                logits_prev = logits[:, 0, :]  # becomes (B, C)
                # apply softmax to get probabilities
                probs = F.softmax(logits_prev, dim=-1)  # (B, C)
                # sample from the distribution
                idx_prev = torch.multinomial(probs, num_samples=1)  # (B, 1)
            elif self.prediction_mode == NEXT_TOKEN:
                # focus only on the last "time step"
                logits_next = logits[:, -1, :]  # becomes (B, C)
                # apply softmax to get probabilities
                probs = F.softmax(logits_next, dim=-1)  # (B, C)
                # sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            else:
                raise ValueError(WRONG_PREDICTION_MODE)
            # append sampled index to the running sequence
            if idx_prev is not None:
                # either at the start
                idx = torch.cat((idx_prev, idx), dim=1)  # (B, T+1)
            if idx_next is not None:
                # or at the end
                idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


def train(
    model,
    train_batch_iterator,
    val_batch_iterator,
    hyperparameters,
    device,
    checkpoint_id,
):
    m = model.to(device)
    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(m.parameters(), lr=hyperparameters["learning_rate"])
    tic = time.time()
    loss_timeline = list()
    for iter in range(hyperparameters["max_iters"]):
        # every once in a while evaluate the loss on train and val sets
        if (
            iter % hyperparameters["eval_interval"] == 0
            or iter == hyperparameters["max_iters"] - 1
        ):
            losses = estimate_loss(
                {"train": train_batch_iterator, "val": val_batch_iterator},
                hyperparameters["eval_iters"],
            )
            loss_timeline.append((iter, losses))
            print(
                f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )
        if (
            iter % hyperparameters["checkpoint_interval"] == 0
            or iter == hyperparameters["max_iters"] - 1
        ):
            save_checkpoint(
                file_path="{checkpoint_id}_{num:06}".format(
                    checkpoint_id=checkpoint_id, num=iter
                ),
                model=model,
                hyperparameters=hyperparameters,
                loss_timeline=loss_timeline,
            )

        # sample a batch of data
        x, y = next(train_batch_iterator)
        # evaluate the loss
        logits, loss = m(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    print("Training time:", time.time() - tic)
    return loss_timeline


if __name__ == "__main__":
    import sys

    # Handling of arguments
    args = dict()
    i = 1
    while i < len(sys.argv) - 1:
        if sys.argv[i].startswith("--"):
            args[sys.argv[i][2:]] = sys.argv[i + 1].lower()
        else:
            break
        i += 2
    if (
        not args.get("prediction_mode")
        or args["prediction_mode"] not in PREDICTION_MODES
    ):
        print(
            "--prediction_mode has to be one of: {}.".format(
                ", ".join(["<{}>".format(mode) for mode in PREDICTION_MODES])
            )
        )
        exit()
    if not args.get("data_source") or args["data_source"] not in DATASOURCES:
        print(
            "--data_source has to be one of: {}.".format(
                ", ".join(["<{}>".format(mode) for mode in DATASOURCES])
            )
        )
        exit()
    # End of handling of arguments

    # Checking gpu availability
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    # ------------
    if args["data_source"] == TINYSTORIES_DATASOURCE:
        vocab_size, get_train_batch, get_val_batch, encode, decode = get_tinystories()
    else:
        vocab_size, get_train_batch, get_val_batch, encode, decode = get_shakespeare()

    # Start from checkpoint or from scratch
    if "from_checkpoint" in args:
        with open(args["from_checkpoint"], "rb") as f:
            checkpoint = torch.load(f)
        hyperparameters = checkpoint["hyperparameters"]
        model = checkpoint["model"]
        loss_timeline = checkpoint["loss_timeline"]
        model.device = device
    else:
        # Setting seed so that we have comparable runs
        torch.manual_seed(1337)
        # hyperparameters
        hyperparameters = {
            "datasource": args["data_source"],
            "vocab_size": vocab_size,
            "batch_size": 128,  # how many independent sequences will we process in parallel?
            "block_size": 256,  # what is the maximum context length for predictions?
            "max_iters": 1000,
            "eval_interval": 250,
            "eval_iters": 50,
            "checkpoint_interval": 500,
            "learning_rate": 3e-4,
            "n_embd": 384,
            "n_head": 3,
            "n_layer": 4,
            "dropout": 0.2,
            "prediction_mode": args["prediction_mode"],
        }
        print("hyperparameters:")
        print(hyperparameters)
        print()
        model = GPTLanguageModel(
            vocab_size=hyperparameters["vocab_size"],
            n_embd=hyperparameters["n_embd"],
            block_size=hyperparameters["block_size"],
            n_head=hyperparameters["n_head"],
            n_layer=hyperparameters["n_layer"],
            dropout=hyperparameters["dropout"],
            prediction_mode=hyperparameters["prediction_mode"],
            device=device,
        )
        loss_timeline = list()

    # print the number of parameters in the model
    print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")

    train_batch_iterator = get_train_batch(
        hyperparameters["block_size"],
        hyperparameters["batch_size"],
        hyperparameters["prediction_mode"],
        device,
    )
    val_batch_iterator = get_val_batch(
        hyperparameters["block_size"],
        hyperparameters["batch_size"],
        hyperparameters["prediction_mode"],
        device,
    )
    checkpoint_id = "checkpoints/checkpoint_{}".format(int(time.time()))
    loss_timeline = loss_timeline + train(
        model,
        train_batch_iterator,
        val_batch_iterator,
        hyperparameters,
        device,
        checkpoint_id,
    )

    # generate from the model
    tic = time.time()
    if hyperparameters["prediction_mode"] == NEXT_TOKEN:
        context = (
            torch.tensor([encode("KING:")], device=device)
            if hyperparameters["datasource"] == SHAKESPEARE_DATASOURCE
            else torch.tensor([encode("Mary said")], device=device)
        )
    elif hyperparameters["prediction_mode"] == PREV_TOKEN:
        context = (
            torch.tensor([encode("the king.")], device=device)
            if hyperparameters["datasource"] == SHAKESPEARE_DATASOURCE
            else torch.tensor([encode("to the party.")], device=device)
        )
    else:
        raise ValueError(WRONG_PREDICTION_MODE)
    generation = decode(model.generate(context, max_new_tokens=250)[0].tolist())
    print(generation)
    print("\nGenerate time:", time.time() - tic)
    with open("{}.txt".format(checkpoint_id), "w") as f:
        f.write(generation)
    plot_loss(loss_timeline)
