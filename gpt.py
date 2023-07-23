# adapted from https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py
import time
import torch
import torch.nn as nn
from torch.nn import functional as F


# Prediction modes
PREV_TOKEN = "previous_token"
NEXT_TOKEN = "next_token"
BIDIRECTIONAL = "bidirectional"
PREDICTION_MODES = [PREV_TOKEN, NEXT_TOKEN, BIDIRECTIONAL]


def get_shakespeare_train_val_data():
    with open("tiny_shakespeare.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # here are all the unique characters that occur in this text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    # create a mapping from characters to integers
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [
        stoi[c] for c in s
    ]  # encoder: take a string, output a list of integers
    decode = lambda l: "".join(
        [itos[i] for i in l]
    )  # decoder: take a list of integers, output a string

    # Train and test splits
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))  # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]
    return vocab_size, train_data, val_data, encode, decode


# data loading
def get_batch(data, block_size, batch_size, prediction_mode, device):
    # generate a small batch of data of inputs x and targets y
    low = 0 if prediction_mode == NEXT_TOKEN else 1
    ix = torch.randint(low=low, high=len(data) - block_size, size=(batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    x = x.to(device)
    if prediction_mode in (PREV_TOKEN, BIDIRECTIONAL):
        yp = torch.stack([data[i - 1 : i + block_size - 1] for i in ix])
        yp = yp.to(device)
        y = yp
    if prediction_mode in (NEXT_TOKEN, BIDIRECTIONAL):
        yn = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
        yn = yn.to(device)
        y = yn
    if prediction_mode == BIDIRECTIONAL:
        y = yp, yn
    return x, y


@torch.no_grad()
def estimate_loss(
    data_dict, eval_iters, block_size, batch_size, prediction_mode, device
):
    """
    receives a dict containing data splits and their names
    e.g. {'train': <data>, 'val: <data>'}
    """
    out = {}
    model.eval()
    for split in data_dict:
        losses = torch.zeros(eval_iters, 2)
        for k in range(eval_iters):
            x, y = get_batch(
                data_dict[split], block_size, batch_size, prediction_mode, device
            )
            if prediction_mode == BIDIRECTIONAL:
                for i, yt, mode in zip(range(len(y)), y, ("previous", "next")):
                    logits, loss_mode = model(x, yt, bidirectional_mode=mode)
                    losses[k, i] = loss_mode.item()
            else:
                logits, loss = model(x, y)
                losses[k, 0] = loss.item()
        out[split] = (
            losses.mean() if prediction_mode == BIDIRECTIONAL else losses[:, 0].mean()
        )
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
        if prediction_mode in (PREV_TOKEN, BIDIRECTIONAL):
            self.register_buffer("triu", torch.triu(torch.ones(block_size, block_size)))
            self.register_buffer("tri", self.triu)
        if prediction_mode in (NEXT_TOKEN, BIDIRECTIONAL):
            self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
            self.register_buffer("tri", self.tril)
        if prediction_mode == BIDIRECTIONAL:
            self.register_buffer(
                "pyr",
                torch.cat(
                    (
                        torch.repeat_interleave(
                            torch.flip(
                                torch.tril(
                                    torch.ones(block_size // 2, block_size // 2)
                                ),
                                dims=(1,),
                            ),
                            torch.tensor(
                                [2 for i in range(block_size // 2)], dtype=torch.long
                            ),
                            dim=0,
                        ),
                        torch.cat(
                            (
                                torch.zeros(1, block_size // 2),
                                torch.repeat_interleave(
                                    torch.tril(
                                        torch.ones(block_size // 2, block_size // 2)
                                    ),
                                    torch.tensor(
                                        [2 for i in range(block_size // 2 - 1)] + [1],
                                        dtype=torch.long,
                                    ),
                                    dim=0,
                                ),
                            ),
                            dim=0,
                        ),
                    ),
                    dim=1,
                ),
            )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, bidirectional_mode):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)  # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        # compute attention scores ("affinities")
        wei = (
            q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        )  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        if not bidirectional_mode:
            wei = wei.masked_fill(self.tri[:T, :T] == 0, float("-inf"))  # (B, T, T)
        elif bidirectional_mode == "previous":
            wei = wei.masked_fill(self.triu[:T, :T] == 0, float("-inf"))  # (B, T, T)
        elif bidirectional_mode == "next":
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        elif bidirectional_mode == "bi":
            wei = wei.masked_fill(
                self.pyr[
                    :T,
                    (self.block_size - T) // 2 : self.block_size
                    + (-(self.block_size - T)) // 2,
                ]
                == 0,
                float("-inf"),
            )  # (B, T, T)
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

    def forward(self, x, bidirectional_mode):
        out = torch.cat(
            [h(x, bidirectional_mode=bidirectional_mode) for h in self.heads], dim=-1
        )
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
        x, bidirectional_mode = inp
        x = x + self.sa(self.ln1(x), bidirectional_mode)
        x = x + self.ffwd(self.ln2(x))
        return (x, bidirectional_mode)


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

    def forward(self, idx, targets=None, bidirectional_mode=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=self.device)
        )  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x, _ = self.blocks((x, bidirectional_mode))  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            if self.prediction_mode in (PREV_TOKEN, NEXT_TOKEN):
                logits = logits.view(B * T, C)
                targets = targets.view(B * T)
            else:
                if bidirectional_mode == "previous":
                    logits = logits.view(B * T, C)[:, : C // 2]
                if bidirectional_mode == "next":
                    logits = logits.view(B * T, C)[:, C // 2 :]
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
            elif self.prediction_mode == BIDIRECTIONAL:
                _, l = idx.shape
                if l > self.block_size:
                    break
                else:
                    idx_cond = idx
            # get the predictions
            if self.prediction_mode == BIDIRECTIONAL:
                logits, loss = self(idx_cond, bidirectional_mode="bi")
            else:
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
                logits_prev = logits[:, 0, : C // 2]  # becomes (B, C)
                probs = F.softmax(logits_prev, dim=-1)  # (B, C)
                idx_prev = torch.multinomial(probs, num_samples=1)  # (B, 1)
                logits_next = logits[:, -1, C // 2 :]  # becomes (B, C)
                probs = F.softmax(logits_next, dim=-1)  # (B, C)
                idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
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
    train_data,
    val_data,
    learning_rate,
    max_iters,
    eval_interval,
    eval_iters,
    block_size,
    batch_size,
    prediction_mode,
    device,
):
    m = model.to(device)
    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
    tic = time.time()
    loss_timeline = list()
    for iter in range(max_iters):
        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(
                {"train": train_data, "val": val_data},
                eval_iters,
                block_size,
                batch_size,
                prediction_mode,
                device,
            )
            loss_timeline.append(losses)
            print(
                f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )

        # sample a batch of data
        x, y = get_batch(train_data, block_size, batch_size, prediction_mode, device)
        # evaluate the loss
        if prediction_mode == BIDIRECTIONAL:
            loss = 0
            for target, mode in zip(y, ("previous", "next")):
                logits, loss_mode = m(x, target, bidirectional_mode=mode)
                loss += loss_mode
        else:
            logits, loss = m(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    print("Training time:", time.time() - tic)
    return loss_timeline


if __name__ == "__main__":
    import sys
    import json

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
    # End of handling of arguments

    # Checking gpu availability
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    # ------------
    vocab_size, train_data, val_data, encode, decode = get_shakespeare_train_val_data()

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
            "vocab_size": vocab_size,
            "batch_size": 32,  # how many independent sequences will we process in parallel?
            "block_size": 256,  # what is the maximum context length for predictions?
            "max_iters": 1000,
            "eval_interval": 200,
            "learning_rate": 3e-4,
            "eval_iters": 50,
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

    loss_timeline = loss_timeline + train(
        model,
        train_data,
        val_data,
        hyperparameters["learning_rate"],
        hyperparameters["max_iters"],
        hyperparameters["eval_interval"],
        hyperparameters["eval_iters"],
        hyperparameters["block_size"],
        hyperparameters["batch_size"],
        hyperparameters["prediction_mode"],
        device,
    )

    checkpoint_id = "checkpoints/checkpoint_{}".format(int(time.time()))
    with open("{}.pt".format(checkpoint_id), "wb") as f:
        torch.save(
            {
                "hyperparameters": hyperparameters,
                "model": model,
                "loss_timeline": loss_timeline,
            },
            f,
        )
    with open("{}.json".format(checkpoint_id), "w") as f:
        json.dump(
            {
                "hyperparameters": hyperparameters,
                "loss_timeline": [
                    {k: loss_dict[k].item() for k in loss_dict}
                    for loss_dict in loss_timeline
                ],
            },
            f,
        )

    # generate from the model
    tic = time.time()
    if hyperparameters["prediction_mode"] == NEXT_TOKEN:
        context = torch.tensor([encode("KING:")], device=device)
    elif hyperparameters["prediction_mode"] == PREV_TOKEN:
        context = torch.tensor([encode("the king.")], device=device)
    else:
        context = torch.tensor([encode("KING:")], device=device)
    generation = decode(model.generate(context, max_new_tokens=500)[0].tolist())
    print(generation)
    print("\nGenerate time:", time.time() - tic)
    with open("{}.txt".format(checkpoint_id), "w") as f:
        f.write(generation)
