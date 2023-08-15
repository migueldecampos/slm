# adapted from https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py
import torch
import torch.nn as nn
from torch.nn import functional as F

from constants import PREV_TOKEN, NEXT_TOKEN, WRONG_PREDICTION_MODE


@torch.no_grad()
def estimate_loss(model, batch_iterators_dict, eval_iters):
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
