import functools
import json
import torch
import os.path

from constants import (
    PREV_TOKEN,
    NEXT_TOKEN,
    TOKENIZED_TINYSTORIES_VAL,
    TOKENIZED_TINYSTORIES_TRAIN,
    TINYSTORIES_METADATA_JSON,
    TINYSTORIES_NOT_TOKENIZED,
)


def _create_char_encode_decode(chars):
    # create a mapping from characters to integers
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [
        stoi[c] for c in s
    ]  # encoder: take a string, output a list of integers
    decode = lambda l: "".join(
        [itos[i] for i in l]
    )  # decoder: take a list of integers, output a string
    return encode, decode


## Shakesperare ##
def _get_shakespeare_train_val_data():
    with open("tiny_shakespeare.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # here are all the unique characters that occur in this text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    encode, decode = _create_char_encode_decode(chars)

    # Train and test splits
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))  # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]
    return vocab_size, train_data, val_data, encode, decode


def get_shakespeare_batch(block_size, batch_size, prediction_mode, device, data):
    # generate a small batch of data of inputs x and targets y
    low = 0 if prediction_mode == NEXT_TOKEN else 1
    while True:
        ix = torch.randint(low=low, high=len(data) - block_size, size=(batch_size,))
        x = torch.stack([data[i : i + block_size] for i in ix])
        x = x.to(device)
        if prediction_mode == PREV_TOKEN:
            yp = torch.stack([data[i - 1 : i + block_size - 1] for i in ix])
            yp = yp.to(device)
            y = yp
        if prediction_mode == NEXT_TOKEN:
            yn = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
            yn = yn.to(device)
            y = yn
        yield x, y


def get_shakespeare():
    vocab_size, train_data, val_data, encode, decode = _get_shakespeare_train_val_data()

    get_train_batch = functools.partial(get_shakespeare_batch, data=train_data)
    get_val_batch = functools.partial(get_shakespeare_batch, data=val_data)

    return vocab_size, get_train_batch, get_val_batch, encode, decode


## Tiny Stories ##
def tokenize_tinystories(val_file_path, train_file_path):
    with open(val_file_path, "r", encoding="utf-8") as f:
        text_val = f.read()
    with open(train_file_path, "r", encoding="utf-8") as f:
        text_train = f.read()

    chars_val = set(text_val)
    chars_train = set(text_train)

    chars = chars_val | chars_train
    chars = sorted(list(chars))
    print(chars)
    vocab_size = len(chars)
    print(vocab_size)
    encode, _ = _create_char_encode_decode(chars)

    data_val = encode(text_val)
    with open("./{}".format(TOKENIZED_TINYSTORIES_VAL), "w") as f:
        f.write("\n".join([str(i) for i in data_val]))
    data_train = encode(text_train)
    with open("./{}".format(TOKENIZED_TINYSTORIES_TRAIN), "w") as f:
        for i in data_train:
            f.write("{}\n".format(i))

    with open("./{}".format(TINYSTORIES_METADATA_JSON), "w") as f:
        json.dump({"chars": chars, "vocab_size": vocab_size}, f)
    return


def get_tinystories_batch(block_size, batch_size, prediction_mode, device, file_path):
    while True:
        count = 0
        tokens = list()
        with open(file_path, "r") as f:
            for line in f:
                tokens.append(int(line.rstrip("\n")))
                count += 1
                if count == block_size * batch_size + 1:
                    fst = torch.stack(
                        [
                            torch.tensor(tokens[i * block_size : (i + 1) * block_size])
                            for i in range(batch_size)
                        ]
                    )
                    scd = torch.stack(
                        [
                            torch.tensor(
                                tokens[1 + i * block_size : 1 + (i + 1) * block_size]
                            )
                            for i in range(batch_size)
                        ]
                    )
                    if prediction_mode == NEXT_TOKEN:
                        x = fst.to(device)
                        y = scd.to(device)
                    else:
                        x = scd.to(device)
                        y = fst.to(device)
                    count = 0
                    tokens = list()
                    yield x, y


def get_tinystories():
    if (
        not os.path.isfile(TINYSTORIES_METADATA_JSON)
        or not os.path.isfile(TOKENIZED_TINYSTORIES_VAL)
        or not os.path.isfile(TOKENIZED_TINYSTORIES_TRAIN)
    ):
        raise ValueError(TINYSTORIES_NOT_TOKENIZED)

    with open("./{}".format(TINYSTORIES_METADATA_JSON), "r") as f:
        tinystories_metadata = json.load(f)

    encode, decode = _create_char_encode_decode(tinystories_metadata["chars"])

    get_train_batch = functools.partial(
        get_tinystories_batch, file_path="./{}".format(TOKENIZED_TINYSTORIES_TRAIN)
    )
    get_val_batch = functools.partial(
        get_tinystories_batch, file_path="./{}".format(TOKENIZED_TINYSTORIES_VAL)
    )

    return (
        tinystories_metadata["vocab_size"],
        get_train_batch,
        get_val_batch,
        encode,
        decode,
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 3:
        tokenize_tinystories(val_file_path=sys.argv[1], train_file_path=sys.argv[2])
