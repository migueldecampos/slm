import time
import torch

from constants import (
    DATASOURCES,
    TINYSTORIES_DATASOURCE,
    SHAKESPEARE_DATASOURCE,
    PREV_TOKEN,
    NEXT_TOKEN,
    PREDICTION_MODES,
    WRONG_PREDICTION_MODE,
)
from gpt import GPTLanguageModel, estimate_loss
from utils import get_device, plot_loss, save_checkpoint
from data_loaders import get_shakespeare, get_tinystories


def train(
    model,
    train_batch_iterator,
    val_batch_iterator,
    hyperparameters,
    device,
    checkpoint_id,
    iterations_from_previous_run,
):
    m = model.to(device)
    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(m.parameters(), lr=hyperparameters["learning_rate"])
    tic = time.time()
    loss_timeline = list()
    for iter in range(hyperparameters["max_iters"]):
        total_number_of_iterations_over_all_runs = iterations_from_previous_run + iter
        # every once in a while evaluate the loss on train and val sets
        if (
            iter % hyperparameters["eval_interval"] == 0
            or iter == hyperparameters["max_iters"] - 1
        ):
            losses = estimate_loss(
                model,
                {"train": train_batch_iterator, "val": val_batch_iterator},
                hyperparameters["eval_iters"],
            )
            loss_timeline.append((iter, losses))
            print(
                f"step {total_number_of_iterations_over_all_runs}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )
        if (
            iter % hyperparameters["checkpoint_interval"] == 0
            or iter == hyperparameters["max_iters"] - 1
        ):
            save_checkpoint(
                file_path="{checkpoint_id}_{num:06}".format(
                    checkpoint_id=checkpoint_id,
                    num=total_number_of_iterations_over_all_runs,
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
    if not args.get("datasource") or args["datasource"] not in DATASOURCES:
        print(
            "--data_source has to be one of: {}.".format(
                ", ".join(["<{}>".format(mode) for mode in DATASOURCES])
            )
        )
        exit()
    # End of handling of arguments

    # Checking gpu availability
    device = get_device()
    # ------------
    if args["datasource"] == TINYSTORIES_DATASOURCE:
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
            "datasource": args["datasource"],
            "vocab_size": vocab_size,
            "batch_size": 64,  # how many independent sequences will we process in parallel?
            "block_size": 512,  # what is the maximum context length for predictions?
            "max_iters": 6000,
            "eval_interval": 250,
            "eval_iters": 50,
            "checkpoint_interval": 500,
            "learning_rate": 3e-4,
            "n_embd": 384,
            "n_head": 4,
            "n_layer": 6,
            "dropout": 0.2,
            "prediction_mode": args["prediction_mode"],
        }
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

    print("hyperparameters:")
    print(hyperparameters)
    print()
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
    iterations_from_previous_run = 0
    if "from_checkpoint" in args:
        iterations_from_previous_run = hyperparameters["max_iters"]
        number_of_batches_from_previous_run = (
            hyperparameters["max_iters"]
            + hyperparameters["max_iters"] // hyperparameters["eval_interval"]
            + 2
        )
        print(
            "Dry run of {} batches from previous run.".format(
                number_of_batches_from_previous_run
            )
        )
        tic = time.time()
        for i in range(number_of_batches_from_previous_run):
            next(train_batch_iterator)
        print("Dry run took {}.".format(int(time.time() - tic)))

    checkpoint_id = "checkpoints/checkpoint_{}".format(int(time.time()))
    loss_timeline = loss_timeline + train(
        model,
        train_batch_iterator,
        val_batch_iterator,
        hyperparameters,
        device,
        checkpoint_id,
        iterations_from_previous_run,
    )

    # generate from the model
    tic = time.time()
    if hyperparameters["prediction_mode"] == NEXT_TOKEN:
        context = (
            torch.tensor([encode("KING:")], device=device)
            if hyperparameters["datasource"] == SHAKESPEARE_DATASOURCE
            else torch.tensor([encode("Once upon a time")], device=device)
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
