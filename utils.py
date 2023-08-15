import json
import matplotlib.pyplot as plt
import pandas as pd
import torch


def plot_loss(loss_timeline):
    iter_list = list()
    train_losses = list()
    val_losses = list()
    for iters, l in loss_timeline:
        iter_list.append(iters)
        train_losses.append(
            l["train"] if type(l["train"]) is int else l["train"].item()
        )
        val_losses.append(l["val"] if type(l["val"]) is int else l["val"].item())

    df = pd.DataFrame({"iters": iter_list, "train": train_losses, "val": val_losses})
    df.set_index(keys="iters", inplace=True)
    df.plot()
    plt.show()
    return


def save_checkpoint(file_path, model, hyperparameters, loss_timeline):
    with open("{}.pt".format(file_path), "wb") as f:
        torch.save(
            {
                "hyperparameters": hyperparameters,
                "model": model,
                "loss_timeline": loss_timeline,
            },
            f,
        )
    with open("{}.json".format(file_path), "w") as f:
        json.dump(
            {
                "hyperparameters": hyperparameters,
                "loss_timeline": [
                    (iters, {k: loss_dict[k].item() for k in loss_dict})
                    for iters, loss_dict in loss_timeline
                ],
            },
            f,
        )
    return
