import torch


def save_checkpoint(state: dict, filename="my_checkpoint.pth.tar") -> None:
    """
    save_checkpoint is a function that saves the current state of the model

    Parameters
    ----------
    state : dict
        model state dict
    filename : str, optional
        name of the checkpoint file, by default "my_checkpoint.pth.tar"

    Returns
    -------
    None
        This function does not return anything.

    Examples
    --------
    >>> save_checkpoint(state, filename="my_checkpoint.pth.tar")
    >>> assert isinstance(save_checkpoint(state, filename="my_checkpoint.pth.tar"), None)
    """
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint: dict, model: torch.nn.Module) -> None:
    """
    load_checkpoint is a function that loads the model state from a checkpoint.

    Parameters
    ----------
    checkpoint : dict
        checkpoint dictionary containing the model state
    model : torch.nn.Module
        model to load the state into

    Returns
    -------
    None
        This function does not return anything.
    Examples
    --------
    >>> load_checkpoint(checkpoint, model)
    >>> assert isinstance(load_checkpoint(checkpoint, model), None)

    """
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def check_accuracy(loader, model, device="cuda") -> None:
    """
    check_accuracy is a function that calculates the accuracy and
    the dice score of the model

    Parameters
    ----------
    loader : Dataloader
        dataloader for the validation set
    model : torch.nn.Module
        model to be evaluated
    device : str, optional
        device to be used, by default "cuda"
    """
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

    print(
        f"Got {num_correct}/{num_pixels} with acc {
            num_correct / num_pixels * 100:.2f
        }"
    )
    print(f"Dice score: {dice_score / len(loader)}")
    model.train()
