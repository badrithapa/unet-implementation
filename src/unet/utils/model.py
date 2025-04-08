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
