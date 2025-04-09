from typing import Callable, Tuple

import torch
import torchvision
from torch.utils.data import DataLoader

from ..dataset import CustomCarDataset


def get_loaders(
    train_dir: str,
    train_maskdir: str,
    val_dir: str,
    val_maskdir: str,
    batch_size: int,
    train_transform: Callable,
    val_transform: Callable,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    get_loaders is a function that returns dataloaders for training and
    validation sets.

    Parameters
    ----------
    train_dir :
        Dataset directory containing the training images.
    train_maskdir : str
        Directory containing the training masks.
    val_dir : str
        Directory containing the validation images.
    val_maskdir : str
        Directory containing the validation masks.
    batch_size : int
        Number of samples per batch.
    train_transform : Callable
        function to transform the training data.
    val_transform : Callable
        function to transform the validation data.
    num_workers : int
        how many subprocesses to use for data loading.
        0 means that the data will be loaded in the main process. (default: 4)
    pin_memory : bool
        If True, the data loader will copy Tensors into
        device/CUDA pinned memory before returning them. (default: True)

    Returns
    -------
    tuple(DataLoader, DataLoader)
        Return a tuple containing the training and validation dataloaders.
    """
    train_dataset = CustomCarDataset(
        img_dir=train_dir, mask_dir=train_maskdir, transform=train_transform
    )
    val_dataset = CustomCarDataset(
        img_dir=val_dir, mask_dir=val_maskdir, transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )
    return train_loader, val_loader


def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
) -> None:
    """
    save_predictions_as_imgs is a function that saves the predictions of the model
    as images

    Parameters
    ----------
    loader : Dataloader
        dataloader for the validation set
    model : torch.nn.Module
        model to be evaluated
    folder : str, optional
        folder to save the images, by default "saved_images/"
    device : str, optional
        device to be used, by default "cuda"
    """
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(preds, f"{folder}{idx}_pred.jpg")
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.jpg")
    model.train()
