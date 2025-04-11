import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# Imports assuming 'unet' is installed as a package
from unet.model import UNet
from unet.utils import (
    load_checkpoint,
    save_checkpoint,
    check_accuracy,
    get_loaders,
    save_predictions_as_imgs,
)

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 3
NUM_WORKERS = 2
IMAGE_HEIGHT = 160  # 1280 originally
IMAGE_WIDTH = 240  # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "dataset/train_images/"
TRAIN_MASK_DIR = "dataset/train_masks/"
VAL_IMG_DIR = "dataset/val_images/"
VAL_MASK_DIR = "dataset/val_masks/"


def train_fn(loader, model, optimizer, loss_fn, scaler) -> None:
    """
    train_fn is a function that trains the model for one epoch.


    Parameters
    ----------
    loader : _Iterator_
        dataloader for the training set.
    model : UNet
        The UNet model to be trained.
    optimizer : torch.optim.Optimizer
        The optimizer to be used for training.
    loss_fn : torch.nn.Module
        The loss function to be used for training.
    scaler : torch.cuda.amp.GradScaler
        The gradient scaler for mixed precision training.

    Returns
    -------
    None
        This function does not return anything.

    Examples
    --------
    >>> train_fn(loader, model, optimizer, loss_fn, scaler)

    """

    loop = tqdm(loader, leave=True)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(DEVICE)
        targets = targets.float().unsqueeze(1).to(DEVICE)

        # Forward pass with automatic mixed precision
        with torch.amp.autocast(device_type=DEVICE):
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())


def main() -> None:
    """
    main is a function that trains the model.

    Parameters
    ----------
    None

    Returns
    -------
    None
        This function does not return anything.

    Examples
    --------
    >>> main()
    >>> assert isinstance(main(), None)
    """
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = UNet(in_channels=3, out_channels=1).to(DEVICE)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transform,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    scaler = torch.amp.GradScaler()

    if LOAD_MODEL:
        # Assuming checkpoints are saved relative to the project root now
        load_checkpoint(torch.load("./model_ckpt/my_checkpoint.pth"), model)

    check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.amp.GradScaler() # Re-initializing scaler, might be redundant

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        checkpoint_state = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        # Assuming checkpoints are saved relative to the project root now
        save_checkpoint(
            checkpoint_state,
            filename="./model_ckpt/my_checkpoint.pth",
        )

        check_accuracy(val_loader, model, device=DEVICE)

        # Assuming saved images are relative to the project root now
        save_predictions_as_imgs(
            val_loader, model, folder="./saved_images/", device=DEVICE
        )


if __name__ == "__main__":
    main()
    
