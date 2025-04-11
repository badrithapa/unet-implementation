import torch
import argparse
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from unet.model import UNet  # Assuming unet is installed
from unet.utils import load_checkpoint  # Import the utility function

# Default values (can be overridden by command-line arguments)
DEFAULT_CHECKPOINT_PATH = "./model_ckpt/my_checkpoint.pth"
DEFAULT_IMAGE_HEIGHT = 160
DEFAULT_IMAGE_WIDTH = 240
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Removed custom load_model function


def preprocess_image(image_path, height, width):
    """
    preprocess_image loads and preprocesses an image for inference.

    Parameters
    ----------
    image_path : str
        Path to the input image.
    height : int
        height of the image.
    width : _type_
        width of the image

    Returns
    -------
    torch.Tensor
        Preprocessed input tensor.

    Examples:
    ---------
    >>> image_path = "path/to/image.jpg"
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    transform = A.Compose(
        [
            A.Resize(height=height, width=width),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )

    augmented = transform(image=img)
    input_tensor = (
        augmented["image"].unsqueeze(0).to(DEVICE)
    )  # Add batch dimension
    return input_tensor


def perform_inference(model, input_tensor):
    """
    perform_inference performs inference on the input tensor.

    Parameters
    ----------
    model : torch.nn.Module
        Model to perform inference on
    input_tensor : torch.Tensor
        Input tensor to perform inference on

    Returns
    -------
    torch.Tensor
        Output tensor from the model.

    Examples:
    ---------
    >>> input_tensor = preprocess_image(image_path, height, width)
    """
    with torch.no_grad():
        with torch.amp.autocast(device_type=DEVICE):
            preds = torch.sigmoid(model(input_tensor))
            preds = (preds > 0.5).float()  # Apply threshold
    return preds


def save_prediction(prediction_tensor, output_path):
    """
    save_prediction saves the prediction tensor as an image


    Parameters
    ----------
    prediction_tensor : torch.Tesnor
        predicted tensor from the model
    output_path : _type_
        path to save the images.

    Returns:
    --------
    None

    Examples:
    --------
    >>> save_prediction(prediction_tensor, output_path)
    """
    # Remove batch dimension, move to CPU, convert to numpy
    prediction_np = prediction_tensor.squeeze(0).squeeze(0).cpu().numpy()
    # Convert to 8-bit image (0 or 255)
    prediction_img = (prediction_np * 255).astype(np.uint8)
    cv2.imwrite(output_path, prediction_img)
    print(f"Prediction saved to {output_path}")


def inference(
    image_path: str,
    checkpoint_path: str,
    output_path: str,
    height: int,
    width: int,
) -> None:
    """
    Main inference function.

    Parameters:
    -----------
    image_path : str
        Path to the input image.
    checkpoint_path : str
        Path to the model checkpoint file.
    output_path : str
        Path to save the output prediction mask.
    height : int
        Image height for resizing.
    width : int
        Image width for resizing.

    Returns:
    --------
    None

    Raises:
    -------
    FileNotFoundError
        If the checkpoint file is not found.
    Exception
        If there is an error loading the checkpoint.

    Examples
    --------
    >>> inference("image.jpg", "checkpoint.pth", "output.jpg", 256, 256)
    """
    # Instantiate the model
    model = UNet(in_channels=3, out_channels=1).to(DEVICE)

    # Load the checkpoint using the utility function
    print(f"Loading checkpoint from: {checkpoint_path}")
    try:
        load_checkpoint(
            torch.load(checkpoint_path, map_location=DEVICE), model
        )
        print("Checkpoint loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    model.eval()  # Ensure model is in evaluation mode after loading checkpoint

    print(f"Loading and preprocessing image: {image_path}")
    input_tensor = preprocess_image(image_path, height, width)

    print("Performing inference...")
    prediction = perform_inference(model, input_tensor)

    print(f"Saving prediction to: {output_path}")
    save_prediction(prediction, output_path)

    print("Inference complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform inference using a trained UNet model."
    )
    parser.add_argument(
        "image_path", type=str, help="Path to the input image."
    )
    parser.add_argument(
        "output_path", type=str, help="Path to save the output mask."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT_PATH,
        help=f"Path to the model checkpoint file (default: {DEFAULT_CHECKPOINT_PATH}).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_IMAGE_HEIGHT,
        help=f"Image height for resizing (default: {DEFAULT_IMAGE_HEIGHT}).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_IMAGE_WIDTH,
        help=f"Image width for resizing (default: {DEFAULT_IMAGE_WIDTH}).",
    )

    args = parser.parse_args()

    inference(
        args.image_path,
        args.checkpoint,
        args.output_path,
        args.height,
        args.width,
    )
