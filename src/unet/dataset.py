import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class CustomCarDataset(Dataset):
    """
    CustomCarDataset class inherits from the PyTorch Dataset class.
    It is used to load images and their corresponding masks for training a
    segmentation model.
    The dataset consists of images and masks stored in separate directories.
    The images are expected to be in JPG format and the masks in GIF format.
    The masks are converted to binary format (0 and 1) for segmentation tasks.

    Attributes
    ----------
    img_dir : str
        Directory containing the input images.
    mask_dir : str
        Directory containing the corresponding masks.
    transform : callable, optional
        Optional transform to be applied to the images and masks.
    images : list
        List of image filenames in the img_dir.
    masks : list
        List of mask filenames in the mask_dir.
    Methods
    -------
    __len__() -> int
        Returns the number of images in the dataset.
    __getitem__(idx: int) -> tuple
        Returns a tuple containing the image and its corresponding mask
        at the specified index.

    """

    def __init__(self, img_dir, mask_dir, transform=None):
        """
        __init__ is the constructor method for the CustomCarDataset class.
        It initializes the dataset by setting the image and mask directories,
        and loading the filenames of the images and masks.

        Parameters
        ----------
        img_dir : str
            Directory containing the input images.
        mask_dir : str
            Directory containing the corresponding masks.
        transform : callable, optional
            Optional transform to be applied to the images and masks,
            by default None
        Returns
        -------
        None
        Examples
        --------
        >>> dataset = CustomCarDataset(
        ... img_dir='path/to/images',
        ... mask_dir='path/to/masks',
        ... transform=None
        ... )
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.images = os.listdir(img_dir)
        self.masks = os.listdir(mask_dir)

    def __len__(self):
        """
        __len__ is used by the built-in len() function to get the size of the
        dataset.

        Parameters
        ----------
        None

        Returns
        -------
        int
            The number of images in the dataset.
        Examples
        --------
        >>> dataset = CustomCarDataset(
        ... img_dir='path/to/images',
        ... mask_dir='path/to/masks',
        ... transform=None
        ... )
        >>> len(dataset)
        5
        >>> assert len(dataset) == 5

        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        __getitem__ is used to get a single item from the dataset.
        It loads the image and its corresponding mask from the specified

        Parameters
        ----------
        idx : int
            The index of the image and mask to be loaded.

        Returns
        -------
        tuple
            A tuple containing the image and its corresponding mask.
        Examples
        --------
        >>> dataset = CustomCarDataset(
        ... img_dir='path/to/images',
        ... mask_dir='path/to/masks',
        ... transform=None
        ... )
        >>> image, mask = dataset[0]
        >>> assert image.shape == (3, 161, 161)
        index.
        The image is loaded as a 3-channel RGB image and the mask is loaded
        as a single-channel grayscale image. The mask is then converted to
        binary format (0 and 1) for segmentation tasks.
        
        """
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(
            self.mask_dir, self.images[idx].replace(".jpg", "_mask.gif")
        )

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)

        mask[mask == 255.0] = 1.0  # Convert mask to binary (0 and 1)
        mask[mask == 0.0] = 0.0

        if self.transform:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask
