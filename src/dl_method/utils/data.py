import os
from typing import Tuple, Optional, List
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class SegmentationDataset(Dataset):
    """Custom dataset for semantic segmentation, loading images and masks from directories."""

    def __init__(self, image_dir: str, mask_dir: str, transform: Optional[A.Compose] = None) -> None:
        """Initialize the dataset with image and mask directories.

        Args:
            image_dir (str): Path to the directory containing input images.
            mask_dir (str): Path to the directory containing segmentation masks.
            transform (Optional[A.Compose], optional): Albumentations transform pipeline. Defaults to None.

        Raises:
            ValueError: If image_dir or mask_dir is invalid or empty, or if the number of images and masks mismatch.
        """
        if not os.path.isdir(image_dir) or not os.path.isdir(mask_dir):
            raise ValueError("Invalid directory path for images or masks")

        # Collect image and mask paths, filtering for common image extensions
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        self.image_paths: List[str] = sorted([
            os.path.join(image_dir, f) for f in os.listdir(image_dir)
            if os.path.splitext(f)[1].lower() in valid_extensions
        ])
        self.mask_paths: List[str] = sorted([
            os.path.join(mask_dir, f) for f in os.listdir(mask_dir)
            if os.path.splitext(f)[1].lower() in valid_extensions
        ])

        if not self.image_paths or not self.mask_paths:
            raise ValueError("No valid images or masks found in the specified directories")
        if len(self.image_paths) != len(self.mask_paths):
            raise ValueError(f"Mismatch between number of images ({len(self.image_paths)}) "
                             f"and masks ({len(self.mask_paths)})")

        self.transform = transform

    def __len__(self) -> int:
        """Return the total number of samples in the dataset.

        Returns:
            int: Number of image-mask pairs.
        """
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve the image and mask at the specified index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Image tensor of shape (channels, height, width) and
                mask tensor of shape (height, width) with integer class labels.

        Raises:
            FileNotFoundError: If the image or mask file cannot be loaded.
            ValueError: If the image or mask is invalid (e.g., empty or corrupted).
        """
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # Load image and mask
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Failed to load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Failed to load mask: {mask_path}")

        # Apply transformations if provided
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']  # Already converted to tensor
            mask = augmented['mask']    # Already converted to tensor
        else:
            # Convert to tensors manually if no transform is applied
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            mask = torch.from_numpy(mask).long()

        return image, mask.long()


def get_transforms() -> A.Compose:
    """Create a pipeline of data augmentations for semantic segmentation.

    Returns:
        A.Compose: Albumentations transform pipeline including flips, rotations, distortions,
            normalization, and tensor conversion.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ElasticTransform(p=0.3),
        A.GridDistortion(p=0.3),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2()
    ])