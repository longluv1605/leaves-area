import os
from typing import Tuple, Optional, List, Dict, Any
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
            transform (Optional[A.Compose], optional): Albumentations transform pipeline.
                Defaults to None.

        Raises:
            ValueError: If image_dir or mask_dir is invalid, empty, or if image and mask counts mismatch.
            FileNotFoundError: If no valid image or mask files are found.
        """
        if not os.path.isdir(image_dir) or not os.path.isdir(mask_dir):
            raise ValueError(f"Invalid directory path: image_dir={image_dir}, mask_dir={mask_dir}")

        # Collect image and mask paths
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
            raise FileNotFoundError("No valid images or masks found in the specified directories")
        if len(self.image_paths) != len(self.mask_paths):
            raise ValueError(f"Mismatch between number of images ({len(self.image_paths)}) "
                             f"and masks ({len(self.mask_paths)})")

        # Validate file name consistency (optional, assuming paired files have same base names)
        for img_path, mask_path in zip(self.image_paths, self.mask_paths):
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            mask_name = os.path.splitext(os.path.basename(mask_path))[0]
            if img_name != mask_name.split('_mask')[0]:
                raise ValueError(f"Mismatched image and mask names: {img_name} vs {mask_name}")

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
            Tuple[torch.Tensor, torch.Tensor]: Image tensor of shape (channels, height, width)
                and mask tensor of shape (height, width) with integer class labels.

        Raises:
            IndexError: If idx is out of range.
            FileNotFoundError: If the image or mask file cannot be loaded.
            ValueError: If the image or mask is invalid (e.g., empty or corrupted).
        """
        if not 0 <= idx < len(self.image_paths):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.image_paths)}")

        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # Load image and mask
        image = cv2.imread(image_path)
        if image is None or image.size == 0:
            raise FileNotFoundError(f"Failed to load image or image is empty: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.size == 0:
            raise FileNotFoundError(f"Failed to load mask or mask is empty: {mask_path}")

        # Apply transformations
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']  # Tensor of shape (C, H, W)
            mask = augmented['mask']    # Tensor of shape (H, W)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            mask = torch.from_numpy(mask).long()

        return image, mask.long()


def get_transforms(config: Optional[Dict[str, Any]] = None) -> A.Compose:
    """Create a pipeline of data augmentations for semantic segmentation.

    Args:
        config (Optional[Dict[str, Any]], optional): Configuration dictionary with augmentation
            parameters. If None, uses default settings. Expected keys include:
            - horizontal_flip_p: Probability for horizontal flip.
            - random_rotate_p: Probability for random 90-degree rotation.
            - elastic_transform_p: Probability for elastic transform.
            - grid_distortion_p: Probability for grid distortion.
            - normalize_mean: List of mean values for normalization.
            - normalize_std: List of std values for normalization.
            Defaults to None.

    Returns:
        A.Compose: Albumentations transform pipeline including flips, rotations, distortions,
            normalization, and tensor conversion.

    Raises:
        ValueError: If config contains invalid augmentation parameters.
    """
    # Default augmentation parameters
    default_config = {
        "horizontal_flip_p": 0.5,
        "random_rotate_p": 0.5,
        "elastic_transform_p": 0.3,
        "grid_distortion_p": 0.3,
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
    }

    # Update with provided config
    if config is not None:
        default_config.update(config)

    # Validate parameters
    for prob_key in ["horizontal_flip_p", "random_rotate_p", "elastic_transform_p", "grid_distortion_p"]:
        prob = default_config[prob_key]
        if not isinstance(prob, (int, float)) or not 0 <= prob <= 1:
            raise ValueError(f"{prob_key} must be a number between 0 and 1, got {prob}")
    for key in ["normalize_mean", "normalize_std"]:
        values = default_config[key]
        if not isinstance(values, list) or len(values) != 3 or any(not isinstance(v, (int, float)) for v in values):
            raise ValueError(f"{key} must be a list of three numbers, got {values}")

    return A.Compose([
        A.HorizontalFlip(p=default_config["horizontal_flip_p"]),
        A.RandomRotate90(p=default_config["random_rotate_p"]),
        A.ElasticTransform(p=default_config["elastic_transform_p"]),
        A.GridDistortion(p=default_config["grid_distortion_p"]),
        A.Normalize(
            mean=default_config["normalize_mean"],
            std=default_config["normalize_std"],
        ),
        ToTensorV2(),
    ])