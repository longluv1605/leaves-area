import os
import numpy as np
import torch
import cv2
from src.dl_method.models.deeplab import DeepLabV3Plus

def load_model(model_path: str) -> torch.nn.Module:
    """Load a segmentation model from the specified path.

    Args:
        model_path (str): Path to the model .pt or .pth file.

    Returns:
        torch.nn.Module: Loaded PyTorch model.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found")
    model = DeepLabV3Plus(in_channels=3, num_classes=3)
    model.load_state_dict(torch.load(model_path))
    return model


def prepare_image(image_path: str, resize_shape: tuple=(640, 640)) -> torch.Tensor:
    """Load and preprocess an image for model input.

    Args:
        image_path (str): Path to the input image.

    Returns:
        torch.Tensor: Preprocessed image tensor.

    Raises:
        FileNotFoundError: If the image file does not exist.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image {image_path} not found")
    
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if resize_shape is not None:
        image = cv2.resize(image, resize_shape[::-1])  # OpenCV uses (width, height)
    
    image_tensor = torch.from_numpy(image).float().permute(2, 0, 1)
    return image_tensor

def get_mask(model: torch.nn.Module, image: torch.Tensor,
             device: torch.device) -> np.ndarray:
    """Generate a segmentation mask from the model.

    Args:
        model (torch.nn.Module): Loaded segmentation model.
        image (torch.Tensor): Preprocessed input image tensor.
        device (torch.device): Computation device ("cpu" or "cuda").

    Returns:
        np.ndarray: Predicted mask with values 0 (background), 1 (coin), 2 (leaf).
    """
    model.eval()
    image = image.to(device)
    model = model.to(device)
    with torch.inference_mode():
        mask = model(image.unsqueeze(0).to(device))
        mask = torch.argmax(mask.squeeze(), dim=0).cpu().numpy()
    return mask.astype(np.uint8)


def compute_leaves_area(mask: np.ndarray, coin_diameter_mm: float = 21.25) -> float:
    """Compute the real-world leaf area in mm² using a coin as reference.

    Args:
        mask (np.ndarray): Segmentation mask (0: background, 1: coin, 2: leaf).
        coin_diameter_mm (float): Diameter of the reference coin in mm. Defaults to 21.25.

    Returns:
        float: Estimated leaf area in mm².

    Raises:
        ValueError: If no coin is detected in the mask.
    """
    coin_pixel_count = np.sum(mask == 1)
    leaf_pixel_count = np.sum(mask == 2)
    
    if coin_pixel_count == 0:
        raise ValueError("No coin detected in mask. Cannot compute scale.")
    
    coin_area_mm2 = np.pi * (coin_diameter_mm / 2) ** 2
    mm2_per_pixel = coin_area_mm2 / coin_pixel_count
    leaf_area_mm2 = leaf_pixel_count * mm2_per_pixel
    return float(leaf_area_mm2)