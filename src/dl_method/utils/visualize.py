import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import List, Optional, Union, Tuple


def visualize_train_process(
    train_losses: List[float],
    train_ious: List[Union[float, np.ndarray]],
    train_accs: List[float],
    val_losses: List[float],
    val_ious: List[Union[float, np.ndarray]],
    val_accs: List[float],
    save_path: Optional[str] = None,
) -> None:
    """Plot training and validation loss, mean IoU, and accuracy over epochs.

    Generates three subplots for loss, mean IoU, and accuracy, comparing training and validation
    metrics. Optionally saves the plot to a file or displays it interactively.

    Args:
        train_losses (List[float]): List of training losses per epoch.
        train_ious (List[Union[float, np.ndarray]]): List of training IoU scores per epoch.
            Each IoU can be a scalar or a NumPy array of per-class IoUs.
        train_accs (List[float]): List of training accuracies per epoch.
        val_losses (List[float]): List of validation losses per epoch.
        val_ious (List[Union[float, np.ndarray]]): List of validation IoU scores per epoch.
            Each IoU can be a scalar or a NumPy array of per-class IoUs.
        val_accs (List[float]): List of validation accuracies per epoch.
        save_path (Optional[str], optional): Path to save the plot (e.g., 'plot.png').
            If None, the plot is displayed interactively. Defaults to None.

    Raises:
        ValueError: If input lists are empty or have inconsistent lengths.
        OSError: If saving the plot to save_path fails (e.g., invalid path or permissions).
    """
    # Validate inputs
    inputs = [train_losses, train_ious, train_accs, val_losses, val_ious, val_accs]
    if any(not lst for lst in inputs):
        raise ValueError("Input lists cannot be empty")
    if len(set(len(lst) for lst in inputs)) != 1:
        raise ValueError("All input lists must have the same length")

    epochs = list(range(1, len(train_losses) + 1))

    # Compute mean IoU per epoch (handle both scalar and array inputs)
    train_iou_means = [
        float(iou.mean()) if isinstance(iou, np.ndarray) else float(iou)
        for iou in train_ious
    ]
    val_iou_means = [
        float(iou.mean()) if isinstance(iou, np.ndarray) else float(iou)
        for iou in val_ious
    ]

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss plot
    axes[0].plot(epochs, train_losses, "-o", label="Train Loss")
    axes[0].plot(epochs, val_losses, "-o", label="Val Loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    # mIoU plot
    axes[1].plot(epochs, train_iou_means, "-o", label="Train mIoU")
    axes[1].plot(epochs, val_iou_means, "-o", label="Val mIoU")
    axes[1].set_title("Mean IoU")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("mIoU")
    axes[1].legend()
    axes[1].grid(True)

    # Accuracy plot
    axes[2].plot(epochs, train_accs, "-o", label="Train Acc")
    axes[2].plot(epochs, val_accs, "-o", label="Val Acc")
    axes[2].set_title("Accuracy")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Accuracy")
    axes[2].legend()
    axes[2].grid(True)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save or display plot
    if save_path is not None:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        except OSError as e:
            raise OSError(f"Failed to save plot to {save_path}: {str(e)}")
    else:
        plt.show()

    # Close figure to prevent memory leaks
    plt.close(fig)


def show_images(
    image: np.ndarray,
    pred_mask: np.ndarray,
    true_mask: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
) -> None:
    """Display or save an image, predicted mask, and optional ground truth mask.

    Args:
        image (np.ndarray): RGB image array of shape (height, width, 3).
        pred_mask (np.ndarray): Predicted mask array of shape (height, width).
        true_mask (Optional[np.ndarray], optional): Ground truth mask array of shape
            (height, width). If None, only image and predicted mask are shown. Defaults to None.
        save_path (Optional[str], optional): Path to save the plot (e.g., 'plot.png').
            If None, the plot is displayed interactively. Defaults to None.

    Raises:
        ValueError: If image or masks have invalid shapes or types.
        OSError: If saving the plot to save_path fails (e.g., invalid path or permissions).
    """
    # Validate inputs
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be an RGB array of shape (height, width, 3)")
    if not isinstance(pred_mask, np.ndarray) or pred_mask.ndim != 2:
        raise ValueError("pred_mask must be a 2D array of shape (height, width)")
    if true_mask is not None and (
        not isinstance(true_mask, np.ndarray) or true_mask.ndim != 2
    ):
        raise ValueError("true_mask must be a 2D array of shape (height, width)")
    if true_mask is not None and true_mask.shape != pred_mask.shape:
        raise ValueError("true_mask and pred_mask must have the same shape")
    if image.shape[:2] != pred_mask.shape:
        raise ValueError("image and pred_mask must have matching height and width")

    # Set up subplots
    num_plots = 3 if true_mask is not None else 2
    fig, axes = plt.subplots(1, num_plots, figsize=(4 * num_plots, 4))

    # Plot image
    axes[0].imshow(image)
    axes[0].set_title("Image")

    # Plot true mask if provided
    if true_mask is not None:
        axes[1].imshow(true_mask, cmap="nipy_spectral")
        axes[1].set_title("Ground Truth")
        pred_ax = axes[2]
    else:
        pred_ax = axes[1]

    # Plot predicted mask
    pred_ax.imshow(pred_mask, cmap="nipy_spectral")
    pred_ax.set_title("Prediction")

    # Remove axes
    for ax in axes:
        ax.axis("off")

    # Adjust layout
    plt.tight_layout()

    # Save or display plot
    if save_path is not None:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        except OSError as e:
            raise OSError(f"Failed to save plot to {save_path}: {str(e)}")
    else:
        plt.show()

    # Close figure to prevent memory leaks
    plt.close(fig)


def visualize_segmentation(
    model: nn.Module,
    device: torch.device,
    dataset: Optional[Dataset] = None,
    idx: int = 0,
    image_path: Optional[str] = None,
    resize_shape: Optional[Tuple[int, int]] = (640, 640),
    save_path: Optional[str] = None,
) -> None:
    """Visualize segmentation predictions for a dataset image or a new image.

    If a dataset and index are provided, displays the image, ground truth mask, and predicted mask.
    If an image path is provided, displays the image and predicted mask. The image is optionally
    resized to a specified shape before prediction.

    Args:
        model (nn.Module): The segmentation model (e.g., DeepLabV3Plus).
        device (torch.device): Device to run the model on (e.g., 'cuda' or 'cpu').
        dataset (Optional[Dataset], optional): Dataset containing images and masks (e.g.,
            SegmentationDataset). Required if image_path is None. Defaults to None.
        idx (int, optional): Index of the image in the dataset to visualize. Defaults to 0.
        image_path (Optional[str], optional): Path to a new image for segmentation. If provided,
            dataset and idx are ignored. Defaults to None.
        resize_shape (Optional[Tuple[int, int]], optional): Shape (height, width) to resize the
            image to before prediction. If None, no resizing is applied. Defaults to (640, 640).
        save_path (Optional[str], optional): Path to save the plot (e.g., 'plot.png').
            If None, the plot is displayed interactively. Defaults to None.

    Raises:
        ValueError: If inputs are invalid (e.g., missing dataset/image_path, invalid idx, or shapes).
        FileNotFoundError: If image_path or dataset paths are invalid.
        RuntimeError: If model inference fails (e.g., device issues).
    """
    # Validate inputs
    if dataset is None and image_path is None:
        raise ValueError("Either dataset or image_path must be provided")
    if dataset is not None and image_path is not None:
        raise ValueError("Cannot provide both dataset and image_path")
    if not isinstance(model, nn.Module):
        raise ValueError("model must be a torch.nn.Module instance")
    if not isinstance(device, torch.device):
        raise ValueError("device must be a torch.device instance")
    if resize_shape is not None and (
        not isinstance(resize_shape, tuple)
        or len(resize_shape) != 2
        or any(s <= 0 for s in resize_shape)
    ):
        raise ValueError(
            "resize_shape must be a tuple of two positive integers (height, width)"
        )

    model.eval()
    model.to(device)

    if dataset is not None:
        # Validate dataset and index
        if not hasattr(dataset, "image_paths") or not hasattr(dataset, "mask_paths"):
            raise ValueError("dataset must have image_paths and mask_paths attributes")
        if not (0 <= idx < len(dataset.image_paths)):
            raise ValueError(
                f"idx must be between 0 and {len(dataset.image_paths) - 1}"
            )

        # Load image and true mask
        image = cv2.imread(dataset.image_paths[idx])
        if image is None:
            raise FileNotFoundError(f"Failed to load image: {dataset.image_paths[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        true_mask = cv2.imread(dataset.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        if true_mask is None:
            raise FileNotFoundError(f"Failed to load mask: {dataset.mask_paths[idx]}")
    else:
        # Load and resize image
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Failed to load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if resize_shape is not None:
            image = cv2.resize(image, resize_shape[::-1])  # OpenCV uses (width, height)
        true_mask = None

    # Prepare input tensor
    image_tensor = torch.from_numpy(image).float().permute(2, 0, 1)

    # Run inference
    with torch.no_grad():
        try:
            pred_mask = model(image_tensor.unsqueeze(0).to(device))
            pred_mask = torch.argmax(pred_mask.squeeze(), dim=0).cpu().numpy()
        except RuntimeError as e:
            raise RuntimeError(f"Model inference failed: {str(e)}")

    # Display or save images
    show_images(image, pred_mask, true_mask, save_path)
