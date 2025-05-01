import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Union
import numpy as np

def show_image(image: np.ndarray, label: str = '') -> None:
    """Display a single image using matplotlib.

    Args:
        image (np.ndarray): Image array (RGB or grayscale).
        label (str, optional): Title for the image. Defaults to empty string.
    """
    if len(image.shape) > 2:  # Convert BGR to RGB if necessary
        image = image[..., ::-1]
    plt.imshow(image)
    plt.title(label)
    plt.axis('off')

def show_images(
    images: List[np.ndarray],
    labels: List[str],
    plot_size: Tuple[int, int] = (2, 2),
    figsize: Tuple[int, int] = (16, 16),
    save_path: Optional[str] = None
) -> None:
    """Display a grid of images with corresponding labels.

    Args:
        images (List[np.ndarray]): List of image arrays (RGB or grayscale).
        labels (List[str]): List of titles for each image.
        plot_size (Tuple[int, int], optional): Grid size (rows, columns). Defaults to (2, 2).
        figsize (Tuple[int, int], optional): Figure size (width, height). Defaults to (16, 16).
        save_path (str, optional): Path to save the figure. Defaults to None.

    Raises:
        ValueError: If lengths of images and labels do not match, or plot_size is invalid.
    """
    if len(images) != len(labels):
        raise ValueError("Number of images must match number of labels.")
    if len(images) > plot_size[0] * plot_size[1]:
        raise ValueError("Too many images for the specified plot size.")

    rows, cols = plot_size
    if rows <= 0 or cols <= 0:
        raise ValueError("Plot size must have positive dimensions.")

    fig = plt.figure(figsize=figsize)

    for i, (image, label) in enumerate(zip(images, labels)):
        plt.subplot(rows, cols, i + 1)
        show_image(image, label=label)

    # Hide unused subplots
    for i in range(len(images), rows * cols):
        plt.subplot(rows, cols, i + 1)
        plt.axis('off')

    plt.tight_layout()

    if save_path:
        try:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        except Exception as e:
            raise ValueError(f"Failed to save figure to {save_path}: {e}")

    plt.show()
    plt.close(fig)  # Close figure to free memory