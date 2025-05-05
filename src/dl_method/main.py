import os
from datetime import datetime
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
import platform
import yaml
import time

from models.deeplab import DeepLabV3Plus
from utils.data import SegmentationDataset, get_transforms
from utils.loss import DiceCELoss
from utils.training import train_model, eval_model, evaluate_metrics, EarlyStopping
from utils.visualize import visualize_train_process, visualize_segmentation


def load_config(config_path: str) -> dict:
    """Load configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the config file is invalid.
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML file: {str(e)}")


def setup_data_loaders(
    train_image_dir: str,
    train_mask_dir: str,
    val_image_dir: str,
    val_mask_dir: str,
    batch_size: int = 4,
    num_workers: int = 0,
    transform_config: Optional[dict] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Set up training and validation data loaders.

    Args:
        train_image_dir (str): Path to training images directory.
        train_mask_dir (str): Path to training masks directory.
        val_image_dir (str): Path to validation images directory.
        val_mask_dir (str): Path to validation masks directory.
        batch_size (int, optional): Batch size for data loaders. Defaults to 4.
        num_workers (int, optional): Number of worker threads for data loading.
            Defaults to 0 for compatibility.
        transform_config (Optional[dict], optional): Configuration for data transformations.
            Defaults to None.

    Returns:
        Tuple[DataLoader, DataLoader]: Training and validation data loaders.

    Raises:
        ValueError: If directory paths are invalid or batch_size is non-positive.
        FileNotFoundError: If directories do not exist.
    """
    # Validate inputs
    for path in [train_image_dir, train_mask_dir, val_image_dir, val_mask_dir]:
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Directory does not exist: {path}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")

    # Initialize datasets
    transform = get_transforms(transform_config)
    train_dataset = SegmentationDataset(
        image_dir=train_image_dir,
        mask_dir=train_mask_dir,
        transform=transform,
    )
    val_dataset = SegmentationDataset(
        image_dir=val_image_dir,
        mask_dir=val_mask_dir,
        transform=transform,
    )

    # Initialize data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, val_loader


def train_and_evaluate(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    early_stopping: EarlyStopping=None,
) -> Tuple[List[float], List[float], List[float], List[float], List[float], List[float]]:
    """Train and evaluate the model, collecting metrics for each epoch.

    Args:
        model (nn.Module): The segmentation model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        criterion (nn.Module): Loss function.
        optimizer (optim.Optimizer): Optimizer for training.
        device (torch.device): Device to run computations on.
        num_epochs (int): Number of training epochs.
        early_stopping (EarlyStopping): Early stopping mechanism.

    Returns:
        Tuple[List[float], List[float], List[float], List[float], List[float], List[float]]:
            Lists of training losses, IoUs, accuracies, and validation losses, IoUs, accuracies.

    Raises:
        RuntimeError: If training or evaluation fails (e.g., device issues).
    """
    train_losses, train_ious, train_accs = [], [], []
    val_losses, val_ious, val_accs = [], [], []

    for epoch in range(num_epochs):
        print(f"[Epoch {epoch + 1}/{num_epochs}]")
        try:
            train_loss = train_model(model, train_loader, criterion, optimizer, device)
            val_loss = eval_model(model, val_loader, criterion, device)
            train_iou, train_acc = evaluate_metrics(model, train_loader, num_classes=3, device=device)
            val_iou, val_acc = evaluate_metrics(model, val_loader, num_classes=3, device=device)
        except RuntimeError as e:
            raise RuntimeError(f"Training/evaluation failed at epoch {epoch + 1}: {str(e)}")

        # Store metrics
        train_losses.append(train_loss)
        train_ious.append(train_iou)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_ious.append(val_iou)
        val_accs.append(val_acc)

        print(f"\t-> Train Loss: {train_loss:.4f}, mIoU: {train_iou:.4f}, Acc: {train_acc:.4f}")
        print(f"\t-> Val Loss: {val_loss:.4f}, mIoU: {val_iou:.4f}, Acc: {val_acc:.4f}")
        print('#################################################################')

        ########################################
        new_image_paths = ["images/im1.jpg"]
        visualize_results(
            model=model,
            device=device,
            new_image_paths=new_image_paths
        )
        
        # Check early stopping
        if early_stopping and early_stopping(val_loss, model):
            print("Early stopping triggered")
            break
        
    return train_losses, train_ious, train_accs, val_losses, val_ious, val_accs


def visualize_results(
    model: nn.Module,
    device: torch.device,
    val_dataset: SegmentationDataset,
    new_image_paths: List[str],
    results_dir: str=None,
) -> None:
    """Visualize segmentation results for validation dataset and new images.

    Args:
        model (nn.Module): The trained segmentation model.
        device (torch.device): Device to run computations on.
        val_dataset (SegmentationDataset): Validation dataset for visualization.
        new_image_paths (List[str]): List of paths to new images for segmentation.
        results_dir (str): Directory to save visualization results.

    Raises:
        FileNotFoundError: If image paths or results_dir are invalid.
        RuntimeError: If visualization fails (e.g., model inference issues).
    """
    # Ensure results directory exists
    if results_dir:
        os.makedirs(results_dir, exist_ok=True) 

    # Visualize validation dataset predictions
    for i in range(len(val_dataset)):
        if results_dir:
            save_path = os.path.join(results_dir, f"val_{i}.png")
        else:
            save_path = None
        try:
            visualize_segmentation(
                model=model,
                device=device,
                dataset=val_dataset,
                idx=i,
                save_path=save_path,
            )
        except (FileNotFoundError, RuntimeError) as e:
            raise RuntimeError(f"Failed to visualize validation image {i}: {str(e)}")

    # Visualize new image segmentations
    for i, image_path in enumerate(new_image_paths, 1):
        if results_dir:
            save_path = os.path.join(results_dir, f"new_image_{i}.png")
        else:
            save_path = None
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image does not exist: {image_path}")
        try:
            visualize_segmentation(
                model=model,
                device=device,
                image_path=image_path,
                resize_shape=(640, 640),
                save_path=save_path,
            )
        except (FileNotFoundError, RuntimeError) as e:
            raise RuntimeError(f"Failed to visualize new image {image_path}: {str(e)}")

def inference_time(
    model: nn.Module,
    dataset: SegmentationDataset,
    device: torch.device,
) -> float:
    """Compute model inference time on a dataset.

    Args:
        model (nn.Module): The trained segmentation model.
        dataset (SegmentationDataset): Validation dataset for visualization.
        device (torch.device): Device to run computations on.

    Return:
        float: Average time computing on a dataloader.
    """
    model.to(device)
    model.eval()
    
    start = time.time()
    with torch.inference_mode():
        for image, _ in dataset:
            image = image.to(device)
            pred_mask = model(image.unsqueeze(0))
            pred_mask = torch.argmax(pred_mask.squeeze(), dim=0).cpu().numpy()
    end = time.time()
    total_time = end - start
    avg_time = total_time / len(dataset)
    
    return avg_time
def main() -> None:
    """Main function to train and evaluate a DeepLabV3+ model for semantic segmentation.

    Loads configuration from config.yaml, sets up the model, datasets, and training pipeline,
    trains the model with early stopping, visualizes training metrics, and generates segmentation
    results for validation and new images.

    Raises:
        ValueError: If configuration parameters are invalid.
        FileNotFoundError: If dataset, image paths, or config file are invalid.
        RuntimeError: If training, evaluation, or visualization fails.
    """
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    config = load_config(config_path)

    # Configuration
    data_dir = config.get("data_dir", "datasets/spinach")
    results_dir = config.get("results_dir", "results/deeplabv3plus")
    models_dir = config.get("models_dir", "save/models")
    batch_size = config.get("batch_size", 4)
    num_epochs = config.get("num_epochs", 10)
    learning_rate = config.get("learning_rate", 1e-4)
    num_classes = config.get("num_classes", 3)
    num_workers = 0 if platform.system() == "Windows" else config.get("num_workers", 4)
    early_stopping = config.get("early_stopping", {})
    transform_config = config.get("transformations", {})
    loss_config = config.get("loss", {})
    model_name = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
    model_path = os.path.join(models_dir, model_name)

    # Validate directories
    for dir_path in [data_dir, results_dir, models_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model
    model = DeepLabV3Plus(in_channels=3, num_classes=num_classes).to(device)

    # Setup data loaders
    train_image_dir = os.path.join(data_dir, "images/train")
    train_mask_dir = os.path.join(data_dir, "masks/train")
    val_image_dir = os.path.join(data_dir, "images/val")
    val_mask_dir = os.path.join(data_dir, "masks/val")
    train_loader, val_loader = setup_data_loaders(
        train_image_dir=train_image_dir,
        train_mask_dir=train_mask_dir,
        val_image_dir=val_image_dir,
        val_mask_dir=val_mask_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        transform_config=transform_config,
    )

    # Setup loss, optimizer, and early stopping
    class_weights = torch.tensor(loss_config.get("class_weights", [0.1, 1.0, 5.0]))
    criterion = DiceCELoss(
        class_weights=class_weights.to(device),
        smooth=loss_config.get("smooth", 1e-5),
        dice_ce_weights=tuple(loss_config.get("dice_ce_weights", [0.5, 0.5])),
        ignore_index=loss_config.get("ignore_index", None),
    )
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # early_stopping = EarlyStopping(**early_stopping, mode='min', save_path=model_path)
    early_stopping = None

    # Train and evaluate
    train_losses, train_ious, train_accs, val_losses, val_ious, val_accs = train_and_evaluate(
        model, train_loader, val_loader, criterion, optimizer, device, num_epochs, early_stopping
    )

    # Visualize training process
    visualize_train_process(
        train_losses,
        train_ious,
        train_accs,
        val_losses,
        val_ious,
        val_accs,
        # save_path=os.path.join(results_dir, "training_metrics.png"),
    )

    # Load best model
    if early_stopping is not None and early_stopping.best_model_state is not None:
        model.load_state_dict(early_stopping.best_model_state)
    else:
        torch.save(model.state_dict(), model_path)

    # Visualize results
    new_image_paths = [f"images/im{i}.jpg" for i in range(1, 4)]
    # visualize_results(model, device, val_loader.dataset, new_image_paths, results_dir=None)
    avg_time = inference_time(model, train_loader.dataset, device)
    print(f'Average inference time: {avg_time} seconds.')

if __name__ == "__main__":
    main()