import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassAccuracy
from tqdm import tqdm
from typing import Tuple, Optional


def train_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train the model for one epoch and compute the average loss.

    Args:
        model (nn.Module): The neural network model to train.
        loader (DataLoader): DataLoader providing training data.
        criterion (nn.Module): Loss function to optimize.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        device (torch.device): Device to run computations on (e.g., 'cuda' or 'cpu').

    Returns:
        float: Average loss over the epoch.

    Raises:
        ValueError: If loader is empty or inputs are invalid.
        RuntimeError: If device placement fails or forward pass encounters errors.
    """
    if not loader:
        raise ValueError("DataLoader cannot be empty")
    if not isinstance(device, torch.device):
        raise ValueError("device must be a torch.device instance")

    model.train()
    epoch_loss = 0.0
    num_batches = len(loader)
    i = 0
    for batch_idx, (images, targets) in enumerate(tqdm(loader, desc="Training")):
        try:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        except RuntimeError as e:
            raise RuntimeError(f"Error in training batch {batch_idx}: {str(e)}")
        
    return epoch_loss / num_batches


def eval_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Evaluate the model on a dataset and compute the average loss.

    Args:
        model (nn.Module): The neural network model to evaluate.
        loader (DataLoader): DataLoader providing evaluation data.
        criterion (nn.Module): Loss function to compute evaluation loss.
        device (torch.device): Device to run computations on (e.g., 'cuda' or 'cpu').

    Returns:
        float: Average loss over the dataset.

    Raises:
        ValueError: If loader is empty or inputs are invalid.
        RuntimeError: If device placement fails or forward pass encounters errors.
    """
    if not loader:
        raise ValueError("DataLoader cannot be empty")
    if not isinstance(device, torch.device):
        raise ValueError("device must be a torch.device instance")

    model.eval()
    epoch_loss = 0.0
    num_batches = len(loader)
    i = 0
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(loader, desc="Evaluating")):
            try:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                epoch_loss += loss.item()
            except RuntimeError as e:
                raise RuntimeError(f"Error in evaluation batch {batch_idx}: {str(e)}")
            
    return epoch_loss / num_batches


def evaluate_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int = 3,
) -> Tuple[float, float]:
    """Compute IoU and accuracy metrics for the model on a dataset.

    Args:
        model (nn.Module): The neural network model to evaluate.
        loader (DataLoader): DataLoader providing evaluation data.
        num_classes (int): Number of classes in the segmentation task.
        device (torch.device): Device to run computations on (e.g., 'cuda' or 'cpu').

    Returns:
        Tuple[float, float]: Mean IoU and weighted accuracy across all classes.

    Raises:
        ValueError: If loader is empty, num_classes is invalid, or inputs are invalid.
        RuntimeError: If device placement fails or metric computation encounters errors.
    """
    if not loader:
        raise ValueError("DataLoader cannot be empty")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    if not isinstance(device, torch.device):
        raise ValueError("device must be a torch.device instance")

    model.eval()
    iou_metric = MulticlassJaccardIndex(num_classes=num_classes).to(device)
    acc_metric = MulticlassAccuracy(num_classes=num_classes, average="weighted").to(device)

    i = 0
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(loader, desc="Evaluating metrics")):
            try:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                predictions = torch.argmax(outputs, dim=1)
                iou_metric.update(predictions, targets)
                acc_metric.update(predictions, targets)
            except RuntimeError as e:
                raise RuntimeError(f"Error in metrics batch {batch_idx}: {str(e)}")
            
    iou = iou_metric.compute().cpu().numpy()
    accuracy = acc_metric.compute().item()
    iou_metric.reset()
    acc_metric.reset()

    return iou, accuracy