import os
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
    for batch_idx, (images, targets) in enumerate(tqdm(loader, desc="Training", unit='batch')):
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
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(loader, desc="Evaluating", unit='batch')):
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

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(loader, desc="Evaluating metrics", unit='batch')):
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


class EarlyStopping:
    """Early stopping mechanism to halt training when a metric stops improving."""

    def __init__(
        self,
        patience: int = 5,
        delta: float = 0.0,
        mode: str = 'min',
        save_path: Optional[str] = None,
    ) -> None:
        """Initialize the EarlyStopping module.

        Args:
            patience (int, optional): Number of epochs to wait for improvement before stopping.
                Defaults to 5.
            delta (float, optional): Minimum change in the monitored metric to qualify as an
                improvement. Defaults to 0.0.
            mode (str, optional): One of 'min' or 'max'. If 'min', lower metric values are better
                (e.g., loss). If 'max', higher values are better (e.g., IoU). Defaults to 'min'.
            save_path (Optional[str], optional): Path to save the best model state (e.g., 'best_model.pth').
                If None, no model is saved. Defaults to None.

        Raises:
            ValueError: If patience or delta is negative, mode is invalid, or save_path is invalid.
        """
        if patience < 0:
            raise ValueError("patience must be non-negative")
        if delta < 0:
            raise ValueError("delta must be non-negative")
        if mode not in ['min', 'max']:
            raise ValueError("mode must be 'min' or 'max'")
        if save_path is not None:
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.isdir(save_dir):
                raise ValueError(f"Directory for save_path does not exist: {save_dir}")

        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.save_path = save_path
        self.best_score: Optional[float] = None
        self.counter: int = 0
        self.early_stop: bool = False
        self.best_model_state: Optional[dict] = None

    def __call__(self, metric: float, model: nn.Module) -> bool:
        """Check if training should stop based on the current metric.

        Updates the best score and model state if the metric improves. Increments the counter
        if no improvement is observed. Sets early_stop to True if patience is exceeded.

        Args:
            metric (float): Current value of the monitored metric (e.g., validation loss or IoU).
            model (nn.Module): The model whose state may be saved if the metric improves.

        Returns:
            bool: True if training should stop, False otherwise.

        Raises:
            ValueError: If metric is not a finite number or model is invalid.
        """
        if not isinstance(metric, (int, float)) or not torch.isfinite(torch.tensor(metric)):
            raise ValueError("metric must be a finite number")
        if not isinstance(model, nn.Module):
            raise ValueError("model must be a torch.nn.Module instance")

        score = -metric if self.mode == 'min' else metric

        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
            if self.save_path:
                self._save_model(model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0
            if self.save_path:
                self._save_model(model)

        return self.early_stop

    def _save_model(self, model: nn.Module) -> None:
        """Save the current model state to the specified save_path.

        Args:
            model (nn.Module): The model to save.

        Raises:
            OSError: If saving the model to save_path fails (e.g., permissions error).
        """
        try:
            torch.save(model.state_dict(), self.save_path)
        except OSError as e:
            raise OSError(f"Failed to save model to {self.save_path}: {str(e)}")