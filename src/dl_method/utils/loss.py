import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DiceLoss(nn.Module):
    """Dice loss for semantic segmentation, measuring overlap between predictions and targets."""

    def __init__(self, smooth: float = 1e-5, from_logits: bool = True, ignore_index: Optional[int] = None) -> None:
        """Initialize the DiceLoss module.

        Args:
            smooth (float, optional): Smoothing factor to avoid division by zero. Defaults to 1e-5.
            from_logits (bool, optional): If True, applies softmax to inputs. Defaults to True.
            ignore_index (Optional[int], optional): Class index to ignore in loss computation. Defaults to None.

        Raises:
            ValueError: If smooth is negative or ignore_index is invalid.
        """
        super().__init__()
        if smooth < 0:
            raise ValueError("smooth must be non-negative")
        if ignore_index is not None and not isinstance(ignore_index, int):
            raise ValueError("ignore_index must be an integer or None")

        self.smooth = smooth
        self.from_logits = from_logits
        self.ignore_index = ignore_index

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the Dice loss between predicted logits and target masks.

        Args:
            inputs (torch.Tensor): Predicted logits or probabilities of shape
                (batch_size, num_classes, height, width).
            targets (torch.Tensor): Ground truth masks of shape (batch_size, height, width)
                with integer class labels.

        Returns:
            torch.Tensor: Scalar Dice loss value.

        Raises:
            ValueError: If inputs and targets have incompatible shapes or invalid dimensions.
        """
        # Validate input shapes
        if inputs.dim() != 4 or targets.dim() != 3:
            raise ValueError(f"Expected inputs of shape (batch_size, num_classes, height, width) "
                             f"and targets of shape (batch_size, height, width), but got {inputs.shape} "
                             f"and {targets.shape}")
        if inputs.shape[0] != targets.shape[0] or inputs.shape[2:] != targets.shape[1:]:
            raise ValueError("Batch size and spatial dimensions of inputs and targets must match")

        # Apply softmax if inputs are logits
        if self.from_logits:
            inputs = F.softmax(inputs, dim=1)

        num_classes = inputs.shape[1]

        # Create mask for ignore_index
        if self.ignore_index is not None:
            mask = targets != self.ignore_index
        else:
            mask = torch.ones_like(targets, dtype=torch.bool)

        # One-hot encode targets
        targets_onehot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        # Apply mask to inputs and targets
        inputs = inputs * mask.unsqueeze(1)
        targets_onehot = targets_onehot * mask.unsqueeze(1)

        # Flatten spatial dimensions for Dice computation
        inputs_flat = inputs.contiguous().view(inputs.shape[0], num_classes, -1)
        targets_flat = targets_onehot.contiguous().view(targets_onehot.shape[0], num_classes, -1)

        # Compute intersection and union
        intersection = (inputs_flat * targets_flat).sum(-1)
        union = inputs_flat.sum(-1) + targets_flat.sum(-1) + self.smooth

        # Compute Dice score and loss
        dice = (2 * intersection + self.smooth) / union
        loss = 1 - dice.mean()

        return loss


class DiceCELoss(nn.Module):
    """Combined Dice and Cross-Entropy loss for semantic segmentation."""

    def __init__(self, class_weights: Optional[torch.Tensor] = None, smooth: float = 1e-5) -> None:
        """Initialize the DiceCELoss module.

        Args:
            class_weights (Optional[torch.Tensor], optional): Weights for Cross-Entropy loss,
                shape (num_classes,). Defaults to None (equal weights).
            smooth (float, optional): Smoothing factor for Dice loss. Defaults to 1e-5.

        Raises:
            ValueError: If smooth is negative or class_weights has invalid shape.
        """
        super().__init__()
        if smooth < 0:
            raise ValueError("smooth must be non-negative")
        if class_weights is not None and (class_weights.dim() != 1 or any(w < 0 for w in class_weights)):
            raise ValueError("class_weights must be a 1D tensor with non-negative values")

        self.dice = DiceLoss(smooth=smooth, from_logits=True)
        self.ce = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the combined Dice and Cross-Entropy loss.

        Args:
            inputs (torch.Tensor): Predicted logits of shape
                (batch_size, num_classes, height, width).
            targets (torch.Tensor): Ground truth masks of shape (batch_size, height, width)
                with integer class labels.

        Returns:
            torch.Tensor: Scalar combined loss value (0.5 * Dice + 0.5 * CE).

        Raises:
            ValueError: If inputs and targets have incompatible shapes or invalid dimensions.
        """
        # Validate input shapes
        if inputs.dim() != 4 or targets.dim() != 3:
            raise ValueError(f"Expected inputs of shape (batch_size, num_classes, height, width) "
                             f"and targets of shape (batch_size, height, width), but got {inputs.shape} "
                             f"and {targets.shape}")
        if inputs.shape[0] != targets.shape[0] or inputs.shape[2:] != targets.shape[1:]:
            raise ValueError("Batch size and spatial dimensions of inputs and targets must match")

        dice_loss = self.dice(inputs, targets)
        ce_loss = self.ce(inputs, targets)
        return 0.5 * dice_loss + 0.5 * ce_loss