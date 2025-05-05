import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class DiceLoss(nn.Module):
    """Dice loss for semantic segmentation, measuring overlap between predictions and targets."""

    def __init__(
        self,
        smooth: float = 1e-5,
        from_logits: bool = True,
        ignore_index: Optional[int] = None,
    ) -> None:
        """Initialize the DiceLoss module.

        Args:
            smooth (float, optional): Smoothing factor to avoid division by zero. Defaults to 1e-5.
            from_logits (bool, optional): If True, applies softmax to inputs. Defaults to True.
            ignore_index (Optional[int], optional): Class index to ignore in loss computation.
                Defaults to None.

        Raises:
            ValueError: If smooth is negative or ignore_index is invalid.
        """
        super().__init__()
        if smooth < 0:
            raise ValueError(f"smooth must be non-negative, got {smooth}")
        if ignore_index is not None and not isinstance(ignore_index, int):
            raise ValueError(
                f"ignore_index must be an integer or None, got {ignore_index}"
            )

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
            RuntimeError: If numerical instability occurs during computation.
        """
        # Validate input shapes
        if inputs.dim() != 4 or targets.dim() != 3:
            raise ValueError(
                f"Expected inputs of shape (batch_size, num_classes, height, width) "
                f"and targets of shape (batch_size, height, width), but got {inputs.shape} "
                f"and {targets.shape}"
            )
        if inputs.shape[0] != targets.shape[0] or inputs.shape[2:] != targets.shape[1:]:
            raise ValueError(
                f"Batch size and spatial dimensions must match: "
                f"inputs {inputs.shape}, targets {targets.shape}"
            )

        # Apply softmax if inputs are logits
        if self.from_logits:
            inputs = F.softmax(inputs, dim=1)

        num_classes = inputs.shape[1]

        # Create mask for ignore_index
        mask = torch.ones_like(targets, dtype=torch.bool)
        if self.ignore_index is not None:
            mask = targets != self.ignore_index

        # One-hot encode targets
        targets_onehot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        # Apply mask to inputs and targets
        inputs = inputs * mask.unsqueeze(1)
        targets_onehot = targets_onehot * mask.unsqueeze(1)

        # Flatten spatial dimensions
        inputs_flat = inputs.contiguous().view(inputs.shape[0], num_classes, -1)
        targets_flat = targets_onehot.contiguous().view(
            targets_onehot.shape[0], num_classes, -1
        )

        # Compute intersection and union
        intersection = (inputs_flat * targets_flat).sum(-1)
        union = inputs_flat.sum(-1) + targets_flat.sum(-1) + self.smooth

        # Check for numerical stability
        if torch.any(union <= self.smooth):
            raise RuntimeError(
                "Union is too small, causing numerical instability in Dice loss"
            )

        # Compute Dice score and loss
        dice = (2 * intersection + self.smooth) / union
        loss = 1 - dice.mean()

        return loss


class DiceCELoss(nn.Module):
    """Combined Dice and Cross-Entropy loss for semantic segmentation."""

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        smooth: float = 1e-5,
        dice_ce_weights: Tuple[float, float] = (0.5, 0.5),
        ignore_index: Optional[int] = None,
    ) -> None:
        """Initialize the DiceCELoss module.

        Args:
            class_weights (Optional[torch.Tensor], optional): Weights for Cross-Entropy loss,
                shape (num_classes,). Defaults to None (equal weights).
            smooth (float, optional): Smoothing factor for Dice loss. Defaults to 1e-5.
            dice_ce_weights (Tuple[float, float], optional): Weights for Dice and CE losses,
                respectively. Defaults to (0.5, 0.5).
            ignore_index (Optional[int], optional): Class index to ignore in loss computation.
                Defaults to None.

        Raises:
            ValueError: If smooth is negative, class_weights or dice_ce_weights are invalid,
                or dice_ce_weights do not sum to a positive value.
        """
        super().__init__()
        if smooth < 0:
            raise ValueError(f"smooth must be non-negative, got {smooth}")
        if class_weights is not None and (
            class_weights.dim() != 1 or any(w < 0 for w in class_weights)
        ):
            raise ValueError(
                f"class_weights must be a 1D tensor with non-negative values, got {class_weights}"
            )
        if not isinstance(dice_ce_weights, (tuple, list)) or len(dice_ce_weights) != 2:
            raise ValueError(
                f"dice_ce_weights must be a tuple of two floats, got {dice_ce_weights}"
            )
        if any(w < 0 for w in dice_ce_weights):
            raise ValueError(
                f"dice_ce_weights must be non-negative, got {dice_ce_weights}"
            )
        if sum(dice_ce_weights) <= 0:
            raise ValueError(
                f"Sum of dice_ce_weights must be positive, got {sum(dice_ce_weights)}"
            )

        self.dice = DiceLoss(smooth=smooth, from_logits=True, ignore_index=ignore_index)
        self.ce = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=ignore_index if ignore_index is not None else -100,
        )
        self.dice_weight, self.ce_weight = dice_ce_weights

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the combined Dice and Cross-Entropy loss.

        Args:
            inputs (torch.Tensor): Predicted logits of shape
                (batch_size, num_classes, height, width).
            targets (torch.Tensor): Ground truth masks of shape (batch_size, height, width)
                with integer class labels.

        Returns:
            torch.Tensor: Scalar combined loss value (dice_weight * Dice + ce_weight * CE).

        Raises:
            ValueError: If inputs and targets have incompatible shapes or invalid dimensions.
            RuntimeError: If numerical instability occurs during computation.
        """
        # Validate input shapes
        if inputs.dim() != 4 or targets.dim() != 3:
            raise ValueError(
                f"Expected inputs of shape (batch_size, num_classes, height, width) "
                f"and targets of shape (batch_size, height, width), but got {inputs.shape} "
                f"and {targets.shape}"
            )
        if inputs.shape[0] != targets.shape[0] or inputs.shape[2:] != targets.shape[1:]:
            raise ValueError(
                f"Batch size and spatial dimensions must match: "
                f"inputs {inputs.shape}, targets {targets.shape}"
            )

        dice_loss = self.dice(inputs, targets)
        ce_loss = self.ce(inputs, targets)
        return self.dice_weight * dice_loss + self.ce_weight * ce_loss
