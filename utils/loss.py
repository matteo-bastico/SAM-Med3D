import torch
import torch.nn as nn

from typing import List
from torchvision.ops import sigmoid_focal_loss


class MultipleLoss(nn.Module):
    def __init__(self,
                 loss_fns: List[nn.Module],
                 reduction: str = 'mean',
                 weights: List[float] = None) -> None:
        super().__init__()
        self.loss_fns = loss_fns
        self.reduction = reduction
        # TODO: ensure correct use of reduction and weights
        self.weights = weights

    def forward(self, inputs, targets):
        # Calculate your custom loss here
        losses = []
        for loss_fn, input, target in zip(self.loss_fns, inputs, targets):
            losses.append(loss_fn(input, target))

        if self.reduction == 'mean':
            return sum(losses) / len(losses)
        elif self.reduction == 'sum':
            return sum(losses)
        elif self.reduction == 'none':
            return losses
        elif self.reduction == 'weighted':
            return sum([weight*loss for weight, loss in zip(self.weights, losses)])


class FLWithLogitsLoss(nn.Module):
    def __init__(
            self,
            alpha: float = 0.25,
            gamma: float = 2.0,
            reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        return sigmoid_focal_loss(inputs, targets, self.alpha, self.gamma, self.reduction)
