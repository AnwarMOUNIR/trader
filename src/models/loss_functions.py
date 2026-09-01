"""
Multi-Horizon Composite Loss Functions for Financial Deep Learning.

Implements:
1. Multi-Class Focal Cross-Entropy Loss with class weighting (addressing severe HOLD class imbalance)
2. Multi-Horizon Huber Loss for continuous return regression across multiple forecast horizons
3. Differential Sharpe Ratio Loss Regularizer for policy optimization and risk-adjusted return maximization
4. Composite Multi-Horizon Loss combining classification, regression, and policy reward
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("Models.LossFunctions")


class FocalLoss(nn.Module):
    """
    Multi-Class Focal Loss with Class-Weighting.
    FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self,
        alpha: Optional[Union[List[float], torch.Tensor]] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
            else:
                self.register_buffer("alpha", alpha.clone().detach().to(torch.float32))
        else:
            # Default weights: SELL=1.5, HOLD=0.5, BUY=1.5
            self.register_buffer("alpha", torch.tensor([1.5, 0.5, 1.5], dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : torch.Tensor
            Classification logits of shape [B, C].
        targets : torch.Tensor
            Target class indices of shape [B] in {0, 1, ..., C-1}.

        Returns
        -------
        torch.Tensor
            Scalar loss or unreduced loss tensor.
        """
        # Compute log-softmax and probabilities stably
        log_p = F.log_softmax(logits, dim=-1)
        p = torch.exp(log_p)

        # Gather target log-probabilities and probabilities
        targets = targets.view(-1, 1)
        target_log_p = log_p.gather(dim=-1, index=targets).squeeze(-1)
        target_p = p.gather(dim=-1, index=targets).squeeze(-1)

        # Modulating factor (1 - p_t)^gamma
        modulating_factor = torch.pow(1.0 - target_p, self.gamma)

        # Class weights alpha_t
        if self.alpha is not None:
            alpha_t = self.alpha.gather(dim=0, index=targets.squeeze(-1))
            focal_loss = -alpha_t * modulating_factor * target_log_p
        else:
            focal_loss = -modulating_factor * target_log_p

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class MultiHorizonHuberLoss(nn.Module):
    """
    Multi-Horizon Huber Loss for Continuous Return Regression.
    L_Huber = sum_{h} w_h * SmoothL1_delta(y_h, y_hat_h)
    """

    def __init__(
        self,
        delta: float = 0.02,
        horizon_weights: Optional[Sequence[float]] = (0.2, 0.3, 0.5),
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.delta = delta
        self.reduction = reduction
        if horizon_weights is not None:
            w_tensor = torch.tensor(horizon_weights, dtype=torch.float32)
            w_tensor = w_tensor / w_tensor.sum()
            self.register_buffer("weights", w_tensor)
        else:
            self.weights = None

    def forward(self, pred_returns: torch.Tensor, target_returns: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pred_returns : torch.Tensor
            Predicted continuous returns of shape [B, num_horizons] or [B, 1].
        target_returns : torch.Tensor
            Target continuous returns of shape [B, num_horizons] or [B, 1].

        Returns
        -------
        torch.Tensor
            Huber regression loss.
        """
        if pred_returns.dim() == 1:
            pred_returns = pred_returns.unsqueeze(-1)
        if target_returns.dim() == 1:
            target_returns = target_returns.unsqueeze(-1)

        # Smooth L1 / Huber loss per element
        huber_per_element = F.smooth_l1_loss(
            pred_returns, target_returns, beta=self.delta, reduction="none"
        )

        if self.weights is not None and huber_per_element.shape[-1] == len(self.weights):
            weighted_huber = huber_per_element * self.weights.view(1, -1)
            loss = weighted_huber.sum(dim=-1)
        else:
            loss = huber_per_element.mean(dim=-1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class DifferentialSharpeLoss(nn.Module):
    """
    Differential Sharpe Ratio Loss Regularizer for Policy Optimization.
    Directly maximizes annualized strategy risk-adjusted return under transaction costs.
    L_Sharpe = - (E[R_strat] / sqrt(Var[R_strat] + eps)) * sqrt(M)
    """

    def __init__(
        self,
        annualization_factor: float = 19656.0,  # 252 * 78 (5m bars)
        friction_cost_bps: float = 5.0,        # 5 bps per unit churn
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.annualization_factor = float(annualization_factor)
        self.friction_cost = friction_cost_bps / 10000.0
        self.eps = eps

    def forward(
        self,
        action_probs: torch.Tensor,
        realized_returns: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        action_probs : torch.Tensor
            Softmax probabilities of shape [B, 3] (SELL=0, HOLD=1, BUY=2) or logits.
        realized_returns : torch.Tensor
            Realized continuous returns of shape [B] or [B, 1].

        Returns
        -------
        torch.Tensor
            Negative Sharpe Ratio loss (lower is better).
        """
        if action_probs.shape[-1] == 3 and not torch.allclose(action_probs.sum(dim=-1), torch.ones(action_probs.shape[0], device=action_probs.device), atol=1e-2):
            probs = F.softmax(action_probs, dim=-1)
        else:
            probs = action_probs

        r = realized_returns.view(-1)
        # Position policy: a_t in [-1.0, 1.0] (BUY prob - SELL prob)
        p_sell = probs[:, 0]
        p_buy = probs[:, 2]
        a = p_buy - p_sell  # [B]

        # Calculate transaction costs from position changes across the batch
        if len(a) > 1:
            delta_a = torch.cat([torch.zeros(1, device=a.device), torch.abs(a[1:] - a[:-1])])
        else:
            delta_a = torch.zeros_like(a)

        # Strategy return
        strat_returns = a * r - self.friction_cost * delta_a

        # Batch Mean and Variance of Strategy Return
        mean_ret = torch.mean(strat_returns)
        var_ret = torch.var(strat_returns, unbiased=False)
        std_ret = torch.sqrt(var_ret + self.eps)

        # Annualized Sharpe Ratio
        annual_factor = torch.sqrt(torch.tensor(self.annualization_factor, device=a.device, dtype=torch.float32))
        sharpe = (mean_ret / std_ret) * annual_factor

        # Bounded loss to prevent exploding gradient: -tanh(Sharpe / 5.0) * 5.0
        loss = -torch.tanh(sharpe / 5.0) * 5.0
        return loss


@dataclass
class LossWeights:
    """Weights for composite multi-horizon loss."""
    lambda_focal: float = 1.0
    lambda_huber: float = 0.5
    lambda_sharpe: float = 0.8
    lambda_l2: float = 1e-5


class CompositeMultiHorizonLoss(nn.Module):
    """
    Total Multi-Horizon Loss combining:
    - Focal Loss (Directional Classification)
    - Huber Loss (Expected Return Regression)
    - Differential Sharpe Ratio Regularizer (Policy Maximization)
    - L2 Weight Regularization
    """

    def __init__(
        self,
        weights: Optional[LossWeights] = None,
        focal_alpha: Optional[List[float]] = None,
        focal_gamma: float = 2.0,
        huber_delta: float = 0.02,
        horizon_weights: Tuple[float, ...] = (0.2, 0.3, 0.5),
        annualization_factor: float = 19656.0,
    ) -> None:
        super().__init__()
        self.weights = weights or LossWeights()
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.huber_loss = MultiHorizonHuberLoss(delta=huber_delta, horizon_weights=horizon_weights)
        self.sharpe_loss = DifferentialSharpeLoss(annualization_factor=annualization_factor)

    def forward(
        self,
        direction_logits: torch.Tensor,
        pred_returns: torch.Tensor,
        target_labels: torch.Tensor,
        target_returns: torch.Tensor,
        model_params: Optional[Sequence[torch.nn.Parameter]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute composite multi-horizon loss.

        Parameters
        ----------
        direction_logits : torch.Tensor
            Predicted class logits [B, 3].
        pred_returns : torch.Tensor
            Predicted returns [B, num_horizons].
        target_labels : torch.Tensor
            Target class indices [B].
        target_returns : torch.Tensor
            Target continuous returns [B, num_horizons].
        model_params : Sequence[torch.nn.Parameter], optional
            Model parameters for L2 regularization.

        Returns
        -------
        Tuple[torch.Tensor, Dict[str, float]]
            (total_loss, loss_breakdown_dict)
        """
        loss_focal = self.focal_loss(direction_logits, target_labels)
        loss_huber = self.huber_loss(pred_returns, target_returns)
        
        # Immediate 1st horizon return for Sharpe policy regularizer
        first_h_return = target_returns[:, 0] if target_returns.dim() > 1 else target_returns
        probs = F.softmax(direction_logits, dim=-1)
        loss_sharpe = self.sharpe_loss(probs, first_h_return)

        # L2 Regularization
        if model_params is not None and self.weights.lambda_l2 > 0:
            l2_reg = sum(torch.sum(p ** 2) for p in model_params if p.requires_grad)
        else:
            l2_reg = torch.tensor(0.0, device=direction_logits.device)

        total_loss = (
            self.weights.lambda_focal * loss_focal
            + self.weights.lambda_huber * loss_huber
            + self.weights.lambda_sharpe * loss_sharpe
            + self.weights.lambda_l2 * l2_reg
        )

        breakdown = {
            "loss_total": float(total_loss.detach().item()),
            "loss_focal": float(loss_focal.detach().item()),
            "loss_huber": float(loss_huber.detach().item()),
            "loss_sharpe": float(loss_sharpe.detach().item()),
            "loss_l2": float(l2_reg.detach().item()),
        }

        return total_loss, breakdown
