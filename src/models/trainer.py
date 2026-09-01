"""
Model Trainer, Purged Time-Series Cross-Validation, and Signal Policy Inference Engine.

Provides:
1. PurgedTimeSeriesSplit: Cross-validation splitter with label horizon purging and embargoing
2. DNNTrainer: PyTorch training loop with AdamW, CosineAnnealing, Early Stopping, and Checkpointing
3. DNNPredictor: Translates deep neural network predictions into actionable BUY/HOLD/SELL portfolio policies:
   ["action", "buy_prob", "hold_prob", "sell_prob", "target_weight", "expected_return", "confidence"]
   where action in {"BUY", "HOLD", "SELL"} and target_weight in [0.0, 0.35].
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.models.feature_pipeline import (
    DatasetTensors,
    FeaturePipeline,
    MultiHorizonDataset,
)
from src.models.loss_functions import CompositeMultiHorizonLoss, LossWeights
from src.models.multi_horizon_dnn import ModelConfig, MultiHorizonTCT, PredictionOutput

logger = logging.getLogger("Models.Trainer")


class PurgedTimeSeriesSplit:
    """
    Purged Time-Series Split with Embargoing (Marcos López de Prado).
    Prevents information leakage between train and validation sets when labels have forward horizons.
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_bars: int = 12,
        embargo_pct: float = 0.02,
    ) -> None:
        self.n_splits = n_splits
        self.purge_bars = purge_bars
        self.embargo_pct = embargo_pct

    def split(
        self,
        n_samples: int,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate (train_indices, val_indices) for walk-forward purged splits.
        """
        fold_size = n_samples // (self.n_splits + 1)
        embargo_bars = int(n_samples * self.embargo_pct)

        for i in range(1, self.n_splits + 1):
            train_end = i * fold_size
            val_start = train_end + self.purge_bars
            val_end = min(val_start + fold_size, n_samples)

            if val_start >= n_samples or val_end <= val_start:
                break

            train_indices = np.arange(0, train_end)
            val_indices = np.arange(val_start, val_end)

            yield train_indices, val_indices


@dataclass
class TrainerConfig:
    """Hyperparameters for DNNTrainer."""
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    early_stopping_patience: int = 6
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir: str = "models/checkpoints"


class DNNTrainer:
    """
    Institutional PyTorch Training Loop for Multi-Horizon TCT model.
    """

    def __init__(
        self,
        model: Optional[MultiHorizonTCT] = None,
        config: Optional[TrainerConfig] = None,
        model_config: Optional[ModelConfig] = None,
        loss_weights: Optional[LossWeights] = None,
    ) -> None:
        self.config = config or TrainerConfig()
        self.model_config = model_config or ModelConfig()
        self.device = torch.device(self.config.device)
        self.model = (model or MultiHorizonTCT(self.model_config)).to(self.device)
        self.loss_fn = CompositeMultiHorizonLoss(weights=loss_weights).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.config.epochs, eta_min=1e-6
        )

        self.history: List[Dict[str, float]] = []
        self.best_val_loss = float("inf")
        self.best_model_state: Optional[Dict[str, torch.Tensor]] = None

    def fit(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
    ) -> Dict[str, List[float]]:
        """
        Train the model over train_dataset with validation monitoring and early stopping.
        """
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=(len(train_dataset) > self.config.batch_size),
        )

        val_loader = (
            DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
            if val_dataset is not None
            else None
        )

        patience_counter = 0

        for epoch in range(1, self.config.epochs + 1):
            self.model.train()
            train_losses = []
            train_focal_losses = []
            train_huber_losses = []

            for batch in train_loader:
                macro_seq = batch["macro_seq"].to(self.device)
                micro_seq = batch["micro_seq"].to(self.device)
                sent_vec = batch["sent_vec"].to(self.device)
                target_labels = batch["target_labels"].to(self.device)
                target_returns = batch["target_returns"].to(self.device)

                self.optimizer.zero_grad()
                pred: PredictionOutput = self.model(macro_seq, micro_seq, sent_vec)

                loss, breakdown = self.loss_fn(
                    direction_logits=pred.direction_logits,
                    pred_returns=pred.expected_return,
                    target_labels=target_labels,
                    target_returns=target_returns,
                    model_params=list(self.model.parameters()),
                )

                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.config.grad_clip_norm
                )
                self.optimizer.step()

                train_losses.append(breakdown["loss_total"])
                train_focal_losses.append(breakdown["loss_focal"])
                train_huber_losses.append(breakdown["loss_huber"])

            self.scheduler.step()
            mean_train_loss = float(np.mean(train_losses)) if train_losses else 0.0

            # Validation evaluation
            val_metrics = self.evaluate(val_loader) if val_loader else {}
            val_loss = val_metrics.get("val_loss", mean_train_loss)

            epoch_record = {
                "epoch": epoch,
                "train_loss": mean_train_loss,
                "train_focal": float(np.mean(train_focal_losses)) if train_focal_losses else 0.0,
                "train_huber": float(np.mean(train_huber_losses)) if train_huber_losses else 0.0,
                **val_metrics,
            }
            self.history.append(epoch_record)

            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}.")
                    break

        # Load best model state
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        return {k: [h[k] for h in self.history if k in h] for k in self.history[0]}

    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model performance on validation DataLoader."""
        self.model.eval()
        val_losses = []
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                macro_seq = batch["macro_seq"].to(self.device)
                micro_seq = batch["micro_seq"].to(self.device)
                sent_vec = batch["sent_vec"].to(self.device)
                target_labels = batch["target_labels"].to(self.device)
                target_returns = batch["target_returns"].to(self.device)

                pred: PredictionOutput = self.model(macro_seq, micro_seq, sent_vec)
                loss, _ = self.loss_fn(
                    direction_logits=pred.direction_logits,
                    pred_returns=pred.expected_return,
                    target_labels=target_labels,
                    target_returns=target_returns,
                )

                val_losses.append(float(loss.item()))
                preds_class = pred.direction_logits.argmax(dim=-1)
                correct += int((preds_class == target_labels).sum().item())
                total += len(target_labels)

        acc = (correct / total) if total > 0 else 0.0
        return {
            "val_loss": float(np.mean(val_losses)) if val_losses else 0.0,
            "val_accuracy": acc,
        }

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        """Save model weights and optimizer state to disk."""
        import dataclasses

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "model_config_dict": dataclasses.asdict(self.model_config),
                "history": self.history,
            },
            save_path,
        )

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """Load model weights from disk."""
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except (TypeError, ValueError):
            checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint.get("history", [])


class DNNPredictor:
    """
    High-level Inference and Trading Policy Generator.
    Translates Multi-Horizon DNN probabilities into actionable portfolio signals.
    """

    def __init__(
        self,
        model: MultiHorizonTCT,
        feature_pipeline: Optional[FeaturePipeline] = None,
        max_single_weight: float = 0.35,
        buy_threshold: float = 0.50,
        sell_threshold: float = 0.50,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.feature_pipeline = feature_pipeline or FeaturePipeline()
        self.max_single_weight = max_single_weight
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    def predict_tensors(self, tensors: DatasetTensors) -> pd.DataFrame:
        """
        Generate actionable trading policy from pre-built DatasetTensors.
        """
        self.model.eval()
        with torch.no_grad():
            macro_seq = tensors.macro_seq.to(self.device)
            micro_seq = tensors.micro_seq.to(self.device)
            sent_vec = tensors.sent_vec.to(self.device)

            pred: PredictionOutput = self.model(macro_seq, micro_seq, sent_vec)

            probs = pred.direction_probs.cpu().numpy()  # [B, 3] -> [SELL, HOLD, BUY]
            exp_rets = pred.expected_return.cpu().numpy()  # [B, num_horizons]
            convictions = pred.conviction_score.cpu().numpy().flatten()  # [B]

        sell_probs = probs[:, 0]
        hold_probs = probs[:, 1]
        buy_probs = probs[:, 2]

        # Calculate composite expected return (weighted across forecast horizons)
        if exp_rets.shape[1] == 3:
            comp_returns = 0.2 * exp_rets[:, 0] + 0.3 * exp_rets[:, 1] + 0.5 * exp_rets[:, 2]
        else:
            comp_returns = exp_rets[:, 0]

        n = len(probs)
        actions = []
        target_weights = []

        for i in range(n):
            p_b = float(buy_probs[i])
            p_h = float(hold_probs[i])
            p_s = float(sell_probs[i])
            conv = float(convictions[i])

            # Policy logic:
            if p_b > self.buy_threshold and p_b > (p_s + 0.15):
                actions.append("BUY")
                # Conviction multiplier in [0.1, 1.0]
                conviction_mult = max(0.1, (p_b - 0.50) / 0.50) * conv
                w = min(self.max_single_weight, self.max_single_weight * conviction_mult)
                target_weights.append(round(float(w), 4))
            elif p_s > self.sell_threshold:
                actions.append("SELL")
                target_weights.append(0.0)
            else:
                actions.append("HOLD")
                target_weights.append(0.0)

        df_out = pd.DataFrame(
            {
                "action": actions,
                "buy_prob": np.round(buy_probs, 4),
                "hold_prob": np.round(hold_probs, 4),
                "sell_prob": np.round(sell_probs, 4),
                "target_weight": target_weights,
                "expected_return": np.round(comp_returns, 4),
                "confidence": np.round(convictions, 4),
            },
            index=tensors.timestamps,
        )

        return df_out

    def predict_signals(
        self,
        intraday_df: pd.DataFrame,
        daily_df: Optional[pd.DataFrame] = None,
        sentiment_df: Optional[pd.DataFrame] = None,
        spy_daily_df: Optional[pd.DataFrame] = None,
        market_cap: float = 1e10,
    ) -> pd.DataFrame:
        """
        Generate complete policy DataFrame from raw market DataFrames.
        Supports both US and Moroccan BVC equities.
        """
        if daily_df is None or daily_df.empty:
            # Resample daily bars from intraday if not explicitly provided
            daily_df = intraday_df.resample("1D").agg(
                {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
            ).dropna()

        macro_scaled, micro_scaled, sent_scaled = self.feature_pipeline.process_features(
            daily_df=daily_df,
            intraday_df=intraday_df,
            sentiment_df=sentiment_df,
            spy_daily_df=spy_daily_df,
            market_cap=market_cap,
        )

        tensors = self.feature_pipeline.build_dataset_tensors(
            macro_scaled_df=macro_scaled,
            micro_scaled_df=micro_scaled,
            sent_scaled_df=sent_scaled,
        )

        return self.predict_tensors(tensors)
