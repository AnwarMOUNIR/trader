"""
Dual-Branch Causal Dilated Temporal Convolutional Transformer (TCT).

PyTorch Multi-Horizon Deep Neural Network integrating:
1. Branch 1: Dilated Causal TCN processing 30-day/60-day secular macro daily sequence
2. Branch 2: Dilated Causal TCN processing high-frequency intraday microstructure sequence
3. Multi-Head Cross-Attention fusing macro market regime with microstructure dynamics
4. Gated Residual Network (GRN) integrating real-time NLP sentiment vectors
5. Multi-Task Output Heads: Directional Logits (BUY/HOLD/SELL), Multi-Horizon Returns, Uncertainty/Volatility
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("Models.MultiHorizonDNN")


@dataclass
class ModelConfig:
    """Configuration hyperparameters for MultiHorizonTCT model."""
    macro_input_dim: int = 8          # Number of daily macro features
    micro_input_dim: int = 12         # Number of intraday microstructure features
    sent_input_dim: int = 7           # Number of continuous NLP sentiment features
    macro_seq_len: int = 60           # Lookback window for macro daily sequence
    micro_seq_len: int = 39           # Lookback window for intraday microstructure sequence
    hidden_dim: int = 64              # Latent representation dimension
    num_heads: int = 4                # Number of cross-attention heads
    num_horizons: int = 3             # Number of forecast return horizons (e.g. h3, h12, h78)
    num_classes: int = 3              # Number of directional classes (SELL=0, HOLD=1, BUY=2)
    macro_dilations: Tuple[int, ...] = (1, 2, 4, 8)
    micro_dilations: Tuple[int, ...] = (1, 2, 4)
    kernel_size: int = 3
    dropout: float = 0.20
    learning_rate: float = 3e-4


@dataclass
class PredictionOutput:
    """Structured container for MultiHorizonTCT inference outputs."""
    direction_logits: torch.Tensor       # Shape: [B, 3] (SELL=0, HOLD=1, BUY=2)
    direction_probs: torch.Tensor        # Shape: [B, 3] (Softmax probabilities)
    expected_return: torch.Tensor        # Shape: [B, num_horizons] (Predicted forward returns)
    predicted_volatility: torch.Tensor   # Shape: [B, 1] (Predicted forward sigma >= 0)
    conviction_score: torch.Tensor       # Shape: [B, 1] (Confidence score in [0.0, 1.0])


class CausalConv1d(nn.Module):
    """
    1D Causal Convolution ensuring zero lookahead bias.
    Pads strictly on the left so output at time t depends only on inputs <= t.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=self.padding,
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: [B, C, T]
        Output: [B, C_out, T] (Strictly causal, sliced to original sequence length)
        """
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class TemporalBlock(nn.Module):
    """
    Residual Temporal Block with Causal Convolutions, LayerNorm, GELU, and Dropout.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.norm1 = nn.LayerNorm(out_channels)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.norm2 = nn.LayerNorm(out_channels)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

        # Residual shortcut
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: [B, C_in, T]
        Output: [B, C_out, T]
        """
        residual = self.shortcut(x)

        # First conv block
        out = self.conv1(x)
        out = out.transpose(1, 2)  # [B, T, C] for LayerNorm
        out = self.norm1(out)
        out = out.transpose(1, 2)  # [B, C, T]
        out = self.act1(out)
        out = self.drop1(out)

        # Second conv block
        out = self.conv2(out)
        out = out.transpose(1, 2)
        out = self.norm2(out)
        out = out.transpose(1, 2)
        out = self.act2(out)
        out = self.drop2(out)

        return F.gelu(out + residual)


class CausalTCNBranch(nn.Module):
    """
    Multi-Layer Dilated Causal TCN Branch for sequence feature extraction.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        dilations: Sequence[int] = (1, 2, 4, 8),
        kernel_size: int = 3,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        layers = []
        cur_dim = in_dim
        for d in dilations:
            layers.append(
                TemporalBlock(
                    in_channels=cur_dim,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=d,
                    dropout=dropout,
                )
            )
            cur_dim = hidden_dim
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: [B, T, D]
        Output: [B, T, hidden_dim]
        """
        # Transpose to [B, D, T] for 1D convolutions
        x_trans = x.transpose(1, 2)
        out = self.network(x_trans)
        # Transpose back to [B, T, hidden_dim]
        return out.transpose(1, 2)


class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN) inspired by Temporal Fusion Transformers (TFT).
    Provides adaptive non-linear processing with gating and residual skip-connections.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: Optional[int] = None,
        context_dim: Optional[int] = None,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim or input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.context_fc = nn.Linear(context_dim, hidden_dim, bias=False) if context_dim else None
        self.fc2 = nn.Linear(hidden_dim, self.output_dim * 2)  # For Gated Linear Unit
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(self.output_dim)

        if input_dim != self.output_dim:
            self.skip_proj = nn.Linear(input_dim, self.output_dim)
        else:
            self.skip_proj = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Input: x [B, ..., input_dim], context [B, ..., context_dim]
        Output: [B, ..., output_dim]
        """
        h = self.fc1(x)
        if context is not None and self.context_fc is not None:
            h = h + self.context_fc(context)
        h = F.elu(h)
        h = self.fc2(self.dropout(h))

        # GLU gating: split into value and gate
        val, gate = torch.chunk(h, 2, dim=-1)
        gated = val * torch.sigmoid(gate)

        skip = self.skip_proj(x)
        return self.layer_norm(skip + gated)


class MultiHorizonTCT(nn.Module):
    """
    Dual-Branch Causal Temporal Convolutional Transformer with Sentiment GRN.

    Architecture:
    1. Macro TCN Branch -> Process 60-day macro history
    2. Micro TCN Branch -> Process 39-bar intraday microstructure
    3. Multi-Head Cross-Attention -> Intraday queries attend to Macro regime context
    4. Sentiment Gated Residual Network -> Fuses attention output with real-time news sentiment
    5. Multi-Task Output Heads:
       - Direction Classification (SELL, HOLD, BUY)
       - Continuous Expected Multi-Horizon Returns (h in {3, 12, 78})
       - Volatility / Downside Risk & Conviction Estimation
    """

    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()

        # Branch 1: Macro Sequence TCN
        self.macro_branch = CausalTCNBranch(
            in_dim=self.config.macro_input_dim,
            hidden_dim=self.config.hidden_dim,
            dilations=self.config.macro_dilations,
            kernel_size=self.config.kernel_size,
            dropout=self.config.dropout,
        )

        # Branch 2: Intraday Microstructure TCN
        self.micro_branch = CausalTCNBranch(
            in_dim=self.config.micro_input_dim,
            hidden_dim=self.config.hidden_dim,
            dilations=self.config.micro_dilations,
            kernel_size=self.config.kernel_size,
            dropout=self.config.dropout,
        )

        # Cross-Attention: Micro Query attends to Macro Key/Value
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.config.hidden_dim,
            num_heads=self.config.num_heads,
            dropout=self.config.dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(self.config.hidden_dim)

        # Sentiment Embedding & GRN
        self.sent_proj = nn.Sequential(
            nn.Linear(self.config.sent_input_dim, self.config.hidden_dim),
            nn.LayerNorm(self.config.hidden_dim),
            nn.GELU(),
        )

        # Multi-modal Fusion GRN
        fusion_dim = self.config.hidden_dim * 3  # [H_cross, H_micro, H_sent]
        self.fusion_grn = GatedResidualNetwork(
            input_dim=fusion_dim,
            hidden_dim=self.config.hidden_dim * 2,
            output_dim=self.config.hidden_dim,
            dropout=self.config.dropout,
        )

        # Multi-Task Output Heads
        # 1. Tri-State Direction Classification Head (SELL=0, HOLD=1, BUY=2)
        self.classification_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim // 2, self.config.num_classes),
        )

        # 2. Multi-Horizon Continuous Return Regression Head
        self.regression_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim // 2, self.config.num_horizons),
        )

        # 3. Uncertainty & Volatility Head
        self.volatility_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim // 2, 1),
            nn.Softplus(),  # Strictly positive volatility
        )

        # 4. Conviction / Confidence Head
        self.conviction_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim // 2, 1),
            nn.Sigmoid(),  # Scaled in [0.0, 1.0]
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with Xavier and Kaiming uniform distributions."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        macro_seq: torch.Tensor,
        micro_seq: torch.Tensor,
        sent_vec: torch.Tensor,
    ) -> PredictionOutput:
        """
        Forward Pass of Multi-Horizon TCT.

        Parameters
        ----------
        macro_seq : torch.Tensor
            Daily macro feature sequence of shape [B, macro_seq_len, macro_input_dim].
        micro_seq : torch.Tensor
            Intraday microstructure sequence of shape [B, micro_seq_len, micro_input_dim].
        sent_vec : torch.Tensor
            Continuous NLP sentiment vector of shape [B, sent_input_dim].

        Returns
        -------
        PredictionOutput
            Structured prediction containing logits, softmax probs, expected returns, volatility, conviction.
        """
        # 1. Process Temporal Branches
        h_macro = self.macro_branch(macro_seq)  # [B, T_macro, hidden_dim]
        h_micro = self.micro_branch(micro_seq)  # [B, T_micro, hidden_dim]

        # 2. Cross-Attention Fusion (Micro Query attends to Macro Key/Value)
        attn_out, _ = self.cross_attention(
            query=h_micro,
            key=h_macro,
            value=h_macro,
        )  # [B, T_micro, hidden_dim]
        h_cross = self.cross_norm(h_micro + attn_out)

        # Extract last time step representations
        h_cross_last = h_cross[:, -1, :]  # [B, hidden_dim]
        h_micro_last = h_micro[:, -1, :]  # [B, hidden_dim]

        # 3. Process Sentiment Vector
        h_sent = self.sent_proj(sent_vec)  # [B, hidden_dim]

        # 4. Multi-modal Fusion via GRN
        fused_input = torch.cat([h_cross_last, h_micro_last, h_sent], dim=-1)  # [B, hidden_dim * 3]
        fused_state = self.fusion_grn(fused_input)  # [B, hidden_dim]

        # 5. Output Heads
        logits = self.classification_head(fused_state)          # [B, 3]
        probs = F.softmax(logits, dim=-1)                       # [B, 3]
        exp_returns = self.regression_head(fused_state)         # [B, num_horizons]
        pred_vol = self.volatility_head(fused_state)            # [B, 1]
        conviction = self.conviction_head(fused_state)          # [B, 1]

        return PredictionOutput(
            direction_logits=logits,
            direction_probs=probs,
            expected_return=exp_returns,
            predicted_volatility=pred_vol,
            conviction_score=conviction,
        )
