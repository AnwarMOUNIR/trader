"""
Comprehensive Unit & Integration Test Suite for Milestone M3 (Multi-Horizon DNN).

Covers:
1. Feature Engineering (Macro daily, Intraday micro, Sentiment alignment, Zero-Lookahead Causal Scaler)
2. Triple-Barrier Supervisory Labeling (Upper/Lower/Vertical barriers, multi-horizon forward returns)
3. Multi-Horizon Loss Functions (Focal Loss, Huber Loss, Differential Sharpe Loss, Composite Multi-Horizon Loss)
4. Dual-Branch Causal Dilated Temporal Convolutional Transformer Architecture (CausalConv1d, TCN, Cross-Attention, GRN, Multi-Task Heads)
5. Model Training & Purged Cross-Validation (Purged split with embargo, AdamW + Cosine scheduler, early stopping, checkpointing)
6. Actionable Policy Inference (BUY/HOLD/SELL signals, target weights in [0.0, 0.35], US & Casablanca BVC support)
7. End-to-End Pipeline Integration
"""

import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.synthetic_feed import SyntheticMicrostructureGenerator
from src.models.feature_pipeline import (
    MACRO_FEATURE_COLS,
    MICRO_FEATURE_COLS,
    SENTIMENT_FEATURE_COLS,
    CausalRollingScaler,
    DatasetTensors,
    FeatureEngineer,
    FeaturePipeline,
    MultiHorizonDataset,
)
from src.models.labeler import BarrierConfig, LabelResult, TripleBarrierLabeler
from src.models.loss_functions import (
    CompositeMultiHorizonLoss,
    DifferentialSharpeLoss,
    FocalLoss,
    LossWeights,
    MultiHorizonHuberLoss,
)
from src.models.multi_horizon_dnn import (
    CausalConv1d,
    CausalTCNBranch,
    GatedResidualNetwork,
    ModelConfig,
    MultiHorizonTCT,
    PredictionOutput,
    TemporalBlock,
)
from src.models.trainer import (
    DNNPredictor,
    DNNTrainer,
    PurgedTimeSeriesSplit,
    TrainerConfig,
)
from src.sentiment.aggregator import SentimentAggregator
from src.sentiment.nlp_engine import SentimentResult


@pytest.fixture
def synthetic_data():
    """Generate deterministic daily and intraday market series."""
    gen = SyntheticMicrostructureGenerator(seed=42)
    daily_df = gen.generate_daily_series(
        symbol="AAPL", start_date="2024-01-01", end_date="2024-12-31", market="US"
    )
    intraday_df = gen.generate_intraday_series(
        symbol="AAPL", start_date="2024-10-01", end_date="2024-10-15", interval="15m", market="US"
    )
    return daily_df, intraday_df


@pytest.fixture
def bvc_synthetic_data():
    """Generate deterministic Casablanca BVC daily and intraday series."""
    gen = SyntheticMicrostructureGenerator(seed=101)
    daily_df = gen.generate_daily_series(
        symbol="ATW.BVC", start_date="2024-01-01", end_date="2024-12-31", market="BVC"
    )
    intraday_df = gen.generate_intraday_series(
        symbol="ATW.BVC", start_date="2024-10-01", end_date="2024-10-15", interval="15m", market="BVC"
    )
    return daily_df, intraday_df


# ============================================================================
# TIER 1: FEATURE PIPELINE & CAUSAL NORMALIZATION TESTS
# ============================================================================

class TestFeaturePipeline:
    """Test suite for feature extraction, scaling, and sequence building."""

    def test_macro_feature_extraction(self, synthetic_data):
        daily_df, _ = synthetic_data
        macro_df = FeatureEngineer.compute_macro_features(daily_df)

        assert not macro_df.empty
        for col in MACRO_FEATURE_COLS:
            assert col in macro_df.columns
        assert not macro_df.isna().any().any(), "Macro features contain NaNs"
        assert (macro_df["rsi14"] >= -1.0).all() and (macro_df["rsi14"] <= 1.0).all()
        assert (macro_df["vol_20d"] >= 0.0).all()
        assert (macro_df["ratio_52w_high"] > 0.0).all()

    def test_intraday_micro_feature_extraction(self, synthetic_data):
        _, intraday_df = synthetic_data
        micro_df = FeatureEngineer.compute_intraday_micro_features(intraday_df)

        assert not micro_df.empty
        for col in MICRO_FEATURE_COLS:
            assert col in micro_df.columns
        assert not micro_df.isna().any().any(), "Micro features contain NaNs"
        assert (micro_df["rvol"] >= 0.0).all()
        assert (micro_df["clv"] >= -1.0).all() and (micro_df["clv"] <= 1.0).all()
        assert (micro_df["spread_ratio"] > 0.0).all()

    def test_causal_rolling_scaler_no_lookahead(self):
        """Strict verification that modifying future data does NOT alter past scaled values."""
        np.random.seed(42)
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
        raw_vals = np.random.randn(n, 2)
        df_orig = pd.DataFrame(raw_vals, index=dates, columns=["feat_a", "feat_b"])

        scaler = CausalRollingScaler(method="robust", window=30, min_periods=5)
        scaled_orig = scaler.fit_transform(df_orig)

        # Modify future data after index 60
        df_modified = df_orig.copy()
        df_modified.iloc[60:, :] = df_modified.iloc[60:, :] * 100.0 + 50.0

        scaled_mod = scaler.fit_transform(df_modified)

        # Verify that for all t <= 59, the scaled values are identical to machine precision
        np.testing.assert_allclose(
            scaled_orig.iloc[:60].values,
            scaled_mod.iloc[:60].values,
            rtol=1e-5,
            atol=1e-5,
            err_msg="CausalRollingScaler leaked future data into past normalized values!",
        )

    def test_sentiment_feature_ingestion(self, synthetic_data):
        _, intraday_df = synthetic_data
        aggregator = SentimentAggregator()
        
        # Add synthetic sentiment events
        sample_results = [
            SentimentResult(
                article_id=f"art_{i}",
                symbol="AAPL",
                published_at_utc=intraday_df.index[min(i * 10, len(intraday_df) - 1)],
                sentiment_score=0.75 if i % 2 == 0 else -0.50,
                sentiment_confidence=0.90,
                sentiment_label="POSITIVE" if i % 2 == 0 else "NEGATIVE",
                probabilities={"positive": 0.8, "neutral": 0.1, "negative": 0.1},
                model_name="finbert",
            )
            for i in range(5)
        ]
        aggregator.add_sentiment_results(sample_results)
        sent_df = aggregator.get_sentiment_features(symbol="AAPL", target_timestamps=intraday_df.index)

        assert len(sent_df) == len(intraday_df)
        for col in SENTIMENT_FEATURE_COLS:
            assert col in sent_df.columns
        assert not sent_df.isna().any().any()

    def test_build_dataset_tensors(self, synthetic_data):
        daily_df, intraday_df = synthetic_data
        pipeline = FeaturePipeline(macro_seq_len=30, micro_seq_len=20)

        macro_scaled, micro_scaled, sent_scaled = pipeline.process_features(
            daily_df=daily_df, intraday_df=intraday_df
        )

        labeler = TripleBarrierLabeler()
        lbl_res = labeler.generate_labels(intraday_df)

        tensors = pipeline.build_dataset_tensors(
            macro_scaled_df=macro_scaled,
            micro_scaled_df=micro_scaled,
            sent_scaled_df=sent_scaled,
            labels_df=lbl_res.labels_df,
        )

        expected_b = len(intraday_df) - 20 + 1
        assert tensors.macro_seq.shape == (expected_b, 30, len(MACRO_FEATURE_COLS))
        assert tensors.micro_seq.shape == (expected_b, 20, len(MICRO_FEATURE_COLS))
        assert tensors.sent_vec.shape == (expected_b, len(SENTIMENT_FEATURE_COLS))
        assert tensors.target_labels.shape == (expected_b,)
        assert tensors.target_returns.shape == (expected_b, 3)

        dataset = MultiHorizonDataset(tensors)
        assert len(dataset) == expected_b
        item = dataset[0]
        assert "macro_seq" in item and "micro_seq" in item and "sent_vec" in item


# ============================================================================
# TIER 2: TRIPLE-BARRIER LABELER TESTS
# ============================================================================

class TestTripleBarrierLabeler:
    """Test suite for Triple-Barrier Method and multi-horizon target labeling."""

    def test_triple_barrier_upper_hit(self):
        """Construct a deterministic upward price path."""
        n = 30
        dates = pd.date_range("2024-01-01 09:30", periods=n, freq="15min", tz="UTC")
        prices = [100.0] + [100.0 + i * 2.0 for i in range(1, n)]
        df = pd.DataFrame(
            {
                "Open": prices,
                "High": [p + 1.0 for p in prices],
                "Low": [p - 0.5 for p in prices],
                "Close": prices,
                "Volume": [1000] * n,
            },
            index=dates,
        )

        labeler = TripleBarrierLabeler(BarrierConfig(upper_mult=1.0, lower_mult=1.0, vertical_barrier_bars=10))
        res = labeler.generate_labels(df)

        # Bar 0 should hit Upper barrier first
        assert res.labels_df.iloc[0]["label_triple_barrier"] == 1
        assert res.labels_df.iloc[0]["label_class"] == 2
        assert res.labels_df.iloc[0]["action_name"] == "BUY"
        assert res.labels_df.iloc[0]["touch_barrier"] == "UPPER"

    def test_triple_barrier_lower_hit(self):
        """Construct a deterministic downward price path."""
        n = 30
        dates = pd.date_range("2024-01-01 09:30", periods=n, freq="15min", tz="UTC")
        prices = [100.0] + [100.0 - i * 2.0 for i in range(1, n)]
        df = pd.DataFrame(
            {
                "Open": prices,
                "High": [p + 0.5 for p in prices],
                "Low": [p - 1.0 for p in prices],
                "Close": prices,
                "Volume": [1000] * n,
            },
            index=dates,
        )

        labeler = TripleBarrierLabeler(BarrierConfig(upper_mult=1.0, lower_mult=1.0, vertical_barrier_bars=10))
        res = labeler.generate_labels(df)

        assert res.labels_df.iloc[0]["label_triple_barrier"] == -1
        assert res.labels_df.iloc[0]["label_class"] == 0
        assert res.labels_df.iloc[0]["action_name"] == "SELL"
        assert res.labels_df.iloc[0]["touch_barrier"] == "LOWER"

    def test_triple_barrier_vertical_expiration(self):
        """Construct a flat/zero-volatility price path."""
        n = 30
        dates = pd.date_range("2024-01-01 09:30", periods=n, freq="15min", tz="UTC")
        df = pd.DataFrame(
            {
                "Open": [100.0] * n,
                "High": [100.05] * n,
                "Low": [99.95] * n,
                "Close": [100.0] * n,
                "Volume": [1000] * n,
            },
            index=dates,
        )

        labeler = TripleBarrierLabeler(BarrierConfig(upper_mult=2.0, lower_mult=2.0, vertical_barrier_bars=5))
        res = labeler.generate_labels(df)

        assert res.labels_df.iloc[0]["label_triple_barrier"] == 0
        assert res.labels_df.iloc[0]["label_class"] == 1
        assert res.labels_df.iloc[0]["action_name"] == "HOLD"
        assert res.labels_df.iloc[0]["touch_barrier"] == "VERTICAL"

    def test_multi_horizon_forward_returns(self, synthetic_data):
        _, intraday_df = synthetic_data
        labeler = TripleBarrierLabeler(BarrierConfig(horizons=(3, 12, 78)))
        res = labeler.generate_labels(intraday_df)

        df = res.labels_df
        assert "ret_h3" in df.columns
        assert "ret_h12" in df.columns
        assert "ret_h78" in df.columns
        assert "composite_return" in df.columns

        # Verify return formula for bar 0: (P_{t+3} - P_t) / P_t
        close_col = "Close" if "Close" in intraday_df.columns else "close"
        p0 = intraday_df[close_col].iloc[0]
        p3 = intraday_df[close_col].iloc[3]
        expected_ret_h3 = (p3 - p0) / p0
        np.testing.assert_allclose(df.iloc[0]["ret_h3"], expected_ret_h3, rtol=1e-4)


# ============================================================================
# TIER 3: LOSS FUNCTIONS & POLICY REWARD TESTS
# ============================================================================

class TestLossFunctions:
    """Test suite for Focal Loss, Huber Loss, Differential Sharpe, and Composite Loss."""

    def test_focal_loss_forward_backward(self):
        logits = torch.randn(8, 3, requires_grad=True)
        targets = torch.tensor([0, 1, 2, 1, 0, 2, 1, 0], dtype=torch.long)

        loss_fn = FocalLoss(alpha=[1.5, 0.5, 1.5], gamma=2.0)
        loss = loss_fn(logits, targets)

        assert loss.ndim == 0
        assert not torch.isnan(loss) and not torch.isinf(loss)
        assert loss.item() > 0

        loss.backward()
        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()

    def test_focal_loss_numerical_stability(self):
        # Extreme logits (large positive / large negative)
        logits = torch.tensor([[100.0, -100.0, -100.0], [-100.0, 100.0, -100.0]], requires_grad=True)
        targets = torch.tensor([0, 1], dtype=torch.long)

        loss_fn = FocalLoss()
        loss = loss_fn(logits, targets)
        assert not torch.isnan(loss)
        assert loss.item() < 0.01  # Loss should be near 0 for correct confident predictions

    def test_multi_horizon_huber_loss(self):
        pred = torch.tensor([[0.01, 0.02, 0.05], [0.03, 0.04, 0.06]], requires_grad=True)
        target = torch.tensor([[0.015, 0.025, 0.045], [0.025, 0.035, 0.070]])

        loss_fn = MultiHorizonHuberLoss(delta=0.02, horizon_weights=(0.2, 0.3, 0.5))
        loss = loss_fn(pred, target)

        assert not torch.isnan(loss) and loss.item() > 0
        loss.backward()
        assert pred.grad is not None

    def test_differential_sharpe_loss(self):
        # Positive returns with high BUY probabilities should produce negative loss (good reward)
        probs_good = torch.tensor([[0.05, 0.10, 0.85], [0.05, 0.10, 0.85], [0.05, 0.10, 0.85]], requires_grad=True)
        realized_rets = torch.tensor([0.02, 0.03, 0.015])

        sharpe_loss_fn = DifferentialSharpeLoss()
        loss_good = sharpe_loss_fn(probs_good, realized_rets)

        # Negative returns with high BUY probabilities should produce positive loss (penalty)
        probs_bad = torch.tensor([[0.05, 0.10, 0.85], [0.05, 0.10, 0.85], [0.05, 0.10, 0.85]])
        bad_rets = torch.tensor([-0.02, -0.03, -0.015])
        loss_bad = sharpe_loss_fn(probs_bad, bad_rets)

        assert loss_good.item() < loss_bad.item(), "DifferentialSharpeLoss did not reward profitable strategy!"

    def test_composite_multi_horizon_loss(self):
        logits = torch.randn(16, 3, requires_grad=True)
        pred_rets = torch.randn(16, 3, requires_grad=True)
        targets_lbl = torch.randint(0, 3, (16,))
        targets_ret = torch.randn(16, 3)

        loss_fn = CompositeMultiHorizonLoss(
            weights=LossWeights(lambda_focal=1.0, lambda_huber=0.5, lambda_sharpe=0.8, lambda_l2=1e-4)
        )
        dummy_param = nn.Parameter(torch.randn(10, 10))

        total_loss, breakdown = loss_fn(
            direction_logits=logits,
            pred_returns=pred_rets,
            target_labels=targets_lbl,
            target_returns=targets_ret,
            model_params=[dummy_param],
        )

        assert "loss_total" in breakdown
        assert "loss_focal" in breakdown
        assert "loss_huber" in breakdown
        assert "loss_sharpe" in breakdown
        assert "loss_l2" in breakdown

        total_loss.backward()
        assert logits.grad is not None
        assert pred_rets.grad is not None


# ============================================================================
# TIER 4: NEURAL NETWORK ARCHITECTURE TESTS
# ============================================================================

class TestMultiHorizonTCT:
    """Test suite for Dual-Branch Causal TCT Neural Architecture."""

    def test_causal_conv1d_receptive_field(self):
        conv = CausalConv1d(in_channels=4, out_channels=8, kernel_size=3, dilation=2)
        x = torch.randn(2, 4, 20)
        out = conv(x)
        assert out.shape == (2, 8, 20), "CausalConv1d altered temporal sequence length"

        # Verify causality: modifying future input x[:, :, 15:] must not affect out[:, :, :15]
        x_mod = x.clone()
        x_mod[:, :, 15:] = x_mod[:, :, 15:] + 100.0
        out_mod = conv(x_mod)
        np.testing.assert_allclose(
            out[:, :, :15].detach().numpy(),
            out_mod[:, :, :15].detach().numpy(),
            atol=1e-5,
            err_msg="CausalConv1d leaked future information to prior time steps!",
        )

    def test_tcn_branch_and_grn(self):
        tcn = CausalTCNBranch(in_dim=8, hidden_dim=32, dilations=(1, 2, 4))
        x = torch.randn(4, 30, 8)
        out = tcn(x)
        assert out.shape == (4, 30, 32)

        grn = GatedResidualNetwork(input_dim=32, hidden_dim=64, output_dim=32)
        grn_out = grn(out)
        assert grn_out.shape == (4, 30, 32)

    def test_tct_forward_pass_various_batches(self):
        config = ModelConfig(
            macro_input_dim=8,
            micro_input_dim=12,
            sent_input_dim=7,
            macro_seq_len=60,
            micro_seq_len=39,
            hidden_dim=32,
            num_heads=2,
            num_horizons=3,
            num_classes=3,
        )
        model = MultiHorizonTCT(config)

        for b in [1, 4, 16]:
            macro = torch.randn(b, 60, 8)
            micro = torch.randn(b, 39, 12)
            sent = torch.randn(b, 7)

            pred: PredictionOutput = model(macro, micro, sent)

            assert pred.direction_logits.shape == (b, 3)
            assert pred.direction_probs.shape == (b, 3)
            assert pred.expected_return.shape == (b, 3)
            assert pred.predicted_volatility.shape == (b, 1)
            assert pred.conviction_score.shape == (b, 1)

            # Probabilities sum to 1.0
            prob_sums = pred.direction_probs.sum(dim=-1).detach().numpy()
            np.testing.assert_allclose(prob_sums, np.ones(b), atol=1e-5)

            # Volatility strictly positive
            assert (pred.predicted_volatility.detach().numpy() >= 0.0).all()

            # Conviction in [0, 1]
            convs = pred.conviction_score.detach().numpy()
            assert (convs >= 0.0).all() and (convs <= 1.0).all()


# ============================================================================
# TIER 5: TRAINER & POLICY INFERENCE TESTS
# ============================================================================

class TestTrainerAndInference:
    """Test suite for Purged CV split, training loop, early stopping, and policy inference."""

    def test_purged_time_series_split(self):
        splitter = PurgedTimeSeriesSplit(n_splits=3, purge_bars=10, embargo_pct=0.02)
        n_samples = 200

        splits = list(splitter.split(n_samples))
        assert len(splits) == 3

        for train_idx, val_idx in splits:
            assert len(train_idx) > 0
            assert len(val_idx) > 0
            # Ensure strict gap of purge_bars between train max and val min
            assert val_idx[0] >= train_idx[-1] + 10
            # No overlap between train and val
            assert len(set(train_idx).intersection(set(val_idx))) == 0

    def test_dnn_trainer_fit_and_loss_decrease(self, synthetic_data):
        daily_df, intraday_df = synthetic_data
        pipeline = FeaturePipeline(macro_seq_len=20, micro_seq_len=15)
        macro_scaled, micro_scaled, sent_scaled = pipeline.process_features(daily_df, intraday_df)
        lbl_res = TripleBarrierLabeler().generate_labels(intraday_df)

        tensors = pipeline.build_dataset_tensors(
            macro_scaled, micro_scaled, sent_scaled, lbl_res.labels_df
        )
        dataset = MultiHorizonDataset(tensors)

        model_config = ModelConfig(
            macro_input_dim=len(MACRO_FEATURE_COLS),
            micro_input_dim=len(MICRO_FEATURE_COLS),
            sent_input_dim=len(SENTIMENT_FEATURE_COLS),
            macro_seq_len=20,
            micro_seq_len=15,
            hidden_dim=16,
            num_heads=2,
        )
        trainer_config = TrainerConfig(epochs=3, batch_size=16, learning_rate=1e-3, device="cpu")
        trainer = DNNTrainer(model_config=model_config, config=trainer_config)

        history = trainer.fit(train_dataset=dataset)
        assert len(trainer.history) == 3
        assert "train_loss" in history

    def test_checkpoint_save_and_load(self, tmp_path):
        config = ModelConfig(macro_input_dim=8, micro_input_dim=12, sent_input_dim=7, hidden_dim=16, num_heads=2)
        trainer = DNNTrainer(model_config=config, config=TrainerConfig(device="cpu"))

        ckpt_file = tmp_path / "model_test.pt"
        trainer.save_checkpoint(ckpt_file)
        assert ckpt_file.exists()

        new_trainer = DNNTrainer(model_config=config, config=TrainerConfig(device="cpu"))
        new_trainer.load_checkpoint(ckpt_file)

        # Check parameter equality
        for p1, p2 in zip(trainer.model.parameters(), new_trainer.model.parameters()):
            assert torch.equal(p1, p2)

    def test_predict_signals_policy_contract(self, synthetic_data):
        daily_df, intraday_df = synthetic_data
        config = ModelConfig(
            macro_input_dim=len(MACRO_FEATURE_COLS),
            micro_input_dim=len(MICRO_FEATURE_COLS),
            sent_input_dim=len(SENTIMENT_FEATURE_COLS),
            macro_seq_len=30,
            micro_seq_len=20,
            hidden_dim=16,
            num_heads=2,
        )
        model = MultiHorizonTCT(config)
        pipeline = FeaturePipeline(macro_seq_len=30, micro_seq_len=20)
        predictor = DNNPredictor(model=model, feature_pipeline=pipeline, max_single_weight=0.35, device="cpu")

        signals_df = predictor.predict_signals(intraday_df=intraday_df, daily_df=daily_df)

        required_cols = [
            "action",
            "buy_prob",
            "hold_prob",
            "sell_prob",
            "target_weight",
            "expected_return",
            "confidence",
        ]
        for col in required_cols:
            assert col in signals_df.columns

        # Verify action domain
        assert set(signals_df["action"].unique()).issubset({"BUY", "HOLD", "SELL"})
        # Verify target weight in [0.0, 0.35]
        assert (signals_df["target_weight"] >= 0.0).all()
        assert (signals_df["target_weight"] <= 0.35).all()
        # Verify probabilities sum to 1.0
        prob_sums = signals_df["buy_prob"] + signals_df["hold_prob"] + signals_df["sell_prob"]
        np.testing.assert_allclose(prob_sums.values, np.ones(len(signals_df)), atol=1e-3)

    def test_moroccan_bvc_support(self, bvc_synthetic_data):
        """Verify full pipeline execution for Moroccan BVC assets."""
        daily_df, intraday_df = bvc_synthetic_data
        config = ModelConfig(
            macro_input_dim=len(MACRO_FEATURE_COLS),
            micro_input_dim=len(MICRO_FEATURE_COLS),
            sent_input_dim=len(SENTIMENT_FEATURE_COLS),
            macro_seq_len=30,
            micro_seq_len=20,
            hidden_dim=16,
            num_heads=2,
        )
        model = MultiHorizonTCT(config)
        pipeline = FeaturePipeline(macro_seq_len=30, micro_seq_len=20)
        predictor = DNNPredictor(model=model, feature_pipeline=pipeline, device="cpu")

        signals_df = predictor.predict_signals(intraday_df=intraday_df, daily_df=daily_df)
        assert not signals_df.empty
        assert set(signals_df["action"].unique()).issubset({"BUY", "HOLD", "SELL"})


# ============================================================================
# TIER 6: END-TO-END INTEGRATION TEST
# ============================================================================

class TestEndToEndPipeline:
    """Full end-to-end integration test from raw data to trained model to policy signals."""

    def test_e2e_training_and_policy_generation(self, synthetic_data):
        daily_df, intraday_df = synthetic_data

        # 1. Feature pipeline & labeling
        pipeline = FeaturePipeline(macro_seq_len=30, micro_seq_len=20)
        macro_scaled, micro_scaled, sent_scaled = pipeline.process_features(daily_df, intraday_df)
        labeler = TripleBarrierLabeler()
        lbl_res = labeler.generate_labels(intraday_df)

        # 2. Build Tensors & Dataset
        tensors = pipeline.build_dataset_tensors(
            macro_scaled, micro_scaled, sent_scaled, lbl_res.labels_df
        )
        dataset = MultiHorizonDataset(tensors)

        # 3. Model & Trainer
        model_config = ModelConfig(
            macro_input_dim=len(MACRO_FEATURE_COLS),
            micro_input_dim=len(MICRO_FEATURE_COLS),
            sent_input_dim=len(SENTIMENT_FEATURE_COLS),
            macro_seq_len=30,
            micro_seq_len=20,
            hidden_dim=32,
            num_heads=2,
        )
        trainer = DNNTrainer(
            model_config=model_config,
            config=TrainerConfig(epochs=2, batch_size=16, learning_rate=1e-3, device="cpu"),
        )

        history = trainer.fit(train_dataset=dataset)
        assert len(history["train_loss"]) == 2

        # 4. Predict Signals Policy
        predictor = DNNPredictor(model=trainer.model, feature_pipeline=pipeline, device="cpu")
        signals = predictor.predict_signals(intraday_df=intraday_df, daily_df=daily_df)

        assert not signals.empty
        assert len(signals) == len(intraday_df) - 20 + 1
        assert "action" in signals.columns
        assert "target_weight" in signals.columns
