## 2026-08-31T15:43:00Z
You are m3_worker_1, a teamwork_preview_worker for Milestone M3 (Multi-Horizon Deep Neural Network).
Your working directory is: c:\Users\Anwar\Documents\trader\.agents\m3_worker_1
Your parent orchestrator is: c:\Users\Anwar\Documents\trader\.agents\orchestrator_1 (conversation ID: 221d5d39-4a0b-46d8-8d77-134da45be38f)
Authoritative user request file: c:\Users\Anwar\Documents\trader\.agents\ORIGINAL_REQUEST.md
Master project specification: c:\Users\Anwar\Documents\trader\PROJECT.md
Detailed technical analysis reference: c:\Users\Anwar\Documents\trader\.agents\survey_explorer_2\analysis.md
Mathematical specification reference: c:\Users\Anwar\Documents\trader\.agents\survey_spec_miner_1\analysis.md
M1 Data Pipeline implementation reference: c:\Users\Anwar\Documents\trader\src\data\
M2 Sentiment Pipeline implementation reference: c:\Users\Anwar\Documents\trader\src\sentiment\

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Scope & Exclusively Owned Files:
- `src/models/__init__.py`
- `src/models/multi_horizon_dnn.py`
- `src/models/feature_pipeline.py`
- `src/models/labeler.py`
- `src/models/loss_functions.py`
- `src/models/trainer.py`
- `tests/test_model_pipeline.py`

Task:
Implement the complete Multi-Horizon Deep Neural Network model and pipeline meeting all R3 requirements:
1. `src/models/feature_pipeline.py`:
   - Extracts macro features from 30y daily bars (50/200 SMA, 14 RSI, 20d volatility, 12m momentum, 52w high ratio, 14 ATR).
   - Computes intraday microstructure features on 5m/15m/60m bars (RVOL, VWAP distance, Order Flow Imbalance / VSA proxies, short-term momentum, volatility).
   - Ingests the 7 continuous sentiment features from `src/sentiment/aggregator.py`.
   - Normalizes features using causal rolling RobustScaler or z-score with zero lookahead bias.
2. `src/models/labeler.py`:
   - Implements Marcos López de Prado's Triple-Barrier Method (+1.5σ upper profit target, -1.0σ lower stop-loss target, 12-bar vertical barrier).
   - Maps path outcomes to discrete targets (BUY=1, HOLD=0, SELL=-1) and multi-horizon forward returns (h in {3, 12, 78} bars).
3. `src/models/loss_functions.py`:
   - Multi-horizon loss combining:
     a) Multi-Class Focal Cross-Entropy Loss with class-weighting (for BUY/HOLD/SELL tri-state classification).
     b) Huber Loss for multi-horizon continuous return regression.
     c) Differential Sharpe Ratio loss regularizer for policy optimization.
4. `src/models/multi_horizon_dnn.py`:
   - Dual-Branch Causal Dilated Temporal Convolutional Transformer (TCT) in PyTorch:
     - Branch 1: Dilated Causal TCN processing 30-day macro daily sequence.
     - Branch 2: Dilated Causal TCN processing 60-bar intraday sequence.
     - Cross-Attention mechanism fusing macro context with high-frequency microstructure.
     - Gated Residual Network (GRN) integrating real-time sentiment features.
     - Output heads: Classification logits (BUY, HOLD, SELL), Multi-horizon expected returns, and Uncertainty/confidence.
5. `src/models/trainer.py`:
   - Model training loop with Purged Time-Series split, early stopping, and checkpointing.
   - `predict_signals(features_df: pd.DataFrame) -> pd.DataFrame` producing actionable policy:
     `["action", "buy_prob", "hold_prob", "sell_prob", "target_weight", "expected_return", "confidence"]`
     where `action ∈ {"BUY", "HOLD", "SELL"}` and `target_weight ∈ [0.0, 0.35]`.
   - Supports both US and Moroccan BVC equities.
6. `tests/test_model_pipeline.py`:
   - Comprehensive unit and integration test suite covering feature engineering, triple-barrier labeling, loss functions, neural network architecture forward pass, model training & inference, and zero-lookahead verification.
7. Run `pytest tests/test_model_pipeline.py` to ensure 100% pass.
8. Write your execution report and handoff to `c:\Users\Anwar\Documents\trader\.agents\m3_worker_1\handoff.md`.
Send a completion message when finished.
