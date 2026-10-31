"""Unit tests for AlpacaPaperTrader engine and execution logic."""
from pathlib import Path
import json
import pytest

from market_research.paper_trader import AlpacaPaperTrader


def test_paper_trader_drawdown_dial_scaling(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "test_key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test_secret")

    trader = AlpacaPaperTrader(
        state_path=tmp_path / "state.json",
        audit_path=tmp_path / "audit.jsonl",
        soft_floor_dollars=6_000.0,
        hard_floor_dollars=8_500.0,
    )

    # 1. Zero drawdown -> 100% dial
    dial, reason = trader.compute_drawdown_dial(current_equity=100_000.0, high_water=100_000.0)
    assert dial == 1.0
    assert "NORMAL_REGIME" in reason

    # 2. Drawdown at $6,000 (soft floor) -> 1.0
    dial, reason = trader.compute_drawdown_dial(current_equity=94_000.0, high_water=100_000.0)
    assert dial == pytest.approx(1.0)
    assert "SOFT_FLOOR" in reason

    # 3. Drawdown midway at $7,250 -> scaled between 1.0 and 0.20
    dial, reason = trader.compute_drawdown_dial(current_equity=92_750.0, high_water=100_000.0)
    assert 0.20 < dial < 1.0
    assert "SOFT_FLOOR" in reason

    # 4. Drawdown >= $8,500 (hard floor) -> 0.0 (100% Cash lock)
    dial, reason = trader.compute_drawdown_dial(current_equity=91_500.0, high_water=100_000.0)
    assert dial == 0.0
    assert "HARD_FLOOR" in reason


def test_paper_trader_state_persistence(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "test_key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test_secret")

    state_file = tmp_path / "state.json"
    trader = AlpacaPaperTrader(
        state_path=state_file,
        audit_path=tmp_path / "audit.jsonl",
    )

    # First load initializes
    state = trader.load_state(current_equity=99_000.0)
    assert state["high_water_mark"] == 99_000.0

    # Higher equity updates peak
    new_peak = trader.update_high_water(current_equity=105_000.0)
    assert new_peak == 105_000.0
    saved = json.loads(state_file.read_text(encoding="utf-8"))
    assert saved["high_water_mark"] == 105_000.0

    # Lower equity preserves previous peak
    retained_peak = trader.update_high_water(current_equity=101_000.0)
    assert retained_peak == 105_000.0


def test_paper_trader_missing_credentials_raises(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)

    with pytest.raises(ValueError, match="Alpaca API credentials missing"):
        AlpacaPaperTrader()
