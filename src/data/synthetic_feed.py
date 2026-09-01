"""
High-Fidelity Synthetic Microstructure & Market Data Generator.
Implements Merton Jump-Diffusion stochastic price dynamics, U-shaped intraday
volume profiles, volatility clustering, Wyckoff institutional volume bursts,
and Casablanca Stock Exchange (BVC) regulatory circuit breaker clamping.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger("DataPipeline.SyntheticFeed")


@dataclass
class MarketMicrostructureParams:
    """Microstructure and stochastic parameters for asset simulation."""
    market: str = "US"                           # "US" or "BVC"
    currency: str = "USD"                        # "USD" or "MAD"
    initial_price: float = 150.0
    annual_drift: float = 0.12
    annual_volatility: float = 0.28
    jump_intensity: float = 3.0                  # Average number of jumps per year (Poisson lambda)
    jump_mean: float = 0.0                       # Mean jump size
    jump_std: float = 0.04                       # Jump size standard deviation
    vol_clustering_alpha: float = 0.15           # ARCH shock memory
    vol_clustering_beta: float = 0.80            # GARCH persistence
    base_daily_volume: float = 10_000_000.0
    u_shape_a: float = 3.5                       # Quadratic coefficient for intraday U-curve
    u_shape_b: float = 0.35                      # Baseline intraday volume floor
    circuit_breaker_limit: Optional[float] = None  # e.g. 0.10 for BVC (+/-10%)
    market_cap: float = 50_000_000_000.0
    spread_bps: float = 2.0                      # Basis points half-spread


# Standard asset presets calibrated to historical stylized facts
ASSET_PRESETS: Dict[str, MarketMicrostructureParams] = {
    # US Mega-Cap & Momentum Tech
    "NVDA": MarketMicrostructureParams(
        market="US", currency="USD", initial_price=120.0, annual_drift=0.55,
        annual_volatility=0.45, jump_intensity=6.0, jump_mean=0.01, jump_std=0.06,
        base_daily_volume=45_000_000.0, market_cap=2_800_000_000_000.0, spread_bps=1.5
    ),
    "AAPL": MarketMicrostructureParams(
        market="US", currency="USD", initial_price=180.0, annual_drift=0.22,
        annual_volatility=0.22, jump_intensity=2.5, jump_mean=0.005, jump_std=0.03,
        base_daily_volume=50_000_000.0, market_cap=3_000_000_000_000.0, spread_bps=1.0
    ),
    "MSFT": MarketMicrostructureParams(
        market="US", currency="USD", initial_price=400.0, annual_drift=0.24,
        annual_volatility=0.23, jump_intensity=2.5, jump_mean=0.005, jump_std=0.03,
        base_daily_volume=22_000_000.0, market_cap=3_100_000_000_000.0, spread_bps=1.0
    ),
    "AMZN": MarketMicrostructureParams(
        market="US", currency="USD", initial_price=175.0, annual_drift=0.25,
        annual_volatility=0.28, jump_intensity=3.5, jump_mean=0.005, jump_std=0.04,
        base_daily_volume=35_000_000.0, market_cap=1_900_000_000_000.0, spread_bps=1.5
    ),
    "META": MarketMicrostructureParams(
        market="US", currency="USD", initial_price=480.0, annual_drift=0.35,
        annual_volatility=0.36, jump_intensity=5.0, jump_mean=0.01, jump_std=0.06,
        base_daily_volume=18_000_000.0, market_cap=1_250_000_000_000.0, spread_bps=2.0
    ),
    "GOOGL": MarketMicrostructureParams(
        market="US", currency="USD", initial_price=160.0, annual_drift=0.20,
        annual_volatility=0.25, jump_intensity=3.0, jump_mean=0.005, jump_std=0.035,
        base_daily_volume=25_000_000.0, market_cap=2_000_000_000_000.0, spread_bps=1.5
    ),
    "TSLA": MarketMicrostructureParams(
        market="US", currency="USD", initial_price=220.0, annual_drift=0.30,
        annual_volatility=0.52, jump_intensity=8.0, jump_mean=0.00, jump_std=0.08,
        base_daily_volume=65_000_000.0, market_cap=750_000_000_000.0, spread_bps=2.5
    ),
    "PLTR": MarketMicrostructureParams(
        market="US", currency="USD", initial_price=25.0, annual_drift=0.45,
        annual_volatility=0.48, jump_intensity=6.0, jump_mean=0.01, jump_std=0.07,
        base_daily_volume=40_000_000.0, market_cap=60_000_000_000.0, spread_bps=3.0
    ),
    "SPY": MarketMicrostructureParams(
        market="US", currency="USD", initial_price=450.0, annual_drift=0.11,
        annual_volatility=0.16, jump_intensity=1.5, jump_mean=0.00, jump_std=0.02,
        base_daily_volume=70_000_000.0, market_cap=500_000_000_000.0, spread_bps=0.5
    ),

    # Casablanca Stock Exchange (BVC) Blue Chips
    "ATW.BVC": MarketMicrostructureParams(
        market="BVC", currency="MAD", initial_price=480.0, annual_drift=0.16,
        annual_volatility=0.14, jump_intensity=2.0, jump_mean=0.00, jump_std=0.025,
        base_daily_volume=80_000.0, circuit_breaker_limit=0.10,
        market_cap=110_000_000_000.0, spread_bps=15.0
    ),
    "BCP.BVC": MarketMicrostructureParams(
        market="BVC", currency="MAD", initial_price=295.0, annual_drift=0.12,
        annual_volatility=0.13, jump_intensity=1.8, jump_mean=0.00, jump_std=0.02,
        base_daily_volume=60_000.0, circuit_breaker_limit=0.10,
        market_cap=60_000_000_000.0, spread_bps=15.0
    ),
    "IAM.BVC": MarketMicrostructureParams(
        market="BVC", currency="MAD", initial_price=98.0, annual_drift=0.06,
        annual_volatility=0.12, jump_intensity=1.5, jump_mean=0.00, jump_std=0.02,
        base_daily_volume=120_000.0, circuit_breaker_limit=0.10,
        market_cap=85_000_000_000.0, spread_bps=12.0
    ),
    "BOA.BVC": MarketMicrostructureParams(
        market="BVC", currency="MAD", initial_price=190.0, annual_drift=0.14,
        annual_volatility=0.15, jump_intensity=2.0, jump_mean=0.00, jump_std=0.025,
        base_daily_volume=45_000.0, circuit_breaker_limit=0.10,
        market_cap=40_000_000_000.0, spread_bps=18.0
    ),
    "CSR.BVC": MarketMicrostructureParams(
        market="BVC", currency="MAD", initial_price=210.0, annual_drift=0.10,
        annual_volatility=0.16, jump_intensity=2.0, jump_mean=0.00, jump_std=0.03,
        base_daily_volume=35_000.0, circuit_breaker_limit=0.10,
        market_cap=20_000_000_000.0, spread_bps=20.0
    ),
    "LBV.BVC": MarketMicrostructureParams(
        market="BVC", currency="MAD", initial_price=4400.0, annual_drift=0.18,
        annual_volatility=0.18, jump_intensity=2.5, jump_mean=0.00, jump_std=0.03,
        base_daily_volume=3_500.0, circuit_breaker_limit=0.10,
        market_cap=13_000_000_000.0, spread_bps=22.0
    ),
    "HPS.BVC": MarketMicrostructureParams(
        market="BVC", currency="MAD", initial_price=580.0, annual_drift=0.25,
        annual_volatility=0.24, jump_intensity=3.5, jump_mean=0.01, jump_std=0.04,
        base_daily_volume=15_000.0, circuit_breaker_limit=0.10,
        market_cap=4_500_000_000.0, spread_bps=25.0
    ),
    "TQM.BVC": MarketMicrostructureParams(
        market="BVC", currency="MAD", initial_price=1450.0, annual_drift=0.11,
        annual_volatility=0.17, jump_intensity=1.8, jump_mean=0.00, jump_std=0.025,
        base_daily_volume=6_000.0, circuit_breaker_limit=0.10,
        market_cap=6_000_000_000.0, spread_bps=20.0
    ),
    "TGCC.BVC": MarketMicrostructureParams(
        market="BVC", currency="MAD", initial_price=280.0, annual_drift=0.28,
        annual_volatility=0.26, jump_intensity=3.0, jump_mean=0.01, jump_std=0.04,
        base_daily_volume=50_000.0, circuit_breaker_limit=0.10,
        market_cap=9_000_000_000.0, spread_bps=22.0
    ),
    "MSA.BVC": MarketMicrostructureParams(
        market="BVC", currency="MAD", initial_price=330.0, annual_drift=0.22,
        annual_volatility=0.18, jump_intensity=2.2, jump_mean=0.005, jump_std=0.03,
        base_daily_volume=70_000.0, circuit_breaker_limit=0.10,
        market_cap=25_000_000_000.0, spread_bps=16.0
    ),
    "CIM.BVC": MarketMicrostructureParams(
        market="BVC", currency="MAD", initial_price=1750.0, annual_drift=0.09,
        annual_volatility=0.15, jump_intensity=1.6, jump_mean=0.00, jump_std=0.025,
        base_daily_volume=8_000.0, circuit_breaker_limit=0.10,
        market_cap=25_000_000_000.0, spread_bps=20.0
    ),
    "ADH.BVC": MarketMicrostructureParams(
        market="BVC", currency="MAD", initial_price=28.0, annual_drift=0.32,
        annual_volatility=0.38, jump_intensity=4.5, jump_mean=0.00, jump_std=0.06,
        base_daily_volume=350_000.0, circuit_breaker_limit=0.10,
        market_cap=12_000_000_000.0, spread_bps=30.0
    ),
}


class SyntheticMicrostructureGenerator:
    """
    Generates institutional-grade multi-resolution synthetic market time series.
    Supports Merton jump-diffusion, GARCH volatility clustering, empirical U-shaped
    intraday volume distributions, Wyckoff institutional accumulation regimes, and
    Casablanca BVC continuous circuit-breaker clamping.
    """

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed

    def _get_rng(self, symbol: str, extra_salt: int = 0) -> np.random.Generator:
        """Derive reproducible seed from symbol hash and optional master seed."""
        sym_hash = int.from_bytes(symbol.encode("utf-8"), "little") % (2**31 - 1)
        combined_seed = sym_hash ^ extra_salt if self.seed is None else (self.seed + sym_hash + extra_salt) % (2**31 - 1)
        return np.random.default_rng(combined_seed)

    def _resolve_params(
        self,
        symbol: str,
        market: Optional[str] = None,
        custom_params: Optional[MarketMicrostructureParams] = None
    ) -> MarketMicrostructureParams:
        """Resolve microstructure parameters from presets or defaults."""
        if custom_params is not None:
            return custom_params
        if symbol in ASSET_PRESETS:
            return ASSET_PRESETS[symbol]
        
        # Determine market from suffix or explicit param
        is_bvc = market == "BVC" or symbol.upper().endswith(".BVC") or symbol.upper().endswith("_BVC")
        if is_bvc:
            return MarketMicrostructureParams(
                market="BVC",
                currency="MAD",
                initial_price=300.0,
                annual_drift=0.14,
                annual_volatility=0.18,
                jump_intensity=2.0,
                base_daily_volume=50_000.0,
                circuit_breaker_limit=0.10,
                market_cap=20_000_000_000.0,
                spread_bps=18.0
            )
        else:
            return MarketMicrostructureParams(
                market="US",
                currency="USD",
                initial_price=100.0,
                annual_drift=0.15,
                annual_volatility=0.28,
                jump_intensity=3.0,
                base_daily_volume=15_000_000.0,
                circuit_breaker_limit=None,
                market_cap=50_000_000_000.0,
                spread_bps=2.0
            )

    def generate_jump_diffusion_returns(
        self,
        n_steps: int,
        dt: float,
        params: MarketMicrostructureParams,
        rng: np.random.Generator
    ) -> np.ndarray:
        """
        Simulate asset log-returns via Merton (1976) Jump-Diffusion with GARCH(1,1) volatility clustering.
        r_t = (mu - 0.5 * sigma_t^2 - lambda * k) * dt + sigma_t * sqrt(dt) * Z_t + sum_{j=1}^{N_t} Y_j
        where:
        - sigma_t^2 evolves via GARCH(1,1): sigma_t^2 = omega + alpha * epsilon_{t-1}^2 + beta * sigma_{t-1}^2
        - N_t ~ Poisson(lambda * dt)
        - Y_j ~ Normal(mu_J, sigma_J^2)
        - k = exp(mu_J + 0.5 * sigma_J^2) - 1
        """
        # Long-run variance and GARCH setup
        long_run_var = params.annual_volatility ** 2
        omega = long_run_var * (1.0 - params.vol_clustering_alpha - params.vol_clustering_beta)
        if omega <= 0:
            omega = 0.05 * long_run_var
            alpha = 0.10
            beta = 0.85
        else:
            alpha = params.vol_clustering_alpha
            beta = params.vol_clustering_beta

        # Compensator for Merton martingale condition
        k = np.exp(params.jump_mean + 0.5 * params.jump_std**2) - 1.0
        jump_rate_per_step = params.jump_intensity * dt

        returns = np.zeros(n_steps, dtype=np.float64)
        current_var = long_run_var

        # Pre-generate standard normals and Poisson jump counts for speed
        z_shocks = rng.standard_normal(n_steps)
        # Student-t innovation blend for fat tails
        t_shocks = rng.standard_t(df=5.0, size=n_steps) * np.sqrt(3.0 / 5.0)
        blended_shocks = 0.70 * z_shocks + 0.30 * t_shocks

        poisson_jumps = rng.poisson(jump_rate_per_step, size=n_steps)

        for t in range(n_steps):
            sigma_t = np.sqrt(max(current_var, 1e-6))
            # Continuous diffusion component
            diff_drift = (params.annual_drift - 0.5 * current_var - params.jump_intensity * k) * dt
            diff_return = diff_drift + sigma_t * np.sqrt(dt) * blended_shocks[t]

            # Discontinuous jump component
            jump_count = poisson_jumps[t]
            if jump_count > 0:
                jump_return = np.sum(rng.normal(params.jump_mean, params.jump_std, size=jump_count))
            else:
                jump_return = 0.0

            r_t = diff_return + jump_return

            # Apply BVC Circuit Breaker if applicable
            if params.circuit_breaker_limit is not None and dt >= (1.0 / 252.0):
                # Daily circuit breaker clamping
                limit = params.circuit_breaker_limit - 0.001
                r_t = np.clip(r_t, -limit, limit)
            elif params.circuit_breaker_limit is not None:
                # Intraday tick clamp
                tick_limit = min(params.circuit_breaker_limit, 0.03)
                r_t = np.clip(r_t, -tick_limit, tick_limit)

            returns[t] = r_t

            # Update GARCH variance
            annualized_shock = (r_t / np.sqrt(dt)) ** 2
            current_var = omega + alpha * annualized_shock + beta * current_var

        return returns

    def generate_intraday_volume_profile(
        self,
        n_bars: int,
        a: float = 3.5,
        b: float = 0.35
    ) -> np.ndarray:
        """
        Compute empirical U-shaped intraday volume distribution curve:
        U(x) = a * (x - 0.5)^2 + b,  x in [0, 1]
        Returns normalized array summing to 1.0.
        """
        if n_bars <= 1:
            return np.ones(n_bars, dtype=np.float64)
        
        x = np.linspace(0.0, 1.0, n_bars)
        u_curve = a * ((x - 0.5) ** 2) + b
        # Ensure positivity and normalize
        u_curve = np.maximum(u_curve, 0.05)
        return u_curve / np.sum(u_curve)

    def generate_daily_series(
        self,
        symbol: str,
        start_date: str = "1995-01-01",
        end_date: str = "2026-01-01",
        market: Optional[str] = None,
        custom_params: Optional[MarketMicrostructureParams] = None
    ) -> pd.DataFrame:
        """
        Generate 30-year multi-decade daily OHLCV series.
        Includes Wyckoff accumulation/distribution regimes and macroeconomic cycles.
        """
        params = self._resolve_params(symbol, market, custom_params)
        rng = self._get_rng(symbol, extra_salt=101)

        dates = pd.date_range(start=start_date, end=end_date, freq="B", tz="UTC")
        n = len(dates)
        if n == 0:
            raise ValueError(f"Date range {start_date} to {end_date} generated 0 business days.")

        dt = 1.0 / 252.0
        log_returns = self.generate_jump_diffusion_returns(n, dt, params, rng)

        # Inject Wyckoff Smart-Money Markup / Markdown regimes
        num_cycles = max(3, int(n / (252 * 3)))
        volume_series = rng.lognormal(mean=np.log(params.base_daily_volume), sigma=0.45, size=n)
        is_whale_active = np.zeros(n, dtype=np.int32)

        for _ in range(num_cycles * 2):
            start_idx = rng.integers(30, max(31, n - 60))
            duration = rng.integers(15, 50)
            end_idx = min(n, start_idx + duration)
            surge_mult = rng.uniform(2.5, 5.5)
            volume_series[start_idx:end_idx] *= surge_mult
            # Positive absorption drift or distribution drift
            drift_boost = rng.uniform(0.003, 0.012) if rng.random() > 0.3 else -rng.uniform(0.003, 0.010)
            log_returns[start_idx:end_idx] += drift_boost
            is_whale_active[start_idx:end_idx] = 1

        # Clamping for BVC if needed
        if params.circuit_breaker_limit is not None:
            cb = params.circuit_breaker_limit - 0.002
            log_returns = np.clip(log_returns, -cb, cb)

        # Compute price series
        price_trajectory = params.initial_price * np.exp(np.cumsum(log_returns))
        # Prevent zero or negative prices
        price_trajectory = np.maximum(price_trajectory, 0.50)

        # Generate realistic OHLC bar bounds
        bar_vol_factor = params.annual_volatility * np.sqrt(dt) * rng.uniform(0.6, 1.6, size=n)
        close = price_trajectory
        high = close * (1.0 + bar_vol_factor * rng.uniform(0.3, 0.85, size=n))
        low = close * (1.0 - bar_vol_factor * rng.uniform(0.3, 0.85, size=n))
        
        # Open price connects smoothly with prior close + gap shock
        gap_shocks = rng.normal(0.0, bar_vol_factor * 0.3, size=n)
        open_price = np.empty(n, dtype=np.float64)
        open_price[0] = params.initial_price
        for i in range(1, n):
            open_price[i] = close[i - 1] * (1.0 + gap_shocks[i])

        # Enforce OHLC consistency: High >= max(Open, Close) and Low <= min(Open, Close)
        high = np.maximum(high, np.maximum(open_price, close))
        low = np.minimum(low, np.minimum(open_price, close))
        low = np.maximum(low, 0.10)  # Floor

        # VWAP approximation for daily bars
        typical_price = (high + low + close) / 3.0
        vwap = typical_price

        # Estimated trade count
        trade_counts = np.maximum(10, (volume_series / rng.uniform(150, 400, size=n)).astype(np.int32))

        df = pd.DataFrame({
            "open": np.round(open_price, 2),
            "high": np.round(high, 2),
            "low": np.round(low, 2),
            "close": np.round(close, 2),
            "volume": np.round(volume_series, 0),
            "vwap": np.round(vwap, 2),
            "trade_count": trade_counts,
            "market": params.market,
            "currency": params.currency,
            "market_cap": params.market_cap,
            "whale_accumulation": is_whale_active
        }, index=dates)

        return df

    def generate_intraday_series(
        self,
        symbol: str,
        days: int = 60,
        interval: str = "5m",
        market: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        custom_params: Optional[MarketMicrostructureParams] = None
    ) -> pd.DataFrame:
        """
        Generate high-resolution intraday OHLCV bars (5m, 15m, 30m, 60m).
        Accurately enforces session trading hours:
        - US Market: 09:30 - 16:00 EST / EDT (13:30 - 20:00 UTC) -> 78 bars @ 5m / 6.5 hours.
        - Casablanca BVC: 09:30 - 15:30 UTC+1 (08:30 - 14:30 UTC) -> 72 bars @ 5m / 6.0 hours.
        """
        params = self._resolve_params(symbol, market, custom_params)
        rng = self._get_rng(symbol, extra_salt=202)

        # Parse interval minutes
        interval_minutes_map = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "60m": 60, "1h": 60}
        bar_minutes = interval_minutes_map.get(interval.lower(), 5)

        # Determine date range
        if end_date is not None:
            end_dt = pd.to_datetime(end_date, utc=True)
        else:
            end_dt = pd.Timestamp.now(tz="UTC")

        if start_date is not None:
            start_dt = pd.to_datetime(start_date, utc=True)
        else:
            start_dt = end_dt - pd.Timedelta(days=int(days * 1.5))

        trading_days = pd.date_range(start=start_dt.normalize(), end=end_dt.normalize(), freq="B", tz="UTC")
        if len(trading_days) > days:
            trading_days = trading_days[-days:]
        
        if len(trading_days) == 0:
            trading_days = pd.date_range(end=end_dt.normalize(), periods=days, freq="B", tz="UTC")

        # Define session parameters
        if params.market == "BVC":
            session_start_utc = time(8, 30)
            session_end_utc = time(14, 30)
            session_minutes = 360  # 6 hours
        else:
            session_start_utc = time(13, 30)
            session_end_utc = time(20, 0)
            session_minutes = 390  # 6.5 hours

        bars_per_session = session_minutes // bar_minutes
        u_profile = self.generate_intraday_volume_profile(bars_per_session, params.u_shape_a, params.u_shape_b)

        # Intraday dt relative to annual trading year (252 days * bars_per_session)
        dt = 1.0 / (252.0 * bars_per_session)

        all_timestamps: List[pd.Timestamp] = []
        all_opens: List[float] = []
        all_highs: List[float] = []
        all_lows: List[float] = []
        all_closes: List[float] = []
        all_volumes: List[float] = []
        all_vwaps: List[float] = []
        all_trade_counts: List[int] = []

        current_price = params.initial_price

        for day in trading_days:
            # Generate day's log-returns
            day_returns = self.generate_jump_diffusion_returns(bars_per_session, dt, params, rng)
            
            # Baseline volume for the day with lognormal variability
            day_volume_total = params.base_daily_volume * rng.lognormal(0.0, 0.35)
            
            # Intraday volume per bar with U-curve + microstructure noise
            intraday_vols = day_volume_total * u_profile * rng.lognormal(0.0, 0.25, size=bars_per_session)
            intraday_vols = np.maximum(intraday_vols, 10.0)

            # Accumulation for session VWAP
            session_cum_pv = 0.0
            session_cum_v = 0.0

            # Overnight gap shock
            overnight_gap = rng.normal(0.0, params.annual_volatility * np.sqrt(1.0 / 252.0) * 0.4)
            current_price *= np.exp(overnight_gap)
            current_price = max(current_price, 0.50)

            # Generate intraday timestamps
            session_start_dt = day.replace(
                hour=session_start_utc.hour,
                minute=session_start_utc.minute,
                second=0,
                microsecond=0
            )

            for b in range(bars_per_session):
                bar_time = session_start_dt + pd.Timedelta(minutes=b * bar_minutes)
                r_bar = day_returns[b]
                
                open_b = current_price
                close_b = open_b * np.exp(r_bar)
                close_b = max(close_b, 0.10)

                # Intraday bar spread/range
                bar_sigma = params.annual_volatility * np.sqrt(dt)
                high_wiggle = abs(rng.normal(0.0, bar_sigma * 0.8))
                low_wiggle = abs(rng.normal(0.0, bar_sigma * 0.8))

                high_b = max(open_b, close_b) * (1.0 + high_wiggle)
                low_b = min(open_b, close_b) * (1.0 - low_wiggle)
                low_b = max(low_b, 0.05)

                vol_b = intraday_vols[b]
                bar_typical_price = (open_b + high_b + low_b + close_b) / 4.0

                session_cum_pv += bar_typical_price * vol_b
                session_cum_v += vol_b
                vwap_b = session_cum_pv / max(session_cum_v, 1e-6)

                trade_count_b = max(1, int(vol_b / rng.uniform(50, 250)))

                all_timestamps.append(bar_time)
                all_opens.append(round(open_b, 2))
                all_highs.append(round(high_b, 2))
                all_lows.append(round(low_b, 2))
                all_closes.append(round(close_b, 2))
                all_volumes.append(round(vol_b, 0))
                all_vwaps.append(round(vwap_b, 2))
                all_trade_counts.append(trade_count_b)

                current_price = close_b

        df = pd.DataFrame({
            "open": all_opens,
            "high": all_highs,
            "low": all_lows,
            "close": all_closes,
            "volume": all_volumes,
            "vwap": all_vwaps,
            "trade_count": all_trade_counts,
            "market": params.market,
            "currency": params.currency
        }, index=pd.DatetimeIndex(all_timestamps, tz="UTC", name="timestamp"))

        # Slice to requested exact date bounds if provided
        if start_date is not None:
            df = df[df.index >= pd.to_datetime(start_date, utc=True)]
        if end_date is not None:
            df = df[df.index <= pd.to_datetime(end_date, utc=True)]

        return df
