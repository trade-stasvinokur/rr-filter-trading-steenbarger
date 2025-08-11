# R:R (Risk/Reward) filter — reusable template
# ------------------------------------------------
# This file contains:
# 1) Core functions to compute risk/reward (R:R) for trades
# 2) Utilities to derive targets from a desired R:R
# 3) Position sizing based on fixed % risk per trade
# 4) Vectorized helpers for Pandas DataFrames
#
# You can import these utilities into any strategy/backtest.
# Save path: /mnt/data/rr_filter_template.py

from dataclasses import dataclass
from typing import Optional, Literal, Tuple
import math
import pandas as pd

Direction = Literal["long", "short"]


@dataclass
class TradeIdea:
    symbol: str
    direction: Direction
    entry: float
    stop: float
    target: Optional[float] = None
    # Optional ATR-based logic (if you'd rather define stop by ATR)
    atr: Optional[float] = None
    atr_mult: Optional[float] = None  # e.g., 1.5 * ATR for stop

    def resolve_stop(self) -> float:
        if self.atr is not None and self.atr_mult is not None:
            if self.direction == "long":
                return self.entry - self.atr * self.atr_mult
            else:
                return self.entry + self.atr * self.atr_mult
        return self.stop

    def risk_reward(self) -> Optional[float]:
        stop = self.resolve_stop()
        if self.target is None:
            return None
        if self.direction == "long":
            risk = self.entry - stop
            reward = self.target - self.entry
        else:
            risk = stop - self.entry
            reward = self.entry - self.target
        if risk <= 0:
            return None
        return reward / risk

    def target_for_rr(self, desired_rr: float) -> float:
        stop = self.resolve_stop()
        if self.direction == "long":
            risk = self.entry - stop
            return self.entry + desired_rr * risk
        else:
            risk = stop - self.entry
            return self.entry - desired_rr * risk


def position_size_by_risk(
    equity: float,
    risk_pct: float,
    entry: float,
    stop: float,
    direction: Direction,
    contract_multiplier: float = 1.0,
    min_size: float = 1.0
) -> Tuple[float, float]:
    """
    Returns (position_size, $risk_per_unit).
    For spot/CFD shares set contract_multiplier=1.
    For futures, set to contract dollar value per 1 point move.
    """
    if direction == "long":
        per_unit_risk = (entry - stop) * contract_multiplier
    else:
        per_unit_risk = (stop - entry) * contract_multiplier
    if per_unit_risk <= 0:
        return 0.0, 0.0
    total_risk_allowed = equity * risk_pct
    size = math.floor(total_risk_allowed / per_unit_risk)
    size = max(size, 0.0)
    if 0 < size < min_size:
        size = 0.0  # not worth taking
    return float(size), float(per_unit_risk)


def compute_rr(entry: float, stop: float, target: float, direction: Direction) -> Optional[float]:
    if direction == "long":
        risk = entry - stop
        reward = target - entry
    else:
        risk = stop - entry
        reward = entry - target
    if risk <= 0:
        return None
    return reward / risk


def target_from_rr(entry: float, stop: float, desired_rr: float, direction: Direction) -> float:
    if direction == "long":
        risk = entry - stop
        return entry + desired_rr * risk
    else:
        risk = stop - entry
        return entry - desired_rr * risk


def filter_trades_by_rr(df: pd.DataFrame,
                        entry_col: str = "entry",
                        stop_col: str = "stop",
                        target_col: str = "target",
                        direction_col: str = "direction",
                        min_rr: float = 2.0) -> pd.DataFrame:
    """
    Vectorized RR filter for a DataFrame with columns:
    ['symbol', 'direction', 'entry', 'stop', 'target']
    Returns a copy with computed rr and a boolean pass flag.
    """
    df = df.copy()

    def safe_rr(row):
        try:
            rr = compute_rr(row[entry_col], row[stop_col], row[target_col], row[direction_col])
            return rr if rr is not None and math.isfinite(rr) else float("nan")
        except Exception:
            return float("nan")

    df["rr"] = df.apply(safe_rr, axis=1)
    df["pass_rr"] = df["rr"] >= min_rr
    return df