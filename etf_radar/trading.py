"""China ETF execution costs and trade sizing helpers."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class TradingCostModel:
    """Conservative A-share ETF cost model.

    ETFs currently have no stamp duty.  Broker commission, exchange handling,
    half-spread, slippage and square-root market impact are all charged explicitly.
    The default broker tariff is 1.5 bps with no minimum commission.
    """

    commission_rate: float = 0.00015
    minimum_commission: float = 0.0
    exchange_handling_rate: float = 0.00004
    transfer_fee_rate: float = 0.0
    stamp_duty_sell_rate: float = 0.0
    bid_ask_half_spread_bps: float = 2.0
    base_slippage_bps: float = 3.0
    impact_bps_at_full_adv: float = 18.0
    max_participation_rate: float = 0.10
    lot_size: int = 100

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def estimate(
        self,
        side: str,
        price: float,
        shares: float,
        average_daily_amount: Optional[float] = None,
    ) -> Dict[str, float]:
        side_value = str(side).upper()
        if side_value not in {"BUY", "SELL"}:
            raise ValueError("side must be BUY or SELL")
        price_value = max(0.0, float(price))
        shares_value = max(0.0, float(shares))
        gross = price_value * shares_value
        if gross <= 0:
            return {
                "gross": 0.0,
                "fees": 0.0,
                "slippage": 0.0,
                "total_cost": 0.0,
                "cash_delta": 0.0,
                "effective_price": 0.0,
                "participation_rate": 0.0,
                "requested_participation_rate": 0.0,
                "capacity_exceeded": False,
                "impact_bps": 0.0,
            }
        commission = max(float(self.minimum_commission), gross * float(self.commission_rate))
        regulatory = gross * (float(self.exchange_handling_rate) + float(self.transfer_fee_rate))
        stamp = gross * float(self.stamp_duty_sell_rate) if side_value == "SELL" else 0.0
        adv = max(float(average_daily_amount or 0.0), gross)
        requested_participation = gross / adv
        participation = min(requested_participation, max(0.0, float(self.max_participation_rate)))
        impact_bps = float(self.impact_bps_at_full_adv) * math.sqrt(max(participation, 0.0))
        slippage_rate = (
            float(self.bid_ask_half_spread_bps)
            + float(self.base_slippage_bps)
            + impact_bps
        ) / 10000.0
        slippage = gross * slippage_rate
        fees = commission + regulatory + stamp
        total_cost = fees + slippage
        cash_delta = -(gross + total_cost) if side_value == "BUY" else gross - total_cost
        effective_price = (
            (gross + total_cost) / shares_value
            if side_value == "BUY"
            else (gross - total_cost) / shares_value
        )
        return {
            "gross": gross,
            "fees": fees,
            "slippage": slippage,
            "total_cost": total_cost,
            "cash_delta": cash_delta,
            "effective_price": effective_price,
            "participation_rate": participation,
            "requested_participation_rate": requested_participation,
            "capacity_exceeded": requested_participation > float(self.max_participation_rate) + 1e-12,
            "impact_bps": impact_bps,
        }

    def round_lot(self, shares: float) -> int:
        lot = max(1, int(self.lot_size))
        return max(0, int(float(shares) // lot) * lot)

    def capacity_lot(self, price: float, average_daily_amount: Optional[float]) -> int:
        price_value = max(float(price), 0.0)
        adv = max(float(average_daily_amount or 0.0), 0.0)
        if price_value <= 0.0 or adv <= 0.0:
            return 0
        return self.round_lot(adv * max(float(self.max_participation_rate), 0.0) / price_value)


DEFAULT_ETF_COST_MODEL = TradingCostModel()


__all__ = ["DEFAULT_ETF_COST_MODEL", "TradingCostModel"]
