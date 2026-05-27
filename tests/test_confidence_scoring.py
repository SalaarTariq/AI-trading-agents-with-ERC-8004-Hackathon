"""Tests for modules.confidence_scoring.combine_signals."""

from __future__ import annotations

import pytest

from config import RegimeConfig, RegimeParams, SignalConfig
from modules.confidence_scoring import combine_signals, effective_confidence_threshold


def _trending_up_with(
    momentum: dict,
    mean_reversion: dict | None = None,
    atr_percentile_rank: float = 0.50,
) -> dict:
    return {
        "regime": "trending_up",
        "momentum": momentum,
        "mean_reversion": mean_reversion or {"signal": 0, "raw_strength": 0.0},
        "atr_percentile_rank": atr_percentile_rank,
    }


class TestNoSignals:
    def test_all_zero_returns_hold(self):
        out = combine_signals(
            _trending_up_with({"signal": 0, "raw_strength": 0.0}),
            current_atr_norm=0.01,
        )
        assert out["action"] == "HOLD"
        assert out["confidence"] == 0.0


class TestAgreementBonus:
    def test_rule_agreement_bumps_confidence(self):
        cfg = SignalConfig(
            execute_confidence_threshold=0.30,  # let it pass so we can read confidence
            strong_support_min_strength=0.20,
            agreement_bonus=0.20,
            use_ai_predictor=False,
        )
        agreed = combine_signals(
            {
                "regime": "ranging",
                "momentum": {"signal": 1, "raw_strength": 0.6},
                "mean_reversion": {"signal": 1, "raw_strength": 0.6},
                "atr_percentile_rank": 0.50,
            },
            current_atr_norm=0.01,
            cfg=cfg,
        )
        disagreed = combine_signals(
            {
                "regime": "ranging",
                "momentum": {"signal": 1, "raw_strength": 0.6},
                "mean_reversion": {"signal": -1, "raw_strength": 0.6},
                "atr_percentile_rank": 0.50,
            },
            current_atr_norm=0.01,
            cfg=cfg,
        )
        assert agreed["confidence"] > disagreed["confidence"]


class TestVolatilityPenalty:
    def test_high_atr_rank_reduces_confidence(self):
        cfg = SignalConfig(
            execute_confidence_threshold=0.30,
            high_volatility_penalty=0.30,
            use_ai_predictor=False,
        )
        calm = combine_signals(
            _trending_up_with(
                {"signal": 1, "raw_strength": 0.8},
                atr_percentile_rank=0.50,
            ),
            current_atr_norm=0.01,
            cfg=cfg,
        )
        spiky = combine_signals(
            _trending_up_with(
                {"signal": 1, "raw_strength": 0.8},
                atr_percentile_rank=0.90,
            ),
            current_atr_norm=0.01,
            cfg=cfg,
        )
        assert calm["confidence"] - spiky["confidence"] == pytest.approx(
            cfg.high_volatility_penalty, rel=1e-3
        )


class TestAIVoter:
    def test_ai_disabled_unchanged(self):
        cfg_no_ai = SignalConfig(
            execute_confidence_threshold=0.30,
            use_ai_predictor=False,
        )
        baseline = combine_signals(
            _trending_up_with({"signal": 1, "raw_strength": 0.7}),
            current_atr_norm=0.01,
            cfg=cfg_no_ai,
            ai_out={"signal": "BUY", "confidence": 0.9},
        )
        assert baseline["details"]["ai"]["enabled"] is False

    def test_ai_agreement_bumps_confidence(self):
        cfg = SignalConfig(
            execute_confidence_threshold=0.30,
            strong_support_min_strength=0.20,
            use_ai_predictor=True,
            ai_predictor_weight=0.20,
            ai_agreement_bonus=0.10,
        )
        with_ai = combine_signals(
            _trending_up_with({"signal": 1, "raw_strength": 0.7}),
            current_atr_norm=0.01,
            cfg=cfg,
            ai_out={"signal": "BUY", "confidence": 0.9},
        )
        without_ai = combine_signals(
            _trending_up_with({"signal": 1, "raw_strength": 0.7}),
            current_atr_norm=0.01,
            cfg=cfg,
            ai_out={"signal": "HOLD", "confidence": 0.0},
        )
        assert with_ai["confidence"] > without_ai["confidence"]

    def test_ai_cannot_outvote_two_agreeing_rules(self):
        cfg = SignalConfig(
            execute_confidence_threshold=0.30,
            strong_support_min_strength=0.20,
            use_ai_predictor=True,
            ai_predictor_weight=0.20,
        )
        # Two rules say BUY with full strength; AI disagrees with full strength.
        out = combine_signals(
            {
                "regime": "ranging",
                "momentum": {"signal": 1, "raw_strength": 1.0},
                "mean_reversion": {"signal": 1, "raw_strength": 1.0},
                "atr_percentile_rank": 0.50,
            },
            current_atr_norm=0.01,
            cfg=cfg,
            ai_out={"signal": "SELL", "confidence": 1.0},
        )
        assert out["action"] == "BUY"


class TestThresholdHelper:
    def test_high_vol_uplifts_threshold(self):
        cfg = SignalConfig(execute_confidence_threshold=0.60)
        low = effective_confidence_threshold(base=0.60, current_atr_norm=0.01, signal_cfg=cfg)
        high = effective_confidence_threshold(base=0.60, current_atr_norm=0.05, signal_cfg=cfg)
        assert high > low
