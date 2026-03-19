---
name: indicator_analysis
description: Audits indicators, removes contradictions, and simplifies signals to a small coherent set
---

## When to Use

- When trade accuracy is low due to contradictory or noisy indicators
- When simplifying strategies to 3–4 core signals (EMA, RSI, MACD, optional Bollinger Bands)
- When you need comparable confidence scoring across modules

## Instructions

1. Identify active indicators and hard gates in:
   - `modules/momentum.py`
   - `modules/mean_reversion.py`
   - `modules/ai_predictor.py`
2. Reduce decision inputs to a coherent set:
   - Trend: EMA spread (fast vs slow)
   - Momentum: MACD histogram
   - Mean reversion: Bollinger Bands + RSI (only when trend is weak)
   - Volatility: ATR (risk sizing / SL/TP)
   - **Deep Learning**: GRU model in `prediction/predict.py` consumes raw OHLCV sequences
3. Ensure indicators are computed consistently using shared utilities:
   - `utils/indicators.py`
4. Preserve module output contract:
   - `{"signal": "BUY|SELL|HOLD", "confidence": float, "metadata": dict}`

