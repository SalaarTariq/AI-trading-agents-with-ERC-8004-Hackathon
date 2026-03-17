---
name: create-trading-strategies
description: Generates trading strategy modules (momentum, mean-reversion, yield)
---

## When to Use

- When asked to create logic for trade decisions
- When implementing or modifying individual trading strategies
- When adding new strategy modules to the agent

## Instructions

1. Generate strategy files: `momentum.py`, `mean_reversion.py`, `yield_optimizer.py`
2. Each module must return a standardized signal format:
   ```python
   {
       "signal": "BUY" | "SELL" | "HOLD",
       "confidence": 0.0 to 1.0,
       "metadata": {
           "indicators": {...},
           "reasoning": "..."
       }
   }
   ```
3. Ensure the AI predictor module (`ai_predictor.py`) can read outputs from all strategies and make weighted decisions
4. Include docstrings and inline comments explaining the trading logic
5. Use technical indicators from `utils/indicators.py`
6. All parameters must come from configuration — never hardcode thresholds

## Strategy Details

### Momentum (`momentum.py`)
- **Core Logic**: Moving average crossover (fast MA crosses above slow MA = BUY)
- **Confirmation**: Volume must be above average to confirm trend
- **Indicators**: SMA/EMA (short + long period), volume moving average
- **Exit Signal**: Fast MA crosses below slow MA, or trailing stop triggered

### Mean Reversion (`mean_reversion.py`)
- **Core Logic**: Price deviation from mean using Bollinger Bands and z-score
- **Entry**: Buy when price drops below lower Bollinger Band (z-score < -2)
- **Exit**: Sell when price returns to mean or exceeds upper band
- **Indicators**: Bollinger Bands, z-score, RSI for oversold/overbought confirmation

### Yield Optimizer (`yield_optimizer.py`)
- **Core Logic**: Evaluate yield/APY opportunities across pools
- **Criteria**: Risk-adjusted returns, liquidity depth, protocol safety
- **Output**: Allocation suggestions with expected yield
- **Constraints**: Max allocation per pool, minimum liquidity requirements

### AI Predictor (`ai_predictor.py`)
- **Core Logic**: Weighted ensemble of all strategy signals
- **Weighting**: Based on recent strategy accuracy and market regime
- **Output**: Final prediction with aggregated confidence
- **Explainability**: Log which strategies contributed most to the decision

## Example

**Input**: "Create momentum strategy for ETH/USDC"

**Output**: `momentum.py` with:
- Configurable moving average periods (default: fast=12, slow=26)
- Volume confirmation filter
- Signal generation returning standardized format
- Confidence score based on crossover strength and volume
