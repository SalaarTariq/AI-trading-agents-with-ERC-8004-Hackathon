# Risk Management Guidelines

## Per-Trade Limits

- **Stop-loss**: 3–5% per trade (configurable, default 4%)
- **Take-profit**: 5–7% per trade (configurable, default 6%)
- **Max position size**: 30% of total portfolio value
- **Minimum confidence threshold**: 60% combined signal confidence to enter a trade

## Portfolio-Level Limits

- **Daily loss cap**: 10% of portfolio value — halt all trading if breached
- **Maximum concurrent positions**: 5 open positions at any time
- **Maximum correlation exposure**: No more than 50% of portfolio in correlated assets

## Volatility Filters

- Skip trades if current volatility exceeds the defined threshold (default: 2x the 30-day average)
- Use ATR (Average True Range) or standard deviation of returns as volatility measure
- During high volatility, reduce position sizes by 50%

## Validation Requirements

- **All trades must be validated** against every rule above before execution
- The risk manager returns a structured result: `{approved: bool, reasons: list, adjusted_size: float}`
- If a trade is rejected, the reason must be logged and included in the proof hash
- Risk parameters must be configurable via `utils/config.py` — never hardcoded in strategy modules

## Emergency Rules

- If 3 consecutive trades hit stop-loss, pause trading for the remainder of the session
- If portfolio drawdown exceeds 15% from peak, enter "defensive mode" (reduce all position sizes by 75%)
- All emergency triggers must be logged with timestamps
