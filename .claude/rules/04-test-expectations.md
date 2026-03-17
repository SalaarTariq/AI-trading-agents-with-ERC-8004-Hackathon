# Testing Requirements

## General Requirements

- Every trading module must have corresponding **unit tests**
- Use **pytest** as the test framework
- Tests must be runnable without external API keys or network access
- All tests must pass before any merge or deployment

## Module-Specific Tests

### Trading Strategies (`modules/`)

- Test that each strategy returns a valid signal format: `{signal: str, confidence: float, metadata: dict}`
- Test with known historical data where expected signals are predetermined
- Test edge cases: empty data, single data point, all-same-price data
- Test that confidence scores are always in range [0.0, 1.0]

### Risk Manager (`risk/`)

- Test stop-loss enforcement at boundary values (exactly 3%, 5%)
- Test take-profit enforcement at boundary values (exactly 5%, 7%)
- Test position size rejection when exceeding 30% portfolio
- Test daily loss cap triggers correctly at 10%
- Test volatility filter blocks trades during high volatility
- Test consecutive stop-loss pause (3 in a row)
- Test defensive mode activation at 15% drawdown
- Test that rejected trades include proper reasons

### Proof Logger (`validation/`)

- Test that identical inputs always produce the same hash (deterministic)
- Test that different inputs produce different hashes
- Test that all required fields are present in the proof record
- Test hash format is valid SHA256 (64 hex characters)
- Test that proof logs can be loaded and verified after writing

### AI Predictor (`modules/ai_predictor.py`)

- Test weighted ensemble logic with known inputs
- Test that predictor handles missing strategy signals gracefully
- Test confidence calibration (output confidence reflects input agreement)

## Test Data

- Use fixture files in `tests/fixtures/` or inline test data
- Historical data samples should cover: trending up, trending down, sideways, high volatility
- Minimum 100 data points per test scenario

## Coverage Target

- Aim for **80%+ code coverage** on all modules
- Risk manager and proof logger should have **95%+ coverage** (critical paths)

## Running Tests

```bash
pytest tests/ -v --tb=short
pytest tests/ --cov=modules --cov=risk --cov=validation --cov-report=term-missing
```
