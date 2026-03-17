---
name: write-unit-tests
description: Generates unit tests for all trading agent modules
---

## When to Use

- When asked to generate tests for the trading agent
- When a new module is created and needs test coverage
- When verifying correctness of existing modules

## Instructions

1. Use **pytest** as the test framework
2. Create test files in `tests/` directory, mirroring the module structure:
   - `tests/test_momentum.py` — tests for momentum strategy
   - `tests/test_mean_reversion.py` — tests for mean-reversion strategy
   - `tests/test_yield_optimizer.py` — tests for yield optimizer
   - `tests/test_risk_manager.py` — tests for risk management (critical)
   - `tests/test_proof_logger.py` — tests for validation/proof hashing
   - `tests/test_ai_predictor.py` — tests for AI ensemble predictor
   - `tests/test_strategy_manager.py` — tests for strategy combination logic
   - `tests/test_paper_trader.py` — tests for trade execution simulation
3. Include **edge case tests** for the risk manager (boundary values, emergency triggers)
4. Ensure proof logger hashing is validated for **determinism** and **correctness**
5. Use pytest fixtures for shared test data (historical prices, portfolio state)
6. Tests must run without network access or API keys

## Test Categories

### Strategy Module Tests
- Valid signal format: `{signal, confidence, metadata}`
- Confidence always in [0.0, 1.0]
- Correct signal direction with known input data
- Graceful handling of edge cases (empty data, single point, flat prices)

### Risk Manager Tests (High Priority)
- Stop-loss triggers at exact boundary (3%, 5%)
- Take-profit triggers at exact boundary (5%, 7%)
- Position size rejection at 30% portfolio threshold
- Daily loss cap halts trading at 10%
- Volatility filter blocks high-volatility trades
- Consecutive stop-loss pause (3 in a row)
- Defensive mode at 15% drawdown
- Rejected trades include proper rejection reasons

### Proof Logger Tests (High Priority)
- Identical inputs produce identical hashes (deterministic)
- Different inputs produce different hashes
- All required fields present in proof record
- Hash is valid SHA256 (64 hex characters)
- Proof log file is append-only and parseable

### Integration Tests
- Full pipeline: data → signals → risk check → trade → proof
- Risk manager correctly blocks trades that violate limits
- Dashboard can read and display trade data

## Example

**Input**: "Create unit tests for the trading agent"

**Output**: Complete pytest test files with:
- Fixtures for sample market data
- Parametrized tests for boundary conditions
- Assertions for signal format, risk rules, and hash correctness
- Clear test names describing what is being verified

## Running Tests

```bash
# Run all tests
pytest tests/ -v --tb=short

# Run with coverage
pytest tests/ --cov=modules --cov=risk --cov=validation --cov-report=term-missing

# Run specific module tests
pytest tests/test_risk_manager.py -v
```
