# Python Coding Standards

## Language and Version

- Use **Python 3.10+** syntax (match statements, union types with `|`, etc.)
- Target compatibility with Python 3.10–3.12

## Project Structure

Use modular design with clear separation of concerns:

```
modules/       # Trading strategy implementations
risk/          # Risk management logic
validation/    # Proof logging and verification
dashboard/     # Visualization and monitoring
utils/         # Shared configuration, data loading, helpers
tests/         # All test files
data/          # Historical and sample data
```

## Code Style

- Follow **PEP 8** formatting (use `black` formatter settings)
- Maximum line length: 100 characters
- Use **type hints** for all function signatures
- Use **docstrings** (Google style) for all public functions and classes
- Comment non-obvious logic — but prefer clear naming over excessive comments

## Naming Conventions

- Files: `snake_case.py`
- Classes: `PascalCase`
- Functions/methods: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private methods: `_leading_underscore`

## Logging

- Include **logging** for all trades and significant events
- Use Python's `logging` module (not `print` statements)
- Log levels: `DEBUG` for calculations, `INFO` for trade events, `WARNING` for risk alerts, `ERROR` for failures
- Every log entry must include a timestamp

## Dependencies

- Keep dependencies minimal and well-justified
- Core: `pandas`, `numpy` for data manipulation
- Dashboard: `streamlit` or `flask`
- Testing: `pytest`
- Hashing: use stdlib `hashlib` (no external dependency needed)

## Error Handling

- Use specific exception types, not bare `except:`
- Trading modules should never crash — catch and log errors, then skip the trade
- Risk manager failures should default to **rejecting** the trade (fail-safe)
