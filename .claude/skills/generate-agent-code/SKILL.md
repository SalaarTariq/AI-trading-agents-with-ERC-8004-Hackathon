---
name: generate-agent-code
description: Generates Python modules for the hybrid AI trading agent
---

## When to Use

- When Claude is asked to generate trading logic, risk management, or dashboard code
- When the user requests the full agent codebase or specific modules
- When scaffolding new components that fit into the agent architecture

## Instructions

1. Create Python module files under the appropriate directories: `modules/`, `risk/`, `dashboard/`, `validation/`, `simulation/`, `utils/`
2. Include comments and Google-style docstrings for all public functions and classes
3. Implement hybrid logic: combine momentum, mean-reversion, and AI predictor signals
4. Integrate `risk_manager` checks before any trade execution
5. Ensure `proof_logger` hashes all trade inputs, outputs, and decisions
6. Create `main.py` to tie everything together as the entry point
7. Use configuration from `config/config.yaml` or `utils/config.py` — never hardcode parameters
8. Include proper logging using Python's `logging` module
9. Use type hints for all function signatures
10. Ensure each module can be imported and tested independently

## Module Checklist

When generating the full agent, ensure these files exist:

- [ ] `main.py` — orchestration entry point
- [ ] `modules/momentum.py` — momentum trading strategy
- [ ] `modules/mean_reversion.py` — mean-reversion trading strategy
- [ ] `modules/yield_optimizer.py` — yield optimization strategy
- [ ] `modules/ai_predictor.py` — AI prediction ensemble
- [ ] `modules/strategy_manager.py` — combines all strategy signals
- [ ] `risk/risk_manager.py` — risk validation and trade gating
- [ ] `simulation/paper_trader.py` — virtual trade execution
- [ ] `validation/proof_logger.py` — SHA256 proof hash logging
- [ ] `dashboard/dashboard.py` — Streamlit/Flask visualization
- [ ] `utils/config.py` — centralized configuration
- [ ] `utils/data_loader.py` — data ingestion
- [ ] `utils/indicators.py` — technical indicator calculations
- [ ] `utils/logger.py` — logging setup

## Example

**Input**: "Generate the full trading agent with all modules"

**Output**: Complete Python module files for every component listed above, each with:
- Proper imports and type hints
- Docstrings explaining the module's purpose
- Core logic implementation
- Logging integration
- Risk management integration (where applicable)
- Proof logging integration (where applicable)

**Input**: "Generate the momentum strategy module"

**Output**: `modules/momentum.py` with:
- Moving average crossover logic
- Volume confirmation filters
- Signal generation with confidence score
- Configurable parameters from config
