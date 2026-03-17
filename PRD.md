
# **Product Requirement Document (PRD)**

**Project Name:** Balanced Hybrid AI Trading Agent
**Hackathon:** AI Trading Agents with ERC-8004 – lablab.ai / Surge
**Pre-Hackathon Phase:** March 2026

---

## **1. Project Overview**

This project is a **trustworthy, autonomous AI trading agent** designed for the ERC-8004 hackathon.

The agent will:

* Trade crypto in a **balanced, risk-managed way**
* Combine **rule-based trading logic** with **AI reasoning** for improved decision-making
* Simulate proof and validation **pre-hackathon** to mimic ERC-8004 registries
* Be modular, Python-based, and testable in VS Code
* Generate dashboards to track performance, trades, and risk

**Goal:** Build a tested, deployable agent before hackathon so integration with Surge sandbox and ERC-8004 is quick and smooth.

---

## **2. Key Features**

| Feature                   | Description                                                                    | Pre-Hackathon Status |
| ------------------------- | ------------------------------------------------------------------------------ | -------------------- |
| Hybrid Trading Logic      | Momentum + Mean-Reversion + Yield Module                                       | Fully implemented    |
| AI Prediction             | ML/LLM predicts trend confidence and adjusts decision                          | Fully implemented    |
| Rule-Based Logic          | Moving averages, RSI, support/resistance, liquidity filters                    | Fully implemented    |
| Risk Management           | Stop-loss, take-profit, daily loss limit, max position size, volatility filter | Fully implemented    |
| Trade Simulation          | Paper trading using historical price data                                      | Fully implemented    |
| Dashboard                 | Shows portfolio value, PnL, trade history, validation logs, risk metrics       | Fully implemented    |
| Validation Simulation     | Hash-based proof of decisions for ERC-8004 compatibility                       | Fully implemented    |
| Multi-Module Coordination | Central agent coordinates all sub-modules for final trading decision           | Fully implemented    |

---

## **3. Trading Strategy**

### 3.1 Momentum Module

* Detects trends using **moving averages** and **volume filters**
* Buys in rising trends, sells in declining trends

### 3.2 Mean-Reversion Module

* Trades when prices are far from historical averages
* Buys undervalued assets, sells overvalued

### 3.3 Yield Optimization Module

* Optional for pre-hackathon: monitors liquidity pools / staking opportunities
* Allocates funds to maximize passive returns

### 3.4 AI Reasoning

* Evaluates all module outputs and predicts likelihood of success
* Confirms or adjusts rule-based recommendations

### 3.5 Risk Rules

* Stop-loss: 3–5%
* Take-profit: 5–7%
* Max position size: 30% of portfolio
* Daily loss cap: 10%
* Volatility filter: avoid high-risk tokens

---

## **4. Architecture Overview**

```
                 +----------------+
                 | Market Data API|
                 +--------+-------+
                          |
           +--------------v--------------+
           |      Data Preprocessing     |
           | - Calculate indicators      |
           | - Normalize data            |
           +--------------+--------------+
                          |
         +----------------v----------------+
         |          Hybrid Agent           |
         | - Momentum Module               |
         | - Mean-Reversion Module         |
         | - Yield Module                  |
         | - AI Prediction Layer           |
         | - Risk Management Module        |
         +----------------+----------------+
                          |
         +----------------v----------------+
         | Trade Simulation / Paper Trading|
         | - Execute trades on virtual funds|
         | - Record trade logs              |
         +----------------+----------------+
                          |
                 +--------v--------+
                 |  Validation Logs|
                 | - Hash-based proof|
                 +--------+--------+
                          |
                 +--------v--------+
                 | Dashboard / GUI |
                 | - PnL / Portfolio|
                 | - Risk Metrics  |
                 | - Trade Signals |
                 +----------------+
```

---

## **5. Technology Stack**

| Component          | Technology                                              |
| ------------------ | ------------------------------------------------------- |
| Programming        | Python                                                  |
| IDE                | VS Code                                                 |
| AI Layer           | ML/LLM (Claude-assisted coding)                         |
| Data Sources       | Historical crypto prices (CoinGecko, Aerodrome testnet) |
| Dashboard          | Streamlit / Flask                                       |
| Validation         | Hash-based simulation (pre-hackathon)                   |
| Trading Simulation | Paper trading engine                                    |

---

## **6. Project Scope Pre-Hackathon**

**Modules to implement now:**

1. Market data fetching & preprocessing
2. Momentum, mean-reversion, yield modules
3. Hybrid AI + rule-based decision-making
4. Risk management rules & enforcement
5. Trade simulation engine
6. Validation hash logging
7. Dashboard for monitoring & reporting

---

## **7. Post-Hackathon Integration Plan**

1. Register agent with ERC-8004 Identity Registry
2. Connect to Surge Sandbox / Capital Vault
3. Implement on-chain validation registry integration
4. Deploy live trading using Risk Router
5. Monitor leaderboard, update reputation scores

---

## **8. Success Criteria (Pre-Hackathon)**

* Agent executes trades in paper trading simulation without errors
* Risk rules enforced correctly
* Dashboard shows correct trade logs, PnL, and risk metrics
* Validation logs record decisions correctly
* Hybrid AI + rule-based module functions cohesively

---

## **9. Notes / Recommendations**

* Focus on **robustness and modularity** pre-hackathon
* Keep code structured for **easy blockchain integration** later
* Paper trading + historical backtesting ensures better leaderboard performance
* Claude will be used for **full project generation** including AI logic, modules, simulation, and dashboard

