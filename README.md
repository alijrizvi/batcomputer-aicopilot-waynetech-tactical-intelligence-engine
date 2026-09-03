# 🦇 WayneTech Tactical Decision Engine & Batcomputer Copilot

### Portfolio Project Piece: **medium article link**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Testing: Pytest](https://img.shields.io/badge/testing-pytest--5%2F5%20passing-brightgreen.svg)](https://docs.pytest.org/)
[![Architecture: Medallion](https://img.shields.io/badge/data--architecture-medallion--gold-gold.svg)](https://databricks.com/glossary/medallion-architecture)

> **An End-to-End Decision Intelligence Platform for Emergency Medical Supply Chain Allocation, Econometric Elasticity Modeling, XGBoost Demand Forecasting, Experimental A/B Testing, Monte Carlo Risk Bounds, and LLM-Powered Tactical Copiloting.**

---

## 📌 Executive Summary

High-stakes emergency logistics across urban environments (such as Gotham District 1) face two compounding operational constraints:
1. **Price Volatility:** Dynamic procurement and unit costs directly shift consumption rates.
2. **Exogenous Threat Surges:** Localized emergency spikes create severe non-linear demand shocks that standard historical averages fail to capture.

The **WayneTech Tactical Decision Engine** replaces static inventory heuristics with an integrated, production-grade **Decision Intelligence Engine**. Built using strict Test-Driven Development (TDD) via the modern **Pocock AI Workflow**, the system unifies microeconomic demand elasticity, XGBoost time-series forecasting, multi-arm statistical hypothesis testing, 1,000-run Monte Carlo stockout risk bounds, and a context-grounded AI decision copilot into an executive Streamlit dashboard.

---

## 🏗️ System Architecture & Vertical Slices

The engine is engineered around four decoupled, deep-module vertical slices:

```text
               WAYNETECH DECISION INTELLIGENCE ARCHITECTURE
               
  [ Raw Sector Logs ] ──► [ Medallion Ingestion & Feature Store ]
                                         │
        ┌────────────────────────────────┼────────────────────────────────┐
        ▼                                ▼                                ▼
[ Slice 1: Econometric ]       [ Slice 2: Forecasting ]        [ Slice 3: Hypothesis ]
• Log-Log OLS Elasticity       • ARIMA(1,1,1) Baseline         • Welch's t-Test (Lift)
• Constant Price Elasticity    • XGBoost Regressor             • Cohen's d Effect Size
• β1 Demand Sensitivity        • 5-Fold Expanding Window CV     • 1,000 Monte Carlo Runs
        │                                │                                │
        └────────────────────────────────┼────────────────────────────────┘
                                         ▼
                 [ Slice 4: Statistical Anomaly & Copilot ]
                 • Z-Score Thresholding (Z > 2.5σ Cutoff)
                 • Hybrid OpenAI API + Deterministic Fallback
                 • Structured Executive Brief Synthesis
