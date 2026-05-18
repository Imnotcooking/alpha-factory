### Oxford Quant Pipeline: Automated Alpha Factory
An end-to-end quantitative research and live execution pipeline built for systematic factor investing.

## 🏗 Architecture Overview
This infrastructure is divided into three distinct environments to ensure strict separation of concerns between research, evaluation, and execution.

Alpha Research Lab (Local): Polars-based engine for testing market anomalies, integrating with the Financial Modeling Prep (FMP) API for point-in-time fundamental and price data.

The Evaluator (Local): A strict out-of-sample validation engine that tests factors across Validation, Holdout, and Crisis regimes, calculating Information Coefficient (IC) while enforcing liquidity constraints.

Execution Engine (AWS Cloud): A headless Ubuntu server running Dockerized Interactive Brokers (IB Gateway) and a daily CRON job to optimize portfolios via SciPy and route trades automatically.

## 🛠 Tech Stack
Data & Compute: Python, Polars, Pandas, NumPy, SciPy

Infrastructure: AWS Lightsail (Ubuntu), Docker, SQLite

Brokerage & APIs: Interactive Brokers (ib_insync), Financial Modeling Prep (FMP) API

Visualization: Streamlit

## ⚙️ Pipeline Flow
Ingestion: Real-time data lake assembly using dynamic API routing and local .parquet caching.

Signal Generation: Cross-sectional Z-scoring of accounting and price anomalies.

Validation: Strict institutional hurdling. Factors failing out-of-sample IC checks are flagged and blocked from deployment.

Execution: Approved factors are passed to the AWS execution server, where the agent calculates target portfolio weights and beams market orders to Wall Street.