# Seafood Wholesale Analytics — Data-Driven Inventory Optimization

> Customer segmentation, lifetime value analysis, and supply-constrained inventory optimization for a traditional seafood wholesaler.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

This project applies data science techniques to transform a traditional Japanese seafood wholesale business into a data-driven operation. Using historical sales data (2014–2025) and [Osaka Central Wholesale Market](https://www.shijou.city.osaka.jp/) bonito supply data (2015–2024), I developed analytical models for:

1. **Customer Segmentation** — B2B/B2C classification + RFM-based sub-clustering
2. **Customer Lifetime Value (LTV) Analysis** — Identifying "Royal Customers"
3. **Inventory Optimization** — Counterfactual simulation under supply constraints

### Key Result

Under a **weekly simulation** of supply-constrained scenarios, the proposed **priority-based allocation strategy** outperforms the conventional First-Come-First-Served (FCFS) approach:

| Supply Ratio (α) | FCFS Profit (JPY) | Proposed Profit (JPY) | Improvement |
|:-:|--:|--:|:-:|
| 0.5 (severe) | 20,772,650 | 24,156,917 | **+16.29%** |
| 0.6 | 24,821,450 | 28,543,804 | **+15.00%** |
| **0.7** | **28,856,817** | **32,641,803** | **+13.12% (+¥3.78M)** |
| 0.8 | 32,924,960 | 36,562,615 | **+11.05%** |
| 0.9 | 36,994,360 | 40,410,971 | **+9.24%** |

> At α=0.7 (severe shortage), the proposed model achieves **+13.12% gross profit improvement** (approx. ¥3.78 million annually) by prioritizing high-LTV "Royal Customers" during supply shortages.

## Project Structure

```
├── data/
│   ├── sample_sales_data.csv          # Anonymized sales transactions
│   └── market_supply_data.csv         # Osaka bonito market supply (2015-2024)
├── notebooks/
│   ├── 01_data_exploration.py         # Exploratory Data Analysis
│   ├── 02_customer_clustering.py      # B2B/B2C segmentation + sub-clustering
│   ├── 03_ltv_analysis.py             # Customer Lifetime Value analysis
│   └── 04_inventory_optimization.py   # Supply-constrained allocation simulation
├── results/
│   ├── figures/                       # Auto-generated visualizations
│   └── simulation_results.csv         # Simulation output
├── requirements.txt
└── README.md
```

## Dataset

### Sales Data
Anonymized transaction records from a seafood wholesale company in Osaka, Japan.

| Column | Description |
|--------|-------------|
| `date` | Transaction date (2014–2025) |
| `customer_code` | Anonymized customer identifier |
| `customer_name` | Anonymized customer name |
| `product_code` | Anonymized product identifier |
| `product_name` | Anonymized product name |
| `quantity` | Units sold |
| `unit_price` | Selling price per unit (JPY) |
| `cost_price` | Cost price per unit (JPY) |
| `sales_amount` | Total transaction value (JPY) |
| `gross_profit` | Gross profit (JPY) |
| `product_type` | Product type code |
| `product_category` | Product category code |

### Market Supply Data
Monthly frozen bonito (skipjack tuna) arrivals at Osaka Central Wholesale Market.

| Column | Description |
|--------|-------------|
| `market_volume_kg` | Monthly arrival volume (kg) |
| `market_value_yen` | Monthly total value (JPY) |
| `market_avg_price` | Average price per kg (JPY) |

## Analysis

### 1. Exploratory Data Analysis ([01_data_exploration.py](notebooks/01_data_exploration.py))
- 79,935 valid transactions across 303 customers and 1,213 products
- Strong Pareto effect: **Top 10 customers account for 67.6% of total revenue**
- Clear seasonality aligned with bonito fishing seasons
- Sales volume loosely correlates with market supply

### 2. Customer Clustering ([02_customer_clustering.py](notebooks/02_customer_clustering.py))
- **B2B/B2C Classification**: Feature-based Logistic Regression using order patterns, volume, and category diversity
- **B2B Sub-clustering**: 3 segments via K-Means (RFM features)
- **B2C Sub-clustering**: 4 segments via K-Means (RFM features)
- Optimal K determined via Elbow Method + Silhouette Score

### 3. LTV Analysis ([03_ltv_analysis.py](notebooks/03_ltv_analysis.py))
- **Gross-profit-based LTV** calculation across full customer history
- Strong Pareto distribution: **Top 20% of customers = 94.4% of total LTV**
- **Royal Customers**: 56 customers with mean LTV of ¥19.4M, avg tenure 7.9 years
- Predictive model: Random Forest achieves **R² = 0.83** for LTV prediction from early behavior

### 4. Inventory Optimization ([04_inventory_optimization.py](notebooks/04_inventory_optimization.py))
- **Weekly granularity** simulation for precise supply shock capture
- **Counterfactual design**: 2015–2023 training, 2024 validation
- Two strategies compared:
  - **FCFS**: First-Come, First-Served (conventional)
  - **Proposed**: LTV-based priority allocation to Royal Customers
- **Result**: +13.12% profit improvement at α=0.7, functioning as "intertemporal arbitrage"

## Getting Started

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/seafood-wholesale-analytics.git
cd seafood-wholesale-analytics

# Install dependencies
pip install -r requirements.txt

# Run the analysis notebooks (from the notebooks/ directory)
cd notebooks
python 01_data_exploration.py
python 02_customer_clustering.py
python 03_ltv_analysis.py
python 04_inventory_optimization.py
```

> **Note**: Scripts use `# %%` cell markers compatible with VS Code / Jupyter interactive mode. To convert to `.ipynb`:
> ```bash
> pip install jupytext
> jupytext --to notebook notebooks/01_data_exploration.py
> ```

## Tech Stack

- **Language**: Python 3.10+
- **Data Processing**: pandas, NumPy
- **Machine Learning**: scikit-learn (K-Means, Logistic Regression, Random Forest, Gradient Boosting)
- **Visualization**: matplotlib, seaborn
- **Simulation**: Custom counterfactual allocation engine

## Context

This project was developed during a **Data Science Internship** at a seafood wholesale company in Osaka, Japan (July 2025 – Present). The company had no prior data analytics infrastructure, and this project served as the foundation for their digital transformation initiative.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

> **Disclaimer**: All data in this repository is anonymized. No actual business data is included.
