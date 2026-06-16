# Supply Chain Demand Forecasting & Inventory Simulation

An end-to-end supply chain simulation built in Python using real Amazon India e-commerce sales data. This project covers demand analysis, forecasting model development, inventory optimization, and cost simulation — packaged into an interactive Streamlit dashboard.

---

## Business Problem

How much inventory should an Amazon seller hold for a fashion category product? Too much = capital tied up in overstock. Too little = stockouts and lost revenue. The answer depends on how accurately you can forecast demand.

This project quantifies the business impact of better demand forecasting on inventory decisions and total supply chain costs.

---

## Dataset

- Source: [Amazon Sales Report — Kaggle](https://www.kaggle.com/datasets/thedevastator/unlock-profits-with-e-commerce-sales-data)
- Scope: Amazon India fashion category sales, April–June 2022
- Size: 128,975 orders across 9 product categories
- Focus: "Set" category (highest volume — 42,261 units)

> Note: Dataset not included in this repo due to size. Download from Kaggle and place `Amazon Sale Report.csv` in the project root.

---

## Project Structure
---

## Methodology

### 1. Exploratory Data Analysis
- Cleaned 128,975 raw orders by removing cancellations, returns, and rejections
- Identified two distinct demand regimes: pre and post Eid al-Fitr (May 2-3, 2022)
- Confirmed the level shift was platform-wide (not category-specific) and driven by fewer orders, not smaller basket sizes
- Promotion analysis showed promo rate only dropped from 79% to 73% — ruling out promotions as the primary cause

### 2. Demand Forecasting
Built two models and evaluated on an 18-day held-out test set (80/20 train/test split):

| Model | MAPE | 
|-------|------|
| Baseline (7-day rolling average) | 17.33% |
| Prophet (with Eid changepoint) | 8.73% |
| **Improvement** | **50% reduction in forecast error** |

Prophet was configured with:
- Manual changepoint at May 10, 2022 (post-Eid demand normalization)
- Weekly seasonality enabled
- `changepoint_prior_scale=0.5` for flexibility around the regime change

### 3. Inventory Optimization
Calculated safety stock, reorder point, and EOQ under two approaches:

| Parameter | Naive | Prophet | Reduction |
|-----------|-------|---------|-----------|
| Safety Stock | 456 units | 335 units | 26.5% |
| Reorder Point | 3,885 units | 3,765 units | 3.1% |
| Capital Freed | — | — | INR 36,209 |

Key insight: Prophet-based safety stock uses std dev of forecast error (76.8 units) rather than std dev of raw demand (104.5 units) — a smaller buffer is needed because the model already explains most demand variation.

### 4. Cost Simulation
Day-by-day inventory simulation over the 18-day test period:

| Metric | Naive | Prophet |
|--------|-------|---------|
| Service Level | 100% | 100% |
| Total Cost (INR) | 73,118 | 68,772 |
| **Cost Saved** | — | **INR 4,346** |
| Annualized per SKU | — | **~INR 88,000** |

---

## Key Findings

1. **Festival-driven demand spikes require changepoint modeling** — a naive rolling average cannot capture structural demand shifts, leading to systematic overforecasting post-event
2. **Better forecasts directly reduce safety stock** — 50% lower forecast error translated to 26.5% less safety stock, freeing INR 36,209 in working capital per SKU
3. **At scale, forecasting accuracy has significant financial impact** — across a catalog of 100 SKUs, Prophet-based forecasting could free ~INR 3.6M in working capital and save ~INR 8.8M annually in holding costs

---

## How to Run

**1. Install dependencies:**
**2. Download dataset** from Kaggle and place in project root as `Amazon Sale Report.csv`

**3. Run individual scripts:**
**4. Launch dashboard:**
---

## Tech Stack

- **Python** — pandas, numpy, matplotlib
- **Prophet** — Facebook's time series forecasting library
- **Streamlit** — interactive dashboard
- **Git/GitHub** — version control

---

## Assumptions

- Lead time: 7 days
- Service level target: 95% (Z = 1.65)
- Order cost: INR 500 per order
- Holding cost: INR 2 per unit per day
- Unit cost: INR 300
- Demand follows a normal distribution (standard inventory theory assumption)

---

## Limitations & Future Work

- 91-day dataset limits long-term seasonality modeling — a full year of data would improve forecast reliability
- Simulation window (18 days) is too short to observe meaningful stockout differences between approaches — a Monte Carlo simulation over a full year would better demonstrate service level trade-offs
- Single SKU focus — extending to multi-SKU portfolio optimization would add significant complexity and realism
- Lead time assumed constant — stochastic lead time simulation would be more realistic

---

## Author

**Mayank Manish Pandey**
M.S. Engineering Management, USC Viterbi
[LinkedIn](www.linkedin.com/in/mayank-manish-pandey) | [GitHub](https://github.com/pandeymatuscedu)

