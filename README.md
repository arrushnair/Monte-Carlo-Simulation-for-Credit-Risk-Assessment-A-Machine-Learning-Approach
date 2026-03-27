# Monte Carlo Simulation for Credit Risk Assessment: A Machine Learning Approach

## Overview
This project transitions credit risk management from static, backward-looking analysis to a **dynamic, machine learning-driven simulation framework**. 

By combining historical credit data with stochastic macroeconomic modeling, the project quantifies potential portfolio losses under various future scenarios. This approach allows financial institutions to compute critical risk metrics like **Value at Risk (VaR)** and **Conditional Value at Risk (CVaR)** in real-time.

---

## Phase 1: Exploratory Data Analysis (EDA)
**Objective:** Identify the key historical drivers of default and understand the relationships between macroeconomic variables and counterparty risk.

### Key Activities:
*   **Data Cleaning:** Handled missing values and standardized formats for a portfolio focused on the Indian market.
*   **Feature Engineering:** Mapped categorical credit ratings to numerical scales and processed date-series for macroeconomic factors.
*   **Correlation Analysis:** Identified that `Volatility_Index` and `Interest_Rate (%)` are the primary systemic drivers influencing counterparty defaults.

---

## Phase 2: Monte Carlo Simulation & Risk Modeling
**Objective:** Build a forward-looking engine to simulate 10,000+ potential future outcomes and quantify "worst-case" portfolio losses.

### Linking Model Performance
Before the simulation, a **Logistic Regression** "Linking Model" was trained to bridge the gap between macro-drivers and defaults. The model was specifically tuned using `class_weight='balanced'` to ensure it remains sensitive to defaults.

**Test Set Performance:**
| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **0 (No Default)** | 1.00 | 0.78 | 0.87 | 1198 |
| **1 (Default)** | 0.00 | 0.50 | 0.01 | 2 |
| **Accuracy** | | | **0.78** | 1200 |


### Simulation Methodology:
1.  **Macroeconomic Forecasting:** Used **Geometric Brownian Motion (GBM)** to model the stochastic "walk" of Volatility and Interest Rates.
2.  **Correlation Modeling:** Applied **Cholesky Decomposition** to ensure simulated risk factors maintain their historical correlation.
3.  **Simulation Loop:** 10,000 paths were run, feeding simulated macro-factors into the Linking Model to generate dynamic PDs and aggregate total portfolio loss.

---

## Portfolio Risk Results
Based on 10,000 simulated paths for a 1-month horizon:

| Metric | Value (INR Cr) | Description |
| :--- | :--- | :--- |
| **Expected Loss (Avg)** | ~19,048 | The average loss expected in normal conditions. |
| **Value at Risk (VaR 95%)** | ~25,775 | The minimum loss in the worst 5% of scenarios. |
| **Value at Risk (VaR 99%)** | ~28,331 | The minimum loss in the extreme 1% of scenarios. |
| **Conditional VaR (CVaR 95%)** | ~27,392 | The average loss *given* that the 95% threshold is breached. |

---

## Key Insights
*   **Dynamic vs. Static:** While Phase 1 showed *who* defaulted in the past, Phase 2 quantifies *how much* we might lose in the future.
*   **Systemic Risk:** The simulation highlights that even a high-rated portfolio can suffer significant losses if systemic macro-drivers (Interest Rates/Volatility) spike simultaneously.
*   **Risk-Averse Modeling:** The Linking Model is intentionally biased toward detecting defaults, providing a conservative and realistic risk buffer.

---

## Tech Stack
*   **Language:** Python
*   **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn.
*   **Framework:** Scikit-Learn `Pipeline` for reproducible preprocessing and modeling.

---

