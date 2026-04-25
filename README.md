# Monte Carlo Simulation for Credit Risk Assessment: A Machine Learning Approach

## Overview
This project transitions credit risk management from static, backward-looking analysis to a **dynamic, machine learning-driven simulation framework**. 

By combining historical credit data with stochastic macroeconomic modeling, the project quantifies potential portfolio losses under various future scenarios. This approach allows financial institutions to compute critical risk metrics like **Value at Risk (VaR)** and **Conditional Value at Risk (CVaR)** in real-time.

---

## Phase 1: Exploratory Data Analysis (EDA)
**Objective:** Identify the key historical drivers of default and understand the relationships between macroeconomic variables and counterparty risk.

### Key Activities:
*   **Data Cleaning:** Handled missing values and standardized formats for a portfolio focused on the Indian market. Dataset is extracted from Bloomberg.
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
## Phase 3: Machine Learning Model Comparison & Optimization

### Objective
Enhance the credit risk modeling framework by evaluating multiple machine learning models and selecting the most effective one for predicting default probabilities (PDs), which are later used in Monte Carlo simulations.

---

### Model Development
To improve upon the baseline Logistic Regression model, the following supervised learning algorithms were implemented and compared:

- Random Forest  
- XGBoost  
- Neural Network (MLP Classifier)  
- Logistic Regression (Benchmark Model)  

All models were trained on the same dataset using a consistent preprocessing pipeline to ensure a fair comparison.

---

### Evaluation Metrics
Models were evaluated using:

- **Accuracy** – Measures overall classification performance  
- **ROC-AUC Score** – Measures the ability to distinguish between default and non-default classes (critical due to class imbalance)

---

### Model Performance

| Model                | Accuracy | ROC-AUC |
|---------------------|----------|---------|
| Random Forest       | 0.9691   | 0.7683  |
| Neural Network      | 0.9613   | 0.7477  |
| XGBoost             | 0.8880   | **0.7938** |
| Logistic Regression | 0.7478   | 0.7932  |

---

### Key Insights

- **Random Forest** achieved the highest accuracy, indicating strong overall classification performance.  
- **XGBoost** delivered the highest ROC-AUC score, making it the most effective model for distinguishing defaults from non-defaults.  
- **Logistic Regression** remains a strong baseline with competitive ROC-AUC and high interpretability.  
- **Neural Network** showed high accuracy but relatively lower ROC-AUC, suggesting weaker performance in identifying default cases.

---

### Model Selection Strategy

Since credit risk modeling prioritizes detecting defaults and capturing tail risk:

- **Selected Model for Simulation:** XGBoost  
- **Benchmark Model:** Logistic Regression (for interpretability and comparison)

---

### Integration with Monte Carlo Simulation

The selected model (XGBoost) replaces the earlier linking model in Phase 2:

1. Simulated macroeconomic variables are generated using Monte Carlo methods  
2. These variables are passed into the trained ML model  
3. The model outputs dynamic Probability of Default (PD)  
4. PDs are used to compute portfolio losses across 10,000+ scenarios  

---

### Impact

Phase 3 enhances the framework by:

- Capturing **nonlinear relationships** between macroeconomic variables and defaults  
- Improving **tail-risk estimation** (VaR & CVaR)  
- Increasing sensitivity to **rare default events**  
- Producing more **realistic and robust credit loss projections**

---

### Key Takeaway

Phase 3 upgrades the system from a single-model approach to a multi-model optimized framework, significantly improving predictive performance and risk estimation accuracy.

## Tech Stack
*   **Language:** Python
*   **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn.
*   **Framework:** Scikit-Learn `Pipeline` for reproducible preprocessing and modeling.

---

