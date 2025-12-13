# Quantitative AI Volume Forecasting System — Final Report (Week 8)

## Executive Summary
This project completes a quantitative AI forecasting system by combining statistical time-series methods (Week 7) with machine learning models (Week 8). The system generates a 13-month volume forecast using engineered numerical features, ensemble weighting, and probabilistic confidence intervals. Deliverables include an automated Python pipeline, a professional Excel workbook, and a visualization suite that quantifies both forecast values and uncertainty.

## Quantitative System Architecture

### Statistical Models (Week 7)
1. **Moving Average (MA-7)**
   - Formula:  \u0177_t = (1/n) Σ x_i
   - Weight: 0.15  
   - Notes: Smooths short-term noise and captures local patterns.

2. **Exponential Smoothing (α=0.3)**
   - Formula:  \u0177_t = αx_t + (1-α)\u0177_(t-1)
   - Weight: 0.15  
   - Notes: Emphasizes recent observations via geometric decay.

### Machine Learning Models (Week 8)
3. **Linear Regression**
   - Model: y = β0 + β1x1 + … + βnxn + ε
   - Weight: 0.20  
   - Feature engineering: lags (1..30), rolling stats (7/30), calendar encodings, sine/cosine seasonality terms.

4. **Random Forest (Ensemble)**
   - Trees: 200
   - Weight: 0.25  
   - Notes: Captures nonlinear interactions; feature importance estimated via impurity reduction.

5. **XGBoost (Gradient Boosting)**
   - Estimators: 300, depth: 6, lr: 0.08
   - Weight: 0.25  
   - Notes: Sequential boosting improves fit on complex patterns (if xgboost installed).

## Statistical Pattern Discovery (Week 7 Findings)
- **Trend:** The dataset includes a clear upward linear trend over the 3-year window.
- **Seasonality:** Yearly sinusoidal component creates a 365-day repeating pattern.
- **Weekly effect:** Weekdays tend to be higher than weekends (encoded in the generator).
- **Monthly spikes:** End-of-month step function increases volume near day 25+.

## Mathematical Feature Engineering (Week 8)
Top feature families used:
1. **Autoregressive lags:** lag_1 … lag_30  
2. **Rolling statistics:** mean/std/min/max over 7 and 30 days  
3. **Time encodings:** day_of_week, month, quarter, time_index  
4. **Fourier-like seasonality:** day_sin/day_cos  
5. **Transformations:** log1p(volume), sqrt(volume), volume^2  

## Quantitative Forecast Results (13 Months)
After running the system:
- **Total Volume Forecast:** [fill from console summary]  
- **Average Monthly Volume:** [fill]  
- **Peak Month:** [fill]  
- **Peak Volume:** [fill]  

## Probabilistic Uncertainty (Confidence Intervals)
Confidence intervals were computed using the **standard deviation across model forecasts**:
- Lower = ensemble − 1.96 * σ
- Upper = ensemble + 1.96 * σ
This produces a numeric 95% confidence band that widens when models disagree and narrows when predictions align.

## Validation and Performance Metrics (ML)
Copy these from your run output / Excel sheet:
| Model | MAE | RMSE | R² |
|------|-----|------|----|
| Linear Regression | [val] | [val] | [val] |
| Random Forest | [val] | [val] | [val] |
| XGBoost (if used) | [val] | [val] | [val] |

## Challenges Encountered
1. **Time-series split (no shuffle):** Preserving temporal order is required for valid forecasting evaluation.  
2. **Feature lag NaNs:** Lag and rolling windows create missing values that must be dropped.  
3. **Optional XGBoost dependency:** The system supports running without it, but can include it if installed.

## Business Value of Quantitative AI
Quantitative forecasting supports:
- inventory planning (reducing overstock/stockouts)
- staffing and resource allocation
- performance tracking via error metrics and confidence intervals
- explainability via feature importance and model comparisons

## Conclusion
This project demonstrates a complete quantitative AI forecasting workflow: generating numerical data, identifying statistical patterns, applying multiple forecasting models, combining them with ensemble weighting, and expressing uncertainty through confidence intervals. The final system produces a professional forecast package with a reusable pipeline and clear quantitative documentation.

**Quantitative AI System Developer:** [Sayan Ouk]  
**Date:** December 13, 2025  
**Course:** CSC1130 — Introduction to AI (Quantitative Methods)