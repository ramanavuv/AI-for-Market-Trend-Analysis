# AI-for-Market-Trend-Analysis



This repository contains an end-to-end **time series analysis and forecasting pipeline** built to study and predict **Redgram market prices** using a **PyTorch-based LSTM model**.

The work goes beyond simple prediction. It focuses on understanding market behavior, capturing temporal patterns, and designing a solution that is both **research-oriented** and **production-ready**. The pipeline cleanly separates data analysis, model training, evaluation, and inference, making it suitable for real-world deployment as well as academic or interview discussions.

---

## Problem Statement

Agricultural commodity prices are highly time-dependent and influenced by seasonal and market dynamics.
Given historical Redgram price data in CSV format, the goals of this project are to:

* Analyze long-term and short-term price trends
* Compute average prices at multiple granularities:

  * Daily
  * Monthly
  * Yearly
* Train a time-series model that can learn non-linear temporal dependencies
* Forecast future prices for a given year
* Determine the overall market trend direction (UP or DOWN)

---

## Approach Overview

### Data Aggregation

* Multiple CSV files are combined into a unified dataset
* Prices are aggregated at a daily level to reduce noise and ensure consistency

### Exploratory Data Analysis (EDA)

* Year-wise price movement to understand long-term trends
* Monthly seasonality analysis to capture cyclical behavior
* Daily price variation to observe short-term volatility

### Modeling Strategy

* An **LSTM-based autoregressive model** is used for forecasting
* A sliding window of past prices is provided as input
* The model predicts the next time step and rolls forward for long-horizon forecasts
* This approach allows the model to learn complex, non-linear temporal relationships that classical models struggle with

### Evaluation Metrics

Model performance is evaluated using:

* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* Mean Absolute Error (MAE)
* R² Score

### Inference Design

* Inference is **CSV-free**
* Only the most recent observed price window is required
* The model generates daily predictions for an entire year
* Monthly and yearly summaries are derived from daily predictions

---

## Tech Stack

* Python
* PyTorch
* Pandas
* NumPy
* Scikit-learn
* Matplotlib

---

## Project Structure

```
├── data/
│   ├── Prakasam-Redgram.csv
│   ├── Redgram_2.csv
│   └── Redgram_3.csv
│
├── notebooks/
│   └── redgram_analysis.ipynb
│
├── models/
│   ├── redgram_price_lstm.pt
│   └── redgram_price_scaler.pkl
│
├── src/
│   ├── train.py
│   ├── inference.py
│   └── utils.py
│
└── README.md
```

---

## Model Architecture

* **Input:** Last `N` days of prices (default `N = 14`)
* **LSTM Layer**

  * Hidden size: 64
  * Batch-first configuration
* **Fully Connected Layer**

  * Outputs the next-day price
* **Loss Function:** Mean Squared Error (MSE)
* **Optimizer:** Adam

---

## Training Results

The LSTM model shows strong predictive performance on unseen data:

```
LSTM Metrics
MSE: 44430.5390625
RMSE: 210.78552858889532
MAE: 107.61100006103516
R2: 0.98097825050354
```

The high R² score indicates that the model captures the underlying price dynamics effectively and explains most of the variance in the data.

---

## Model Saving & Reuse

After training, both the model weights and the data scaler are saved:

* `redgram_price_lstm.pt`
* `redgram_price_scaler.pkl`

This enables:

* Reproducible results
* Fast inference without retraining
* Seamless integration into APIs or applications

---

## Inference Design (Production Perspective)

At inference time, the system does **not** depend on CSV files.

Instead, it requires:

* The most recent price window
* The target year
* The trained model and scaler

This design choice makes the solution suitable for:

* Real-time APIs
* Dashboards
* Decision-support systems

---

## Inference Usage Example

```python
last_14_days = [5420, 5450, 5480, 5470, 5490, 5520, 5530, 5550, 5540, 5560, 5580, 5600, 5590, 5610]

df, monthly_avg, yearly_avg, trend = predict_year_from_window(
    model,
    scaler,
    last_14_days,
    2026
)

print(yearly_avg)
print(trend)
print(monthly_avg)
```

---

## Output Generated

* Daily price predictions for the selected year
* Monthly average price estimates
* Overall yearly average price
* Market trend direction (UP / DOWN)
* Visualizations for interpretability

---

## Key Design Decisions

* LSTM was chosen to model temporal dependencies that linear and tree-based models fail to capture
* Sliding window forecasting provides stable long-horizon predictions
* Clear separation between training and inference pipelines
* CSV dependency removed during inference for production suitability

---

## Use Cases

* Agricultural commodity price forecasting
* Market intelligence and analytics
* Policy and planning support
* Farmer advisory systems
* Supply chain and procurement analysis

---

## Future Enhancements

* Multi-year forecasting
* Confidence intervals and uncertainty estimation
* GRU or Transformer-based time series models
* Explicit trend classification head
* Deployment via FastAPI
* Model explainability and interpretability

---

## Author Notes

This project focuses on building a **clean, realistic, and production-aware** time series system.
The emphasis is on solid modeling choices, reproducibility, and practical deployment considerations rather than just achieving metrics.

Feel free to fork, experiment, and extend this work.
