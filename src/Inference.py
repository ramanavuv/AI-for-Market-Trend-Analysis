import torch
import numpy as np
import pandas as pd
from model import LSTMModel
from utils import load_model

def predict_year(last_window_prices, year, model_path, scaler_path):
model = LSTMModel()
model, scaler = load_model(model, model_path, scaler_path)

```
window = scaler.transform(np.array(last_window_prices).reshape(-1,1))
preds = []

for _ in range(365):
    x = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        p = model(x).numpy()
    preds.append(p[0,0])
    window = np.vstack([window[1:], p])

preds = scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()

dates = pd.date_range(start=f"{year}-01-01", periods=365)
df = pd.DataFrame({"date": dates, "predicted_price": preds})
df["month"] = df["date"].dt.month

monthly_avg = df.groupby("month")["predicted_price"].mean()
yearly_avg = df["predicted_price"].mean()
trend = "UP" if preds[-1] > preds[0] else "DOWN"

return df, monthly_avg, yearly_avg, trend
```

