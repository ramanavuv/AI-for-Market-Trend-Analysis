import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from model import LSTMModel
from utils import save_model, create_sequences

class PriceDataset(Dataset):
def **init**(self, X, y):
self.X = torch.tensor(X, dtype=torch.float32)
self.y = torch.tensor(y, dtype=torch.float32)

```
def __len__(self):
    return len(self.X)

def __getitem__(self, idx):
    return self.X[idx], self.y[idx]
```

def train(csv_paths, window=14, epochs=50, batch_size=32, lr=0.001):
dfs = [pd.read_csv(p) for p in csv_paths]
data = pd.concat(dfs, ignore_index=True)
data.columns = [c.lower().strip() for c in data.columns]
data['date'] = pd.to_datetime(data['date'])
data = data.sort_values('date')

```
daily = data.groupby('date')['modal_price'].mean().reset_index()
values = daily['modal_price'].values.reshape(-1,1)

scaler = MinMaxScaler()
values = scaler.fit_transform(values)

X, y = create_sequences(values, window)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

train_loader = DataLoader(PriceDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(PriceDataset(X_test, y_test), batch_size=batch_size)

model = LSTMModel()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

for epoch in range(epochs):
    model.train()
    loss_sum = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
    print(f"Epoch {epoch+1}/{epochs} Train Loss:", loss_sum / len(train_loader))

model.eval()
preds, true = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        p = model(xb)
        preds.append(p.numpy())
        true.append(yb.numpy())

preds = scaler.inverse_transform(np.vstack(preds))
true = scaler.inverse_transform(np.vstack(true))

mse = mean_squared_error(true, preds)
rmse = np.sqrt(mse)
mae = mean_absolute_error(true, preds)
r2 = r2_score(true, preds)

print("Evaluation Metrics")
print("MSE:", mse)
print("RMSE:", rmse)
print("MAE:", mae)
print("R2:", r2)

save_model(model, scaler, "models/redgram_price")
```

