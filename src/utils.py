import torch
import numpy as np
import joblib

def save_model(model, scaler, path):
torch.save(model.state_dict(), path + "_lstm.pt")
joblib.dump(scaler, path + "_scaler.pkl")

def load_model(model, model_path, scaler_path):
model.load_state_dict(torch.load(model_path, map_location="cpu"))
scaler = joblib.load(scaler_path)
model.eval()
return model, scaler

def create_sequences(values, window):
X, y = [], []
for i in range(len(values) - window):
X.append(values[i+window])
y.append(values[i+window])
return np.array(X), np.array(y)
