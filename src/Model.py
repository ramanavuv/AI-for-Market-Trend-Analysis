import torch
import torch.nn as nn

class LSTMModel(nn.Module):
def init(self, input_size=1, hidden_size=64):
super().init()
self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
self.fc = nn.Linear(hidden_size, 1)

def forward(self, x):
    out, _ = self.lstm(x)
    return self.fc(out[:, -1])
    
    
