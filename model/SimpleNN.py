import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNN(nn.Module):
  def __init__(self, input_size, num_classes):
    super(SimpleNN, self).__init__()
    self.fc1 = nn.Linear(input_size, 128)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(128, num_classes)

  def forward(self, x):
    x = self.fc1(x)
    x = self.relu(x)
    x = self.fc2(x)
    return x
