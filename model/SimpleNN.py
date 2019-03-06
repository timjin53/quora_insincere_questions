import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNN(nn.Module):
  def __init__(self, input_size, num_classes):
    super(SimpleNN, self).__init__()
    self.fc1 = nn.Linear(input_size, 128)
    self.relu1 = nn.ReLU()
    self.fc2 = nn.Linear(128, 256)
    self.relu2 = nn.ReLU()
    self.fc3 = nn.Linear(256, 128)
    self.relu3 = nn.ReLU()
    self.fc4 = nn.Linear(128, 128)
    self.relu4 = nn.ReLU()
    self.fc5 = nn.Linear(128, num_classes)

  def forward(self, x):
    x = self.fc1(x)
    x = self.relu1(x)
    x = self.fc2(x)
    x = self.relu2(x)
    x = self.fc3(x)
    x = self.relu3(x)
    x = self.fc4(x)
    x = self.relu4(x)
    x = self.fc5(x)
    return x
