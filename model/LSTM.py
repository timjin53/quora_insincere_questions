import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
  def __init__(self, embedding_size, hidden_size, num_classes):
    super(LSTMClassifier, self).__init__()
    self.hidden_size = hidden_size

    self.lstm = nn.LSTM(embedding_size, hidden_size, bidirectional=True)
    self.fc = nn.Linear(2*hidden_size, num_classes)
    self.hidden = self.init_hidden()

  def init_hidden(self):
    return (torch.zeros(2, 1, self.hidden_size), torch.zeros(2, 1, self.hidden_size))

  def forward(self, embedding):
    lstm_out, self.hidden = self.lstm(
      embedding, self.hidden
    )
    return self.fc(lstm_out)
