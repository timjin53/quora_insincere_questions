import os
import numpy as np
import pandas as pd
import pickle
from gensim.parsing.preprocessing import remove_stopwords
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from tensorboardX import SummaryWriter

from model.SimpleNN import SimpleNN
from model.LSTM import LSTMClassifier

# load training data
# https://stackoverflow.com/questions/31468117/python-3-can-pickle-handle-byte-objects-larger-than-4gb
max_bytes = 2**31 - 1
bytes_in = bytearray(0)
input_size = os.path.getsize('data/processed_data/train_data.pkl')
with open('data/processed_data/train_data.pkl', 'rb') as f_in:
    for _ in range(0, input_size, max_bytes):
        bytes_in += f_in.read(max_bytes)

train_data = pickle.loads(bytes_in)
# validation_data = pickle.load(open('data/processed_data/validation_data.pkl', 'rb'))
test_data = pickle.load(open('data/processed_data/test_data.pkl', 'rb'))

train_X = train_data['question_text_embedding']
train_y = train_data['target']
# validation_X = validation_data['question_text_embedding']
# validation_y = validation_data['target']
test_X = test_data['question_text_embedding']
test_y = test_data['target']

# hyper parameters
learning_rate = 0.0001
num_of_epochs = 1
batch_size = 512
embedding_size = 300
num_of_batches = int(np.ceil(len(train_X)/batch_size))

# model, loss function and optimizer
nnModel = SimpleNN(input_size=embedding_size,
                num_classes=2)
lstmModel = LSTMClassifier(embedding_size=embedding_size, hidden_size=128, num_classes=2)
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(lstmModel.parameters(), lr=learning_rate)
writer = SummaryWriter(comment="quora_insincere_question")

# training with LSTM
for epoch in range(num_of_epochs):
  for batch in range(num_of_batches):
    lstmModel.zero_grad()
    lstmModel.hidden = lstmModel.init_hidden()
    batch_loss = 0
    validation_loss = 0
    print("epoch:{0}/{1}, batch:{2}/{3}".format(epoch+1, num_of_epochs, batch+1, num_of_batches))
    startIndex = batch*batch_size
    x = torch.tensor(train_X[startIndex:startIndex+batch_size], dtype=torch.float32)
    y = torch.tensor(train_y[startIndex:startIndex+batch_size], dtype=torch.long)
    probs = lstmModel(x.view(len(x), 1, -1))
    loss = loss_func(probs.view(len(probs), -1), y)
    loss.backward()
    optimizer.step()
    batch_loss += loss.item()
    print("training loss: {}".format(batch_loss))
    writer.add_scalar('train_loss', batch_loss, batch + num_of_batches*epoch)

writer.close()

# run test data
lstmModel.eval()

y_pred = []
for index, X in enumerate(test_X):
  X = torch.tensor(X, dtype=torch.float32).view(1, 1, -1)
  sincere_prob, insincere_prob = lstmModel(X)[0][0]
  if sincere_prob > insincere_prob:
    y_pred.append(0)
  else:
    y_pred.append(1)

print(f1_score(test_y, y_pred))
print(confusion_matrix(test_y, y_pred))

torch.save(lstmModel.state_dict(), './saved_models/lstm')
print('Model saved in saved_models/lstm')
