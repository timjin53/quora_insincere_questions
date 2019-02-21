import numpy as np
import pandas as pd
import pickle
from gensim.parsing.preprocessing import remove_stopwords
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter

from model.SimpleNN import SimpleNN
from utils.helper import generate_sentence_embedding

# read data and preprocess
train_data = pd.read_csv('./data/train.csv')

# loading embedding
glove_embedding = pickle.load(open('./embeddings/glove.840B.300d.pkl', 'rb'))

# process data and generate embedding for sentence
train_data = generate_sentence_embedding(train_data, glove_embedding)

# split training data for train and validation
train_data, validation_data = train_test_split(train_data, test_size=0.20, random_state=42)
train_X = train_data['question_text_embedding'].tolist()
train_y = train_data['target'].tolist()
validation_X = validation_data['question_text_embedding'].tolist()
validation_y = validation_data['target'].tolist()

# hyper parameters
learning_rate = 0.0001
num_of_epochs = 3
batch_size = 512
embedding_size = 300
num_of_batches = int(np.ceil(len(train_X)/batch_size))

# model, loss function and optimizer
model = SimpleNN(input_size=embedding_size,
                num_classes=2)
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
writer = SummaryWriter(comment="quora_insincere_question")

# training
for epoch in range(num_of_epochs):
  for batch in range(num_of_batches):
    model.zero_grad()
    batch_loss = 0
    print("epoch:{0}/{1}, batch:{2}/{3}".format(epoch+1, num_of_epochs, batch+1, num_of_batches))
    startIndex = batch*batch_size
    x = torch.tensor(train_X[startIndex:startIndex+batch_size], dtype=torch.float32)
    y = torch.tensor(train_y[startIndex:startIndex+batch_size], dtype=torch.long)
    probs = model(x)
    loss = loss_func(probs, y)
    loss.backward()
    optimizer.step()
    batch_loss += loss.item()
    print("training loss: {}".format(batch_loss))
    writer.add_scalar('train_loss', batch_loss, batch + num_of_batches*epoch)
  
writer.close()

torch.save(model.state_dict(), './saved_models/simple_nn')
print('Model saved in saved_models/simple_nn')
