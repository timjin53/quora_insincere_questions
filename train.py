import numpy as np
import pandas as pd
import pickle
from gensim.parsing.preprocessing import remove_stopwords
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter

# models
from model.SimpleNN import SimpleNN

# read data and preprocess
train_data = pd.read_csv('./data/train.csv')
test_data = pd.read_csv('./data/test.csv')
train_data, validation_data = train_test_split(train_data, test_size=0.20, random_state=42)

# loading embedding
glove_embedding = pickle.load(open('./embeddings/glove.840B.300d.pkl', 'rb'))

# process data
comment_list = train_data['question_text'].str.replace(r'[^\w\s]','').str.lower().tolist()
comment_list_no_stopword = []
comment_embeddings = []

# remove stop words
for comment in comment_list:
  comment_list_no_stopword.append(remove_stopwords(comment))

# get embedding for each word in the sentence 
# get the average of all the word embedding in the sentence
for comment in comment_list_no_stopword:
  embedding = np.array([0.0]*300)
  words = comment.split(' ')
  for word in words:
    embedding += np.array(glove_embedding.get(word, [0.0]*300))
  embedding = embedding / len(words)
  comment_embeddings.append(embedding)

# hyper parameters
learning_rate = 0.001
num_of_epochs = 3
batch_size = 512
embedding_size = 300
num_of_batches = int(np.ceil(len(comment_embeddings)/batch_size))

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
    x = torch.tensor(comment_embeddings[startIndex:startIndex+batch_size], dtype=torch.float32)
    y = torch.tensor(train_data["target"].tolist()[startIndex:startIndex+batch_size], dtype=torch.long)
    probs = model(x)
    loss = loss_func(probs, y)
    loss.backward()
    optimizer.step()
    batch_loss += loss.item()
    writer.add_scalar('loss', batch_loss, batch + num_of_batches*epoch)

writer.close()
 