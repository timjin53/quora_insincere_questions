import numpy as np
import pickle

glove_embedding = {}

# f = open('./glove.840B.300d/glove.840B.300d.txt')
# for line in f:
#   values = line.split(' ')
#   word = values[0]
#   coefs = np.asarray(values[1:], dtype='float32')
#   glove_embedding[word] = coefs
# f.close()

# pickle.dump(glove_embedding, open('./glove.840B.300d.pkl', 'wb'))

with open(f'./glove.840B.300d/glove.840B.300d.txt', 'rb') as f:
  for l in f:
    values = l.decode().split(' ')
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    glove_embedding[word] = coefs

# pickle.dump(glove_embedding, open(f'glove.840B.300d.pkl', 'wb'))
# https://stackoverflow.com/questions/31468117/python-3-can-pickle-handle-byte-objects-larger-than-4gb
max_bytes = 2**31 - 1
bytes_out = pickle.dumps(glove_embedding)
with open('glove.840B.300d.pkl', 'wb') as f_out:
    for idx in range(0, len(bytes_out), max_bytes):
        f_out.write(bytes_out[idx:idx+max_bytes])
