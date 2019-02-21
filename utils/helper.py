import pandas as pd
import numpy as np
import pickle
from gensim.parsing.preprocessing import remove_stopwords

def generate_sentence_embedding(data, word_embedding):
  comment_list = data['question_text'].str.replace(r'[^\w\s]','').str.lower().tolist()
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
      embedding += np.array(word_embedding.get(word, [0.0]*300))
    embedding = embedding / len(words)
    comment_embeddings.append(embedding)

  data['question_text_embedding'] = comment_embeddings

  return data