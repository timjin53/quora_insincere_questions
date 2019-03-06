import pandas as pd
import pickle
import torch

from model.SimpleNN import SimpleNN
from utils.helper import generate_sentence_embedding

test_data = pd.read_csv('./data/test.csv')
glove_embedding = pickle.load(open('./embeddings/glove.840B.300d.pkl', 'rb'))

test_data = generate_sentence_embedding(test_data, glove_embedding)

test_X = test_data['question_text_embedding']

# model, loss function and optimizer
model = SimpleNN(input_size=300,
                num_classes=2)
model.load_state_dict(torch.load('./saved_models/simple_nn'))
model.eval()

result = []
for x in test_X:
  x = torch.tensor(x, dtype=torch.float32)
  sincere_prob, insincere_prob = model(x)
  if(sincere_prob > insincere_prob):
    result.append(0)
  else:
    result.append(1)    

test_data["prediction"] = result

test_data[["qid", "prediction"]].to_csv('submission.csv',index=False)
print("Submission file gerenated.")
