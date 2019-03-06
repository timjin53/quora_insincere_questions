import pandas  as pd
import pickle
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

from utils.helper import generate_sentence_embedding

# read data and preprocess
train_data = pd.read_csv('./data/train.csv')

# loading embedding
glove_embedding = pickle.load(open('./embeddings/glove.840B.300d.pkl', 'rb'))

# process data and generate embedding for sentence
train_data = generate_sentence_embedding(train_data, glove_embedding)

# split data for train and validation/test
train_data, validation_data = train_test_split(train_data, test_size=0.04, random_state=42)
validation_data, test_data = train_test_split(validation_data, test_size=0.5, random_state=42)

# oversampling imbalanced data
print(train_data['target'].value_counts())
sm = SMOTE(random_state=42, n_jobs=4, ratio=0.1)
train_X = train_data['question_text_embedding'].tolist()
train_y = train_data['target'].tolist()

# print("Oversampling data...")
# train_X, train_y = sm.fit_resample(train_X, train_y)

data = {
  'target': train_y
}
oversampled_train_data = pd.DataFrame(data=data)
print(oversampled_train_data['target'].value_counts())

data['question_text_embedding'] = train_X

# https://stackoverflow.com/questions/31468117/python-3-can-pickle-handle-byte-objects-larger-than-4gb
max_bytes = 2**31 - 1
bytes_out = pickle.dumps(data, protocol=4)
with open('data/processed_data/train_data.pkl', 'wb') as f_out:
    for idx in range(0, len(bytes_out), max_bytes):
        f_out.write(bytes_out[idx:idx+max_bytes])

pickle.dump(validation_data, open(f'data/processed_data/validation_data.pkl', 'wb'))
pickle.dump(test_data, open(f'data/processed_data/test_data.pkl', 'wb'))

print('data saved in data/processed_data')
