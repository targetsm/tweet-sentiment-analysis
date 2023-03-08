import os
import numpy as np
from sklearn.linear_model import LogisticRegression

stacking_regressor = LogisticRegression()

def get_dataset():
    tweets = []
    labels = []
    def load_tweets(filename, label):
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                tweets.append(line.rstrip())
                labels.append(label)
    
    load_tweets('twitter-datasets/train_neg_full.txt', -1)
    load_tweets('twitter-datasets/train_pos_full.txt', 1)

    # Convert to NumPy array to facilitate indexing
    tweets = np.array(tweets)
    labels = np.array(labels)

    def get_test_set(filename):
        twt = []
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                twt.append(line.split(",", 1)[1].rstrip())

        return twt

    X_test = get_test_set('twitter-datasets/test_data.txt')

    return tweets, labels, X_test

def train_prediction(X_embd_train, X_embd_test, Y_train, model):
    shuffled_indices = np.random.permutation(len(X_embd_train))
    split_idx = int(1.0 * len(X_embd_train))
    train_indices = shuffled_indices[:split_idx]
    val_indices = shuffled_indices[split_idx:]

    fitted_model = model.fit(X_embd_train[train_indices], Y_train[train_indices])

    # print(model.score(X_embd_train[val_indices], Y_train[val_indices]))

    return fitted_model.predict(X_embd_test)

X_train, Y_train, X_test = get_dataset()

import pickle

reload_embd = False
if reload_embd:
    with open("./embd/word2vec.twitter.train", "rb") as f:
        X_embd_train = pickle.load(f)
    with open("./embd/word2vec.twitter.test", "rb") as f:
        X_embd_test = pickle.load(f)
else:
    # copied from local notebook, might be wrong though
    from gensim.models import Word2Vec
    from gensim.test.utils import common_texts
    from gensim.models.word2vec import PathLineSentences
    total_data = PathLineSentences('embd/twt_data') # contains train_pos/neg_full.txt only
    w2v_model = Word2Vec(sentences=total_data, min_count=1, max_vocab_size=5000, vector_size=25)
    w2v_model.save("word2vec.model")

    def get_word2vec_mean_embedding(X, dim=25):
        embd_list = []
        for twt in X:
            cnt = 0
            embd = np.zeros(dim)
            for word in twt.split(" "):
                try:
                    embd += w2v_model.wv[word]
                    cnt += 1
                except:
                    continue

            embd_list.append(embd / cnt if cnt != 0 else np.zeros(dim))

        return np.array(embd_list)

    X_embd_train = get_word2vec_mean_embedding(X_train)
    X_embd_test = get_word2vec_mean_embedding(X_test)

Y_test_pred = train_prediction(X_embd_train, X_embd_test, Y_train, stacking_regressor)

with open("./word2vec.twitter_reg.csv", "w") as f:
    f.write("Id,Prediction\n")
    for idx, pred in enumerate(Y_test_pred):
        f.write("{},{}\n".format(idx + 1, pred))

exit()