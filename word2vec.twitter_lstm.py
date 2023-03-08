from gensim.models import Word2Vec
from gensim.test.utils import common_texts
from gensim.models.word2vec import PathLineSentences
import numpy as np

def get_dataset():
    tweets = []
    labels = []
    def load_tweets(filename, label):
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                tweets.append(line.rstrip())
                labels.append(label)
    
    load_tweets('twitter-datasets/train_neg_full.txt', 0)
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

X_train, Y_train, X_test = get_dataset()
total_data = PathLineSentences('./embd/twt_data')
w2v_model = Word2Vec(sentences=total_data, min_count=1, max_vocab_size=5000, vector_size=25)
w2v_model.save("word2vec.model")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

class TwitterDataset(Dataset):
    def __init__(self, X, Y=None, word2vec_model=None):
        self.X = X
        self.Y = Y
        self.word2vec_model = Word2Vec.load("word2vec.model")
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        embd = []
        for word in self.X[idx].split(" "):
            try:
                embd.append(self.word2vec_model.wv[word])
            except:
                continue
        if len(embd) == 0:
            embd = [np.zeros(25)]
        
        if self.Y is not None:
            return torch.from_numpy(np.stack(embd, 0)), torch.Tensor([self.Y[idx]])
        else:
            return torch.from_numpy(np.stack(embd, 0))

from torch.nn.utils.rnn import pad_sequence, PackedSequence

def collate_fn_pad(list_pairs_seq_target):
    seqs = [seq for seq, target in list_pairs_seq_target]
    targets = [target for seq, target in list_pairs_seq_target]
    rev_seqs = [seq.flip((0)) for seq, target in list_pairs_seq_target]
    seqs_padded_batched = pad_sequence(seqs, batch_first=True)   # will pad at beginning of sequences
    rev_seqs_padded_batched = pad_sequence(rev_seqs, batch_first=True)   # will pad at beginning of sequences
    targets_batched = torch.stack(targets)
    assert seqs_padded_batched.shape[0] == len(targets_batched)
    return seqs_padded_batched, rev_seqs_padded_batched, targets_batched
    
dataset = TwitterDataset(X_train, Y_train)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_fn_pad)


from torch.optim import Adam
model = nn.LSTM(25, 32, 1, bidirectional=False, proj_size=1, batch_first=True)
optimizer = Adam(model.parameters(), lr=5e-5)

cum_loss = 0.
cum_acc = 0.
for epoch in range(100):
    for iter, (twt, twt_rev, lbl) in enumerate(dataloader):
        h0, c0 = torch.zeros(1, lbl.shape[0], 1), torch.zeros(1, lbl.shape[0], 32)
        out, (hn, cn) = model(twt.float(), (h0, c0))

        optimizer.zero_grad()
        try:
            loss = torch.nn.BCEWithLogitsLoss()(out[:, -1, 0], lbl.reshape(-1))
        except:
            print(out.shape, lbl.shape)
        loss.backward()
        cum_loss += loss.item()
        cum_acc += (((out[:, -1, :]).detach() > 0) == lbl).float().mean().item()
        if iter % 1000 == 0:
            print(cum_loss / 1000., cum_acc / 1000.)
            cum_loss = 0.
            cum_acc = 0.
        optimizer.step()

tstdataset = TwitterDataset(X_test)
tstdataloader = DataLoader(tstdataset, batch_size=1, shuffle=False)

with open("./word2vec.twitter_lstm.csv", "w") as f:
    f.write("Id,Prediction\n")
    for iter, twt in enumerate(tstdataloader):
        twt = twt.reshape((twt.shape[0], -1, 25))
        h0, c0 = torch.zeros(1, twt.shape[0], 1), torch.zeros(1, twt.shape[0], 32)
        out, (hn, cn) = model(twt.float(), (h0, c0))
        f.write("{},{}\n".format(iter + 1, int((out[0, -1, 0]) > 0.) * 2 - 1))