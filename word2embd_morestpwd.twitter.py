import numpy as np
import torch
import torch.nn as nn

from data_util import get_dataset, get_dataset_stopword, get_dataset_morestopword
from models import LSTMClassifier

X_train, Y_train, X_test = get_dataset_morestopword()

word2idx = {}
for twt in X_train:
    for word in twt.split(" "):
        if word not in word2idx:
            word2idx[word] = len(word2idx)

def pad_features(twts):
    twts_idx = [[word2idx[word] + 1 for word in twt.split(" ")] for twt in twts]
    max_len = max([len(twt_idx) for twt_idx in twts_idx])
    features = np.zeros((len(twts), max_len), dtype=int) #zero padding
    for i, twt_idx in enumerate(twts_idx):
        features[i, -(len(twt_idx)):] = twt_idx

    return features

X_train_idx_padded = pad_features(X_train)

from torch.utils.data import TensorDataset, DataLoader
train_data = TensorDataset(torch.from_numpy(X_train_idx_padded), torch.from_numpy(Y_train))
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=64, num_workers=4)

model = LSTMClassifier(embedding_dim=25, hidden_dim=32, vocab_size=len(word2idx) + 1)

def train_model(model):
    model.cuda()
    loss_function = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

    print_every = 100
    cum_loss = 0.
    cum_acc = 0.
    for epoch in range(100): 
        torch.save(model, "./log/more_stop_word/{:02d}.pt".format(epoch))
        for iter, (twt, lbl) in enumerate(train_dataloader):
            
            optimizer.zero_grad()
            tag_scores = model(twt.cuda())
            loss = loss_function(tag_scores.reshape(-1), lbl.float().cuda())
            loss.backward()
            optimizer.step()

            cum_loss += loss.item()
            cum_acc += ((tag_scores.reshape(-1).cpu() > 0) == lbl).float().mean().detach().item()
            if iter % print_every == 0 and iter != 0:
                print(cum_loss / print_every, cum_acc / print_every)
                cum_loss = 0.
                cum_acc = 0.

    return model

# model = train_model(model)
model = torch.load("./log/more_stop_word/99.pt", map_location=torch.device('cpu'))

def pad_features_test(twts, X_train_idx_padded):
    twts_idx = [[word2idx[word] + 1 if word in word2idx else 0 for word in twt.split(" ")] for twt in twts]
    max_len = X_train_idx_padded.shape[-1]
    features = np.zeros((len(twts), max_len), dtype=int) #zero padding
    for i, twt_idx in enumerate(twts_idx):
        features[i, -(len(twt_idx)):] = twt_idx

    return features

X_test_idx_padded = pad_features_test(X_test, X_train_idx_padded)
test_data = TensorDataset(torch.from_numpy(X_test_idx_padded))
test_dataloader = DataLoader(test_data, shuffle=False, batch_size=1)

model.cpu()

with open("./word2embd.morestpwd.twitter_lstm.csv", "w") as f:
    f.write("Id,Prediction\n")
    for iter, twt in enumerate(test_dataloader):
        tag_scores = model(twt[0])
        f.write("{},{}\n".format(iter + 1, int((tag_scores.reshape(-1) > 0.) * 2 - 1)))