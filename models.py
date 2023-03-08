import torch
import torch.nn as nn 

class LSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, num_layers=1):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, num_layers=num_layers)
        self.hidden2class = nn.Linear(hidden_dim, 1) # for BCE loss

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        pred_lbl = self.hidden2class(lstm_out[:, -1, :])
        return pred_lbl


class LSTMClassifier_nopad(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super(LSTMClassifier_nopad, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.hidden2class = nn.Linear(hidden_dim, 1) # for BCE loss

    def forward(self, sentence, sentence_len):
        embeds = self.word_embeddings(sentence)
        pack = torch.nn.utils.rnn.pack_padded_sequence(embeds, sentence_len, enforce_sorted=False, batch_first=True)
        lstm_out, _ = self.lstm(pack)
        unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(lstm_out)
        unpacked = unpacked.transpose(0, 1)
        last_feature = torch.gather(unpacked.cuda(), dim=1, index=unpacked_len.cuda()[None, ..., None].tile(1, 1, 32) - 1)[0]
        #print(last_feature[0], unpacked[0][unpacked_len[0] - 1])
        pred_lbl = self.hidden2class(last_feature)
        return pred_lbl

class LSTMClassifier_nopad2(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super(LSTMClassifier_nopad2, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.hidden2class = nn.Linear(hidden_dim, 1) # for BCE loss

    def forward(self, sentence, sentence_len):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        last_feature = torch.gather(lstm_out, dim=1, index=sentence_len.long().cuda()[None, ..., None].tile(1, 1, 32) - 1)[0]
        #print(last_feature[0], unpacked[0][unpacked_len[0] - 1])
        pred_lbl = self.hidden2class(last_feature)
        return pred_lbl