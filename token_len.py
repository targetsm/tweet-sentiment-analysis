import numpy as np
import pandas as pd
import seaborn as sns
import collections
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, DistilBertModel

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    
    
"""# Load data"""

tweets = []
labels = []

def load_tweets(filename, label):
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            tweets.append(line.rstrip())
            labels.append(label)
    
#load_tweets('twitter-datasets/train_neg.txt', 0)
#load_tweets('twitter-datasets/train_pos.txt', 1)
load_tweets('twitter-datasets/train_neg_full.txt', 0)
load_tweets('twitter-datasets/train_pos_full.txt', 1)

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

token_lens = []
print(f'Total number of tweets: {len(tweets)}\nCurrently processing: ', end='')
for i, tweet in enumerate(tweets):
    if i % 10000 == 0: print(f'{i}, ', end='')
    tokens = tokenizer.encode(tweet, max_length=128, truncation=True)
    token_lens.append(len(tokens))
counter = collections.Counter(token_lens)
print(counter)
sns.distplot(token_lens)
plt.xlim([0, 64]);
plt.xlabel('Token count');
plt.show()