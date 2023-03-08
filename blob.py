import numpy as np
from textblob import TextBlob


"""# Load data"""

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

print(f'{len(tweets)} tweets loaded')

# Sentiment analysis using textblob
pred = []
for tweet in tweets:
    blob = TextBlob(tweet)
    pred.append((blob.sentiment.polarity+1)/2)
pred = np.array(pred)

print('Finished predictions...')

for threshold in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
    correct = 0
    for p in zip(labels,pred):
        if p[1] > threshold:
            if p[0] == 1:
                correct += 1
        else:
            if p[0] == 0:
                correct += 1
    accuracy = correct/len(tweets)
    print(f'threshold at {threshold} results in accuracy of {accuracy}')
    
''' Results:
2500000 tweets loaded
Finished predictions...
threshold at 0.1 results in accuracy of 0.506436
threshold at 0.2 results in accuracy of 0.5149164
threshold at 0.3 results in accuracy of 0.5301128
threshold at 0.4 results in accuracy of 0.5490288
threshold at 0.5 results in accuracy of 0.6004792
threshold at 0.6 results in accuracy of 0.6086356
threshold at 0.7 results in accuracy of 0.5851616
threshold at 0.8 results in accuracy of 0.5480652
threshold at 0.9 results in accuracy of 0.5173792
'''