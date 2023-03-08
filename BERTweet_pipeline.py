from transformers import pipeline
from transformers import AutoTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split


def get_twitter_dataset(tokenizer, set='full', split=0.1, sample=0, output=False):
    # Load training data
    tweets = []
    labels = []

    def load_tweets(filename, label):
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                tweets.append(line.rstrip())
                labels.append(label)
    
    if set == 'full':
        print("Using full twitter dataset")
        load_tweets('twitter-datasets/train_neg_full.txt', 0)
        load_tweets('twitter-datasets/train_pos_full.txt', 1)
    else:
        print("Using small twitter dataset")
        load_tweets('twitter-datasets/train_neg.txt', 0)
        load_tweets('twitter-datasets/train_pos.txt', 1)

    # Convert to dataframe
    data_dict = {
        'tweet': tweets,
        'label': labels
    }
    df = pd.DataFrame(data_dict)

    if sample > 0:
        df = df.sample(n=sample)
    df = df.astype({"tweet": str, "label": int}, errors='raise') 

    # Preprocessing
    df['tweet'] = df['tweet'].str.replace('<user>', '@USER', regex=True)
    df['tweet'] = df['tweet'].str.replace('<url>', 'HTTPURL', regex=True)

    
    if output:
        print(f'{len(df)} tweets loaded:')
        print(f'{df}\n')
    else:
        print(f'{len(df)} tweets loaded.\n')
        
    # Split up
    df_train, df_val = train_test_split(df, test_size=0.1, shuffle=True, random_state=1)

    return df_train, df_val
    
    
def get_test_data(tokenizer):
    # Load test data
    test_tweets = []
    with open('twitter-datasets/test_data.txt', 'r', encoding='utf-8') as f:
        for line in f:
            test_tweets.append(line.rstrip())
    test_dict = { 'tweet' : test_tweets }
    df_test = pd.DataFrame(test_dict)

    df_test['tweet'] = df_test['tweet'].str.replace('<user>', '@USER', regex=True)
    df_test['tweet'] = df_test['tweet'].str.replace('<url>', 'HTTPURL', regex=True)
    
    return df_test
    
    
if __name__ == '__main__':

    tokenizer = AutoTokenizer.from_pretrained('vinai/bertweet-base', use_fast=False)
    df_train, df_val = get_twitter_dataset(
        tokenizer=tokenizer,
        set='small',  # 'full' or 'small' (i.e. 2'500'000 vs 200'000)
        split=0,      # Validation set split
        sample=1000,    # Only use n tweets from the full set; 0 = use all
        output=True   # Print the used set
    )
    print(df_train)
    
    sentiment_pipeline = pipeline(model="finiteautomata/bertweet-base-sentiment-analysis")
    
    correct = 0
    positive = 0
    negative = 0
    neutral = 0
    total = len(df_train)
    for index, row in df_train.iterrows():
    
        pred = sentiment_pipeline(row['tweet'])
        score = pred[0]['score']
        
        if pred[0]['label'] == 'POS':
            prediction = 1
            positive += 1
        elif pred[0]['label'] == 'NEG':
            prediction = 0
            negative += 1
        else:
            prediction = '?'
            neutral += 1
            
        if row['label'] == prediction:
            correct += 1
        elif prediction != '?':
            print(f'pred: {prediction}, real: {row["label"]},  score {str(score)[:4]},  {row["tweet"]}')
            
        #print(f'pred: {prediction}, real: {row["label"]},  score {str(score)[:4]},  {row["tweet"]}')
    
    print(f'\n{neutral} tweets were classified as neutral\nAccuracy: \t\t\t{str(100*(correct+neutral/2)/total)[:4]}%\nAccuracy (ignoring neutrals): \t{str(100*correct/(total-neutral))[:4]}%')
    