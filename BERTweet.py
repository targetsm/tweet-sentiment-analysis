# doc: https://huggingface.co/docs/transformers/model_doc/bertweet
# boilerplate based on https://curiousily.com/posts/sentiment-analysis-with-bert-and-hugging-face-using-pytorch-and-python/

import os
import random
import time
import datetime
import re
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
import torch
from torch import nn, optim
from collections import defaultdict
from transformers import AutoModel, AutoTokenizer, BertModel, BertTokenizer, get_linear_schedule_with_warmup, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import TensorDataset,DataLoader, RandomSampler, SequentialSampler, Dataset

# Use CUDA if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'\nUSING DEVICE: {device}\n')

# Params
MODEL_NAME = 'vinai/bertweet-base'
MODEL_PATH = 'out/model_bertweet_state.bin'
SUBMISSION_PATH = 'out/bertweet_submission.csv'
MAX_LEN = 45
BATCH_SIZE = 16
EPOCHS = 100

nltk_stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]


def get_twitter_dataset(tokenizer, set='full', split=0.1, sample=0, output=False, rm_stopwords=True):
    # Load training data
    tweets = []
    labels = []

    def load_tweets(filename, label):
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                if rm_stopwords:
                    tweets.append(" ".join([x for x in line.rstrip().split(" ") if x not in nltk_stopwords]))
                else:
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

    # Preprocessing (special BERTweet tokens)
    df['tweet'] = df['tweet'].str.replace('<user>', '@USER', regex=True)
    df['tweet'] = df['tweet'].str.replace('<url>', 'HTTPURL', regex=True)

    
    if output:
        print(f'{len(df)} tweets loaded:')
        print(f'{df}\n') 
        #token_lens = []
        #for tweet in tweets:
        #    tokens = tokenizer.encode(tweet, max_length=128)
        #    token_lens.append(len(tokens))
        #sns.distplot(token_lens)
        #plt.xlim([0, 128]);
        #plt.xlabel('Token count');
        #plt.show()
    else:
        print(f'{len(df)} tweets loaded.\n')
        
    # Split up
    df_train, df_val = train_test_split(df, test_size=0.1, shuffle=True, random_state=1)

    train_dl = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
    val_dl = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
    
    return train_dl, val_dl

def get_test_data(tokenizer, rm_stopwords=True):
    # Load test data
    test_tweets = []
    with open('twitter-datasets/test_data.txt', 'r', encoding='utf-8') as f:
        for line in f:
            if rm_stopwords:
                test_tweets.append(" ".join([x for x in line.rstrip().split(" ") if x not in nltk_stopwords]))
            else:
                test_tweets.append(line.rstrip())
    test_dict = { 'tweet' : test_tweets }
    df_test = pd.DataFrame(test_dict)

    df_test['tweet'] = df_test['tweet'].str.replace('<user>', '@USER', regex=True)
    df_test['tweet'] = df_test['tweet'].str.replace('<url>', 'HTTPURL', regex=True)
    
    test_dl = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)
    
    return df_test, test_dl
    

def longest_tweets(tweets):
    x = 0
    for tweet in tweets:
        if len(tweet.split()) > x:
            x = len(tweet.split())
        if len(tweet.split()) >= 64:
            print(tweet.split())
    print(f'\nMax Length: {x}\n')


# Dataset class definition
class TwitterDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_len=64):
        self.df = df
        self.maxlen = max_len
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        tweet = self.df.tweet.values[index]
        sentiment = None
        if 'label' in self.df.columns:
            sentiment = int(self.df.label.values[index])
        encoding = tokenizer.encode_plus(
            tweet,
            max_length = self.maxlen,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt'
        )
        if sentiment == None:            
            return {
                'tweet': tweet,
                'input_ids': encoding.input_ids.flatten(),
                'attention_mask': encoding.attention_mask.flatten(),
            }
        return {
            'tweet': tweet,
            'input_ids': encoding.input_ids.flatten(),
            'attention_mask': encoding.attention_mask.flatten(),
            'targets': torch.tensor(sentiment, dtype=torch.long)
        }


def create_data_loader(df, tokenizer, max_len=64, batch_size=16):
    dataset = TwitterDataset(df, tokenizer, max_len)
    return torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)
    

def check_out_pretrained_model():
    # Tokenization
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

    print(f'\n\ndf: {df}')
    print(f'\n    Tweet: {df.tweet.values[0]}')
    tokens = tokenizer.tokenize(df.tweet.values[0])
    print(f'   Tokens: {tokens} (len: {len(tokens)})')

    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    print(f'Token IDs: {token_ids} (len: {len(token_ids)})')


    print(f'\nClassification token: {tokenizer.cls_token}, id {tokenizer.cls_token_id}')
    print(f'       Padding token: {tokenizer.pad_token}, id {tokenizer.pad_token_id}')
    print(f'   Separation token: {tokenizer.sep_token}, id {tokenizer.sep_token_id}')
    print(f'      Unknown token: {tokenizer.unk_token}, id {tokenizer.unk_token_id}')


    encoding = tokenizer.encode_plus(
        df.tweet.values[0],
        max_length = 64,
        add_special_tokens=True,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_token_type_ids=False,
        return_tensors='pt'
        )

    print(f'\n\nEncoding Input IDs: {encoding["input_ids"]}')
    print(f'        Attention Mask: {encoding["attention_mask"]}')



    df_train, df_val = train_test_split(df, test_size=0.1, random_state=1)
    df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=1)

    print(f'\ndf_train: {df_train.shape}')
    print(f' df_test: {df_test.shape}')
    print(f'  df_val: {df_val.shape}')


    train_dl = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
    val_dl = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
    test_dl = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

    data = next(iter(train_dl))
    print(f'\nDataLoader Input IDs shape: {data["input_ids"].shape}')
    print(f'DataLoader Attn Mask shape: {data["attention_mask"].shape}')
    print(f'DataLoader   Targets shape: {data["targets"].shape}\n')


    # Check Model

    model = AutoModel.from_pretrained(MODEL_NAME)
    out = model(
        input_ids = encoding['input_ids'], 
        attention_mask=encoding['attention_mask']
    )
    print(f'\n\nLast hidden state shape: {out["last_hidden_state"].shape}')
    print(f'    Pooled output shape: {out["pooler_output"].shape}')
    print(f'      Model hidden size: {model.config.hidden_size}')


# Our classifier
class SentimentClassifier(nn.Module):
    def __init__(self, freeze_bert=False):
        super(SentimentClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, 2)
        self.softmax = nn.Softmax(dim=1)
        
        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
    def forward(self, input_ids, attention_mask):
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = out['pooler_output']
        output = self.drop(pooled_output)
        output = self.out(output)
        return self.softmax(output)


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))
    
def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples, t0, epoch):
    model = model.train()
    losses = []
    correct_predictions = 0
    t0_epoch = time.time()
    
    for step, d in enumerate(data_loader):
    
        if step % 100 == 0 and not step == 0:
                print('  Batch {:>5,}  of  {:>5,}.  Loss: {:}'.format(step, len(data_loader), str(np.mean(losses[-100:]))[:5]), end='')
                t = time.time()
                epoch_elapsed = format_time(t - t0_epoch)
                total_elapsed = format_time(t - t0)
                epoch_total = format_time((t - t0_epoch) * (len(data_loader)/step))
                total = format_time(((t - t0_epoch) * (len(data_loader)/step)) * EPOCHS)
                epoch_remaining = format_time(((t - t0_epoch) * (len(data_loader)/step)) - (t - t0_epoch))
                total_remaining = format_time((((t - t0_epoch) * (len(data_loader)/step)) - (t - t0_epoch)) + ((t - t0_epoch) * (len(data_loader)/step)) * (EPOCHS-epoch))
                perc_epoch = str(100*step/len(data_loader))
                if len(perc_epoch) < 4: perc_epoch = perc_epoch+'0000'
                perc_total = str(100*((epoch-1)*len(data_loader)+step)/(EPOCHS*len(data_loader)))
                if len(perc_total) < 4: perc_total = perc_total+'0000'
                print(f' \tEpoch: {perc_epoch[:4]}%  {epoch_elapsed}/{epoch_total} (remaining: {epoch_remaining}) \tTotal: {perc_total[:4]}%  {total_elapsed}/{total} (remaining: {total_remaining})')
            
        input_ids = d['input_ids'].to(device)
        attention_mask = d['attention_mask'].to(device)
        targets = d['targets'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)
        
        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
    return correct_predictions.double()/n_examples, np.mean(losses)
    
def eval_model(model, data_loader, loss_fn, device, n_examples, t0, epoch):
    model = model.eval()
    
    losses = []
    correct_predictions = 0
    
    with torch.no_grad():
        for step, d in enumerate(data_loader):
            if step % 100 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.  Loss: {:}  Elapsed: {:}. '.format(step, len(data_loader), str(np.mean(losses[-100:]))[:5], elapsed))
         
            input_ids = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            targets = d['targets'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)
            
            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())
        
    return correct_predictions.double()/n_examples, np.mean(losses)

def get_predictions(model, data_loader):
    model = model.eval()
    
    tweet_texts = []
    predictions = []
    prediction_probs = []
    real_values = []
    
    losses = []
    correct_predictions = 0
    t0 = time.time()
    
    with torch.no_grad():
        for step, d in enumerate(data_loader):
            if step % 100 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.  Loss: {:}  Elapsed: {:}. '.format(step, len(data_loader), str(np.mean(losses[-100:]))[:5], elapsed))
         
            texts = d['tweet']
            input_ids = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            targets = None
            if 'targets' in d:
                targets = d['targets'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            _, preds = torch.max(outputs, dim=1)
        
            tweet_texts.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(outputs)
            if targets != None:
                real_values.extend(targets)
            
    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()    
    if targets != None:
        real_values = torch.stack(real_values).cpu()

    return tweet_texts, predictions, prediction_probs, real_values

def model_train_new(train_dl, val_dl, tokenizer, predict_at_best_epoch=True, Freeze=False):
    
    model = SentimentClassifier(freeze_bert=Freeze)
    model = model.to(device)
    
    '''
    input_ids = data['input_ids'].to(device)
    attention_mask = data['attention_mask'].to(device)

    out = model(input_ids, attention_mask)
    print(f'Untrained Model output: {out}')
    '''

    # Training

    #optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)

    total_steps = len(train_dl) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    loss_fn = nn.CrossEntropyLoss().to(device)

    history = defaultdict(list)
    best_accuracy = 0
    t0_total = time.time()
    test_data, test_dl = get_test_data(tokenizer)
    
    for epoch in range(EPOCHS):
        
        print("")
        print(f'======================== Epoch {epoch + 1} / {EPOCHS} ========================')
        print('Training...')
        
        t0 = time.time()
        
        train_acc, train_loss = train_epoch(
            model,
            train_dl,
            loss_fn,
            optimizer,
            device,
            scheduler,
            len(train_dl)*BATCH_SIZE,
            t0_total, 
            epoch+1
        )
        
        print(f'Train loss {str(train_loss)[:5]}, Accuracy {str(train_acc)[7:12]}')
        print(f'==============================================================')
        print('Validation...')
        
        val_acc, val_loss = eval_model(
            model,
            val_dl,
            loss_fn,
            device,
            len(val_dl)*BATCH_SIZE,
            t0_total,
            epoch+1
        )
        
        print(f'Train loss {str(train_loss)[:5]}, Accuracy {str(train_acc)[7:12]}')
        print(f'  Val loss {str(val_loss)[:5]}, Accuracy {str(val_acc)[7:12]}\n')
        t = time.time()
        print(f'This epoch took {format_time(t-t0)}')
        print(f'Total time elapsed: {format_time(t - t0_total)}')
        print(f'Expected total time: {format_time((t-t0) * (EPOCHS))}')
        print(f'Expected remaining time: {format_time((t-t0) * (EPOCHS-(epoch+1)))}')
        
        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        if val_acc > best_accuracy:
            torch.save(model.state_dict(), MODEL_PATH)
            best_accuracy = val_acc
            if predict_at_best_epoch:
                print('Best accuracy so far! Running prediction on test set...')
                predictions = predict_tweets(model, test_dl)
                save_submission(predictions)
                print(f'Saved new predictions to {SUBMISSION_PATH}.')
            else:
                print('Best accuracy so far!')
    
    return model

def model_from_file(name=MODEL_NAME):
    print(f'Loading model from file: {MODEL_PATH}\n')
    model = SentimentClassifier()
    model.load_state_dict(torch.load(MODEL_PATH))
    model = model.to(device)
    return model
    
def show_confusion_matrix(confusion_matrix):
    hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
    plt.ylabel('True sentiment')
    plt.xlabel('Predicted sentiment');
    plt.show()
    
def evaluation_metrics(model, dl):
    y_review_texts, y_pred, y_pred_probs, y_test = get_predictions(model, val_dl)
    class_names = ['negative', 'positive']
    print(f'Classification Report: {classification_report(y_test, y_pred, target_names=class_names)}')
    cm = confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    show_confusion_matrix(df_cm)
 
def predict_tweets(model, test_dl):
    y_review_texts, y_pred, y_pred_probs, y_test = get_predictions(model, test_dl)
    return y_pred    
    
def save_submission(predictions):
    ids = range(1,len(predictions)+1)
    out_dict = {
        'Id' : ids,
        'Prediction' : predictions
    }
    out = pd.DataFrame(out_dict)
    out = out.replace(0,-1)
    out.to_csv(SUBMISSION_PATH, index=False)
    print(f'Submission file saved to {SUBMISSION_PATH}' )
   
if __name__ == '__main__':
    
    # BERTweet tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    
    # Load twitter dataset in dataloader and split into train/val/test sets
    print("\n----- LOADING DATA -----\n")
    train_dl, val_dl = get_twitter_dataset(
        tokenizer=tokenizer,
        set='large',         # 'full' or 'small' (i.e. 2'500'000 vs 200'000)
        split=0.01,         # Validation set split
        sample=0,           # Only use n tweets from the full set; 0 = use all
        rm_stopwords=True,  # Remove stopwords from all tweets
        output=True         # Print additional output (sample of used set, ...)
    )
    
    # Get the classifier
    print("\n----- GETTING MODEL -----\n")
    model = model_train_new(
        train_dl, 
        val_dl,
        tokenizer,
        predict_at_best_epoch=True,  # Generate and save prediction on test set after every new best epoch
        Freeze=True                  # Freeze bert model, only trains last layer
    )
    #model = model_from_file(MODEL_NAME)
    
    # Evaluate using validation set
    print("\n----- EVALUATION -----\n")
    evaluation_metrics(model, val_dl)
    
    # Predict test data
    print("\n----- PREDICTION -----\n")
    test_data, test_dl = get_test_data(tokenizer)
    predictions = predict_tweets(model, test_dl)
    
    save_submission(predictions)    
    print("\n----- DONE -----\n")