# Originally based on https://github.com/theartificialguy/NLP-with-Deep-Learning/blob/master/BERT/Amazon%20Review%20Sentiment%20Analysis/amazon_reviews_sentiment_classification.ipynb

import numpy as np
import pandas as pd
import random
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, DistilBertModel

if torch.cuda.is_available():
    torch.cuda.empty_cache()

best_acc = 0.85
train_df = []
test_df = []
model = []

EPOCHS = 3
LR = 5e-6

print(f'Next using {EPOCHS} epochs and LR of {LR}...')

# Dataset class definition
class TwitterDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df
        self.maxlen = 64
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")  
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        tweet = self.df['tweets'].iloc[index].split()
        tweet = ' '.join(tweet)
        sentiment = int(self.df['labels'].iloc[index])        
        encodings = self.tokenizer.encode_plus(
            tweet,
            add_special_tokens=True,
            max_length=self.maxlen,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encodings.input_ids.flatten(),
            'attention_mask': encodings.attention_mask.flatten(),
            'labels': torch.tensor(sentiment, dtype=torch.long)
        }

## Model definition
# Note: Not used currently! Using DistilBertForSequenceClassification instead
class SentimentClassifier(nn.Module):
    def __init__(self):
        super(SentimentClassifier, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.drop0 = nn.Dropout(0.25)
        self.linear1 = nn.Linear(3072, 512)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.25)
        self.linear2 = nn.Linear(512, 2)
        self.relu2 = nn.ReLU()
        
    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids, attention_mask)
        last_hidden_state = outputs[0]
        pooled_output = torch.cat(tuple([last_hidden_state[:, i] for i in [-4, -3, -2, -1]]), dim=-1)
        x = self.drop0(pooled_output)
        x = self.relu1(self.linear1(x))
        x = self.drop1(x)
        x = self.relu2(self.linear2(x))
        return x

    
def set_seed(seed_value=42):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    print(f'\nAll seeds set to {seed_value} for reproducibility!\n')

set_seed()

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

# Convert to dataframe

data_dict = {
    'tweets': tweets,
    'labels': labels
}
df = pd.DataFrame(data_dict)

#df = df.sample(n=50000)

print(f'{len(df)} tweets loaded:')
print(df)
 
        
# Split into train and test datasets
if len(train_df) == 0:
    train_df, test_df = train_test_split(df, test_size=0.01, random_state=1)
train_dataset = TwitterDataset(train_df)
test_dataset = TwitterDataset(test_df)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=24,
    shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=24
)

# Generate model
#model = SentimentClassifier()
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Train model
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.AdamW(model.parameters(), lr=LR)
epochs = EPOCHS

for epoch in range(epochs):
    model.train()
    train_loop = tqdm(train_loader)
    for batch in train_loop:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        output = model(input_ids, attention_mask, labels=labels)
        #loss = output[0]
        #print(output[0]) # seems to be cross entropy loss?
        loss = criterion(output['logits'], labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_loop.set_description(f'Training Epoch: {epoch}')
        train_loop.set_postfix(loss=loss.item())

    # Validate
    model.eval()
    
    # Tracking variables for storing ground truth and predictions 
    predictions , true_labels = [], []

    # Prediction Loop
    valid_loop = tqdm(test_loader)
    for batch in valid_loop: 
        # Unpack the inputs from our dataloader and move to GPU/accelerator  
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Telling the model not to compute or store gradients, saving memory and 
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            output = model(input_ids, attention_mask, labels=labels)

        logits = output['logits']

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = labels.to('cpu').numpy()

        # Store predictions and true labels
        predictions.append(logits)
        true_labels.append(label_ids)

    from sklearn.metrics import classification_report, accuracy_score 

    # Combine the results across all batches. 
    flat_predictions = np.concatenate(predictions, axis=0)

    # For each sample, pick the label (0 or 1) with the higher score.
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

    # Combine the correct labels for each batch into a single list.
    flat_true_labels = np.concatenate(true_labels, axis=0)

    # Accuracy 
    acc = accuracy_score(flat_true_labels, flat_predictions)
    print(acc)

    # Classification Report
    report = classification_report(flat_true_labels, flat_predictions)
    print(report)

       
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    print(f'Finished using {epoch+1} epochs and LR of {LR}...\n\n')
    
    if acc > best_acc:
        best_acc = acc
        # Store model
        torch.save(model.state_dict(), f'out/model_dBERTforSeqClass_state_{acc}_{epoch+1}_{LR}.bin')
        print(f'Model Saved!')

        # Load test data for submission
        test_tweets = []
        predictions = []
        with open('twitter-datasets/test_data.txt', 'r', encoding='utf-8') as f:
            for line in f:
                test_tweets.append(line.rstrip())
        print(f'Predicting submission data...')   
        for i, tweet in enumerate(test_tweets):    
            if i % 500 == 0:
                print(f' -> Currently processing tweet {i}')
            encodings = tokenizer.encode_plus(
                tweet,
                add_special_tokens=True,
                max_length=64,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            with torch.no_grad():
                model.to('cpu')
                preds = model(encodings['input_ids'].to('cpu'), encodings['attention_mask'].to('cpu'))
                preds_argmax = np.argmax(preds['logits'])
                output = preds_argmax.item()
                predictions.append(output)
        model.to(device)
        
        ids = range(1,len(predictions)+1)
        out_dict = {
            'Id' : ids,
            'Prediction' : predictions
        }
        out = pd.DataFrame(out_dict)
        out = out.replace(0,-1)
        out.to_csv('out/submission_dBERTforSeqClass.csv', index=False)
        print(f'Done! Saved to out/submission_dBERTforSeqClass.csv')
