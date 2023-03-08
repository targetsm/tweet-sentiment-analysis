# Based on https://skimai.com/fine-tuning-bert-for-sentiment-analysis/

import os
import re
import random
import time
import datetime
from datetime import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc 
from sklearn.model_selection import train_test_split
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, DistilBertTokenizerFast, DistilBertModel, get_linear_schedule_with_warmup, DistilBertConfig

if torch.cuda.is_available():
    torch.cuda.empty_cache()

nltk_stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "only", "own", "same", "so", "than", "too", "very", "s", "will", "just", "should", "now"]

#Params
MODEL_NAME = 'distilbert-base-uncased'
MODEL_PATH = 'out/model_'+MODEL_NAME+'_state.bin'
FREEZE = 3
EPOCHS = 5
SET = 'full'
SAMPLE = 0
SPLIT = 0.01
BATCH_SIZE = 16
MAX_LEN = 48
RM_STOPWORDS = True
LR = 3e-5
EPS = 1e-8
DISTILBERT_DROPOUT = 0.2
DISTILBERT_ATT_DROPOUT = 0.2

'''
data = []
for MODEL_NAME in ['distilbert-base-cased']:
    for MAX_LEN in [32,48,64]:
        for LR in [5e-5, 2e-5]:
            for RM_STOPWORDS in [True, False]:
                for DISTILBERT_DROPOUT in [0.1,0.2]:
                    DISTILBERT_ATT_DROPOUT = DISTILBERT_DROPOUT
                    print(f'\n\nPARAMETERS: {MODEL_NAME}, len={MAX_LEN}, lr={LR}, stopwords={RM_STOPWORDS}, dropout={DISTILBERT_DROPOUT}\n')
'''

# Dataset class definition
class TwitterDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df
        self.maxlen = MAX_LEN
        self.tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)  
        
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
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encodings.input_ids.flatten(),
            'attention_mask': encodings.attention_mask.flatten(),
            'labels': torch.tensor(sentiment, dtype=torch.long)
        }

def evaluate_roc(probs, y_true):
    """
    - Print AUC and accuracy on the test set
    - Plot ROC
    @params    probs (np.array): an array of predicted probabilities with shape (len(y_true), 2)
    @params    y_true (np.array): an array of the true values with shape (len(y_true),)
    """
    preds = probs[:, 1]
    fpr, tpr, threshold = roc_curve(y_true, preds)
    roc_auc = auc(fpr, tpr)
    print(f'AUC: {roc_auc:.4f}')
       
    # Get accuracy over the test set
    y_pred = np.where(preds >= 0.5, 1, 0)
    accuracy = accuracy_score(y_true, y_pred)
    print(f'Accuracy: {accuracy*100:.2f}%')
    
    # Plot ROC AUC
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig('out/distilbert-ROC.png')

# Create the DistilBertClassifier class
class DistilBertClassifier(nn.Module):
    """Bert Model for Classification Tasks.
    """
    def __init__(self, freeze_bert=True, config=None):
        """
        @param    bert: a BertModel object
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(DistilBertClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H1, H2, D_out = 768, 256, 32, 2
        dropout = 0.2

        # Instantiate BERT model
        self.bert = DistilBertModel.from_pretrained(MODEL_NAME, config=config)

        # Instantiate an one-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(H1, H2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(H2, D_out)
        )

        # Freeze the BERT model
        self.freeze(freeze_bert)
    
    def freeze(self, freeze=True):    
        if freeze:
            for param in self.bert.parameters():
                param.requires_grad = False
        else:
            for param in self.bert.parameters():
                param.requires_grad = True
            
            
    def forward(self, input_ids, attention_mask):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """
        # Feed input to BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]

        # Feed input to classifier to compute logits
        logits = self.classifier(last_hidden_state_cls)

        return logits
        
        
def text_preprocessing(text):
    """
    - Remove entity mentions (eg. '@united')
    - Correct errors (eg. '&amp;' to '&')
    @param    text (str): a string to be processed.
    @return   text (Str): the processed string.
    """    
    # Remove '@name'
    text = re.sub(r'(@.*?)[\s]', ' ', text)

    # Replace '&amp;' with '&'
    text = re.sub(r'&amp;', '&', text)

    # Remove trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# Create a function to tokenize a set of texts
def preprocessing_for_bert(data):
    """Perform required preprocessing steps for pretrained BERT.
    @param    data (np.array): Array of texts to be processed.
    @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
    @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                  tokens should be attended to by the model.
    """
    # Create empty lists to store outputs
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in data:
        # `encode_plus` will:
        #    (1) Tokenize the sentence
        #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
        #    (3) Truncate/Pad sentence to max length
        #    (4) Map tokens to their IDs
        #    (5) Create attention mask
        #    (6) Return a dictionary of outputs
        encoded_sent = tokenizer.encode_plus(
            text=text_preprocessing(sent),  # Preprocess sentence
            add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
            max_length=MAX_LEN,             # Max length to truncate/pad
            padding='max_length',           # Pad sentence to max length
            truncation=True,
            #return_tensors='pt',           # Return PyTorch tensor
            return_attention_mask=True      # Return attention mask
            )
        
        # Add the outputs to the lists
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks


def initialize_model(epochs=4):
    """Initialize the Bert Classifier, the optimizer and the learning rate scheduler.
    """
    size = 30522
    if MODEL_NAME == 'distilbert-base-cased': size = 28996
    # DistilBERT config
    config = DistilBertConfig(dropout=DISTILBERT_DROPOUT, 
                              attention_dropout=DISTILBERT_ATT_DROPOUT, 
                              output_hidden_states=True,
                              vocab_size=size
                          )
                          
    # Instantiate Bert Classifier
    dbert_classifier = DistilBertClassifier(freeze_bert=FREEZE>0, config=config)
    if FREEZE>0:
        print(f'\nNOTE: Freezing distilBERT layers for the first {FREEZE} of {EPOCHS} epochs.')

    # Tell PyTorch to run the model on GPU
    dbert_classifier.to(device)

    # Create the optimizer
    optimizer = optim.AdamW(dbert_classifier.parameters(),
                      lr=LR,     # Default learning rate
                      eps=EPS    # Default epsilon value
                      )

    # Total number of training steps
    total_steps = len(train_dataloader) * epochs

    # Set up the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0, # Default value
                                                num_training_steps=total_steps)
    return dbert_classifier, optimizer, scheduler
    
    
def set_seed(seed_value=42):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    
def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))
    
def train(model, train_dataloader, val_dataloader=None, epochs=4, evaluation=False):
    """Train the BertClassifier model.
    """
    t0_total = time.time()
    epoch_times = []
    freeze_factor = 4.15
    p_inter = max(min((int(len(train_dataloader)/200)*10+20), 1000), 20) # output interval, every p_inter batches
    
    # Start training loop
    print("\nStarting training...\n")
    for epoch_i in range(epochs):
        # =======================================
        #               Training
        # =======================================        
        
        # Potentially unfreeze model
        if epoch_i == FREEZE:
            print("Unfreezing distilBERT layers...")
            model.freeze(False)
            
        # Print the header of the result table
        print(f"{'Epoch':^7} | {'Batch':^17} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9} | {'Epoch:':^54} | {'Total:':^54}  |")
        print("-"*196)

        # Measure the elapsed time of each epoch
        t0_epoch, t0_batch = time.time(), time.time()

        # Reset tracking variables at the beginning of each epoch
        total_loss, batch_loss, batch_counts, time_elapsed = 0, 0, 0, 0
        
        # Put the model into the training mode
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            batch_counts +=1
            # Load batch to GPU
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

            # Zero out any previously calculated gradients
            model.zero_grad()

            # Perform a forward pass. This will return logits.
            logits = model(b_input_ids, b_attn_mask)

            # Compute loss and accumulate the loss values
            loss = loss_fn(logits, b_labels)
            batch_loss += loss.item()
            total_loss += loss.item()

            # Perform a backward pass to calculate gradients
            loss.backward()

            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and the learning rate
            optimizer.step()
            scheduler.step()

            # Print the loss values and time elapsed
            if (step % p_inter == 0 and step != 0) or (step == len(train_dataloader) - 1) or (step == 100 and p_inter > 200):
                # Timings
                t = time.time()
                time_elapsed = t - t0_batch
                
                elapse_epoch = t - t0_epoch
                elapse_total = t - t0_total
                
                percnt_epoch = step/len(train_dataloader)

                frozen_vsteps = len(train_dataloader)*FREEZE/freeze_factor
                full_vsteps = len(train_dataloader)*(EPOCHS-FREEZE)
                vsteps_total = frozen_vsteps + full_vsteps
                if epoch_i<FREEZE:
                    vsteps = (len(train_dataloader)*epoch_i+step)/freeze_factor
                else:
                    vsteps = frozen_vsteps + (len(train_dataloader)*(epoch_i-FREEZE)+step)
                percnt_total = vsteps/vsteps_total
                
                expect_epoch = elapse_epoch/percnt_epoch
                expect_total = elapse_total/percnt_total
                
                remain_epoch = expect_epoch - elapse_epoch
                remain_total = expect_total - elapse_total
                if remain_total < remain_epoch: remain_total = remain_epoch

                # Print training results
                print(f"{epoch_i + 1:^7} | {step:^7} / {len(train_dataloader):^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f} | {str(percnt_epoch*100)[:4]:^4}% {format_time(elapse_epoch):^12} / {format_time(expect_epoch):^12} (Rem.: {format_time(remain_epoch):^12}) | {str(percnt_total*100)[:4]:^4}%  {format_time(elapse_total):^12} / {format_time(expect_total):^12}  (Rem: {format_time(remain_total):^12}) |")

                # Reset batch tracking variables
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)

        print("-"*196)
        # =======================================
        #               Evaluation
        # =======================================
        if evaluation == True:
            # After the completion of each training epoch, measure the model's performance
            # on our validation set.
            val_loss, val_accuracy = evaluate(model, val_dataloader)

            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch
            
            print(f"{epoch_i + 1:^7} | {'-':^17} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f} |")
            print("-"*81)
        
        # Adjust freeze_factor
        epoch_times.append(time_elapsed)
        if epoch_i >= FREEZE and FREEZE > 0:
            freeze_factor = np.mean(epoch_times[FREEZE:])/np.mean(epoch_times[:FREEZE])
            
        
        # Store model
        torch.save(model.state_dict(), MODEL_PATH)
        print("Model state saved to file!\n")
 
    log = ""
    log += dt.now().strftime("%d/%m/%Y %H:%M:%S")
    log += f" | MODEL_NAME='{MODEL_NAME}'"
    log += f" | SET='{SET}'"
    log += f" | SAMPLE={SAMPLE}"
    log += f" | FREEZE={FREEZE}"
    log += f" | EPOCHS={EPOCHS}"
    log += f" | LR={LR}"
    log += f" | EPS={EPS}"
    log += f" | MAX_LEN={MAX_LEN}"
    log += f" | RM_STOPWORDS={RM_STOPWORDS}"
    log += f" | DISTILBERT_DROPOUT={DISTILBERT_DROPOUT}"
    log += f" | VAL_LOSS={val_loss}"
    log += f" | VAL_ACC={val_accuracy}"
    file = open('out/training.log', 'a')
    file.write(log+'\n')
    file.close()
    
    print(f'PARAMETERS: {MODEL_NAME}, len={MAX_LEN}, lr={LR}, stopwords={RM_STOPWORDS}, dropout={DISTILBERT_DROPOUT}\n')
    
    print("Training complete!")


def evaluate(model, val_dataloader):
    """After the completion of each training epoch, measure the model's performance
    on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    # Tracking variables
    val_accuracy = []
    val_loss = []

    # For each batch in our validation set...
    for batch in val_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)

        # Compute loss
        loss = loss_fn(logits, b_labels)
        val_loss.append(loss.item())

        # Get the predictions
        preds = torch.argmax(logits, dim=1).flatten()

        # Calculate the accuracy rate
        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        val_accuracy.append(accuracy)

    # Compute the average accuracy and loss over the validation set.
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)
    
    return val_loss, val_accuracy


def bert_predict(model, test_dataloader):
    """Perform a forward pass on the trained BERT model to predict probabilities
    on the test set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    all_logits = []
    
    # For each batch in our test set...
    for batch in test_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)[:2]

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
        all_logits.append(logits)   
    
    # Concatenate logits from each batch
    all_logits = torch.cat(all_logits, dim=0)

    # Apply softmax to calculate probabilities
    probs = F.softmax(all_logits, dim=1).cpu().numpy()

    return probs
    


# Load training data
tweets = []
labels = []

def load_tweets(filename, label):
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            if RM_STOPWORDS:
                tweets.append(" ".join([x for x in line.rstrip().split(" ") if x not in nltk_stopwords]))
            else:
                tweets.append(line.rstrip())
            labels.append(label)

if SET=='full':
    print('\nUsing full dataset.')
    load_tweets('twitter-datasets/train_neg_full.txt', 0)
    load_tweets('twitter-datasets/train_pos_full.txt', 1)
else:
    print('\nUsing small dataset.')
    load_tweets('twitter-datasets/train_neg.txt', 0)
    load_tweets('twitter-datasets/train_pos.txt', 1)

# Convert to dataframe
data_dict = {
    'tweet': tweets,
    'label': labels
}
df = pd.DataFrame(data_dict)
print(f'{len(df)} tweets loaded.')

if SAMPLE>0:
    print(f'Sampling {SAMPLE} tweets.')
    data = df.sample(n=SAMPLE)
else:
    data = df

print(f'Using {len(data)} tweets:')
print(data)

        
# Split into train and validation datasets

X = data.tweet.values
y = data.label.values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=SPLIT, random_state=42)


# Load test data
test_tweets = []
with open('twitter-datasets/test_data.txt', 'r', encoding='utf-8') as f:
    for line in f:
        if RM_STOPWORDS:
            test_tweets.append(" ".join([x for x in line.rstrip().split(" ") if x not in nltk_stopwords]))
        else:
            test_tweets.append(line.rstrip())
test_dict = { 'tweet' : test_tweets }
test_data = pd.DataFrame(test_dict)




# Load the DistilBERT tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)#, do_lower_case=True)

'''
# Concatenate train data and test data
all_tweets = np.concatenate([data.tweet.values, test_data.tweet.values])

# Encode our concatenated data
encoded_tweets = [tokenizer.encode(sent, add_special_tokens=True) for sent in all_tweets]

# Find the maximum length
max_len = max([len(sent) for sent in encoded_tweets])
print('Max length: ', max_len)
'''


'''
# Print sentence 0 and its encoded token ids
token_ids = list(preprocessing_for_bert([X[0]])[0].squeeze().numpy())
print('Original: ', X[0])
print('Token IDs: ', token_ids)
'''

# Run function `preprocessing_for_bert` on the train set and the validation set
print('\nTokenizing data:')
print(' -> Training set...')
train_inputs, train_masks = preprocessing_for_bert(X_train)
print(' -> Validation set...')
val_inputs, val_masks = preprocessing_for_bert(X_val)

## Create PyTorch DataLoader

# Convert other data types to torch.Tensor
train_labels = torch.tensor(y_train)
val_labels = torch.tensor(y_val)

# Create the DataLoader for our training set
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

# Create the DataLoader for our validation set
val_data = TensorDataset(val_inputs, val_masks, val_labels)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=BATCH_SIZE)


# Generate model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
set_seed(42)    # Set seed for reproducibility
dbert_classifier, optimizer, scheduler = initialize_model(epochs=EPOCHS)

# Specify loss function
loss_fn = nn.CrossEntropyLoss()#.to(device)

# Train model
train(dbert_classifier, train_dataloader, val_dataloader, epochs=EPOCHS, evaluation=True)

# Store model
torch.save(dbert_classifier.state_dict(), MODEL_PATH)

# Compute predicted probabilities on the validation set
probs = bert_predict(dbert_classifier, val_dataloader)

# Evaluate the Bert classifier
evaluate_roc(probs, y_val)


## Predict test data
print('\nPrediction of test set:')

# Run preprocessing on the test set
print('Tokenizing data...')
test_inputs, test_masks = preprocessing_for_bert(test_data.tweet)

# Create the DataLoader for our test set
test_dataset = TensorDataset(test_inputs, test_masks)
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=BATCH_SIZE)

# Compute predicted probabilities on the test set
probs = bert_predict(dbert_classifier, test_dataloader)

# Get predictions from the probabilities
threshold = 0.5
preds = np.where(probs[:, 1] > threshold, 1, 0)

# Number of tweets predicted non-negative
print("Number of tweets predicted positive: ", preds.sum())
print("Number of tweets predicted negative: ", len(preds)-preds.sum())

ids = range(1,len(preds)+1)
out_dict = {
    'Id' : ids,
    'Prediction' : probs[:, 1]
}
out = pd.DataFrame(out_dict)
out = out.replace(0,-1)
out.to_csv('out/probabilities.csv', index=False)

out_dict = {
    'Id' : ids,
    'Prediction' : preds
}
out = pd.DataFrame(out_dict)
out = out.replace(0,-1)
out.to_csv('out/submission.csv', index=False)

