
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, DistilBertModel


def model_from_file(name):
    print(f'Loading model from file: {MODEL_PATH}\n')
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME)
    model.load_state_dict(torch.load(MODEL_PATH))
    model = model.to(device)
    return model

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
    
    
MODEL_NAME = 'distilbert-base-uncased'
MODEL_PATH = 'out/model_dBERTforSeqClass_state_0.90172_2_1e-05.bin'
OUTPT_PATH = MODEL_PATH + '_predictions.csv'

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# tokenizer
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model = model_from_file(MODEL_NAME)

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
        
ids = range(1,len(predictions)+1)
out_dict = {
    'Id' : ids,
    'Prediction' : predictions
}
out = pd.DataFrame(out_dict)
out = out.replace(0,-1)
out.to_csv(OUTPT_PATH, index=False)
print(f'Done! Saved to '+OUTPT_PATH)