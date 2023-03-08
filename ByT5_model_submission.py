import argparse
import random
import pandas as pd
import numpy as np
import torch
from TwitterDataset import TwitterDataset
from T5FineTuner import T5FineTuner
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from transformers import (
    AutoTokenizer
)

from tqdm.auto import tqdm
from sklearn import metrics

import uuid


class ByT5_model:
    def __init__(self):
        def set_seed(seed):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        set_seed(42)

        self.args_dict = dict(
            output_dir="",  # path to save the checkpoints
            #model_name_or_path='google/byt5-small',
            model_name_or_path='./models/t5_base_afdaa4da',
            tokenizer_name_or_path='google/byt5-small',
            max_seq_length=300,
            learning_rate=3e-5,
            weight_decay=0.0,
            adam_epsilon=1e-8,
            warmup_steps=0,
            train_batch_size=2,
            eval_batch_size=2,
            gradient_accumulation_steps=16,
            n_gpu=1,
            early_stop_callback=False,
            fp_16=False,
            max_grad_norm=1.0,  # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
            seed=42,
        )
        self.tokenizer = AutoTokenizer.from_pretrained('google/byt5-small')
        self.generate_dataset()

    def generate_dataset(self):
        with open('./twitter-datasets/train_neg.txt') as f:
            neg_lines = f.readlines()

        with open('./twitter-datasets/train_pos.txt') as f:
            pos_lines = f.readlines()

        df = pd.DataFrame(neg_lines, columns=['text'])
        df["label"] = [0] * len(neg_lines)

        df_pos = pd.DataFrame(pos_lines, columns=['text'])
        df_pos["label"] = [1] * len(pos_lines)
        df = df.append(df_pos, ignore_index=True)
        df = df.sample(frac=1).reset_index(drop=True)
        self.custom_dataset = {"train": df[:-10000], "test": df[-10000:]}

    def train(self):
        self.args_dict.update(
            {'output_dir': 't5_imdb_sentiment', 'num_train_epochs': 5, 'vocab_file': 'tokenizer_config.json'})
        self.args = argparse.Namespace(**self.args_dict)

        train_params = dict(
            accumulate_grad_batches=self.args.gradient_accumulation_steps,
            gpus=self.args.n_gpu,
            max_epochs=self.args.num_train_epochs,
            # early_stop_callback=False,
            precision=16 if self.args.fp_16 else 32,
            gradient_clip_val=self.args.max_grad_norm,
        )

        def get_dataset(tokenizer, type_path, args):
            return TwitterDataset(tokenizer, self.custom_dataset, type_path, ['negative</s>', 'positive</s>'], max_len=300)

        self.model = T5FineTuner(self.args, get_dataset)
        trainer = pl.Trainer(**train_params)
        trainer.fit(self.model)
        # add this in?:
        # from pytorch_lightning.callbacks.early_stopping import EarlyStopping
        # callbacks=[EarlyStopping(monitor='val_loss')]

        save_path = './models/t5_base_' + str(uuid.uuid4())[:8]

        self.model.model.save_pretrained(save_path)
        print("Model saved under ", save_path)
        self.model.model.eval()
        self.validate()

    def validate(self):
        dataset = TwitterDataset(self.tokenizer, self.custom_dataset, 'test', ['negative</s>', 'positive</s>'], max_len=300)
        loader = DataLoader(dataset, batch_size=64, num_workers=0, shuffle=False)
        self.model.model.eval()
        outputs = []
        targets = []
        cudamodel = self.model.model.to('cuda')
        for batch in tqdm(loader):
            outs = cudamodel.generate(input_ids=batch['source_ids'].cuda(),
                                      attention_mask=batch['source_mask'].cuda(),
                                      max_length=11)

            dec = [self.tokenizer.decode(ids[1:]) for ids in outs]
            target = [self.tokenizer.decode(ids[:-1]) for ids in batch["target_ids"]]

            outputs.extend(dec)
            targets.extend(target)
        print("Accuracy score: ", metrics.accuracy_score(targets, outputs))
        print(metrics.classification_report(targets, outputs))
        metrics.ConfusionMatrixDisplay(metrics.confusion_matrix(targets, outputs)).plot()


    def generate_predictions(self):


        with open('./twitter-datasets/test_data.txt') as f:
            test_lines = f.readlines()
        df = pd.DataFrame(test_lines, columns=['text'])
        df["label"] = [0] * len(test_lines)

        custom_dataset = {"test": df}
        dataset = TwitterDataset(self.tokenizer, custom_dataset, 'test', ['negative</s>', 'positive</s>'], max_len=300)
        loader = DataLoader(dataset, batch_size=8, num_workers=0, shuffle=False)

        self.args_dict.update(
            {'output_dir': 't5_imdb_sentiment', 'num_train_epochs': 1, 'vocab_file': 'tokenizer_config.json'})
        self.args = argparse.Namespace(**self.args_dict)

        def get_dataset(tokenizer, type_path, args):
            return TwitterDataset(tokenizer, custom_dataset, type_path, ['negative</s>', 'positive</s>'],
                                  max_len=300)

        self.model = T5FineTuner(self.args, get_dataset)

        self.model.model.eval()
        outputs = []
        cudamodel = self.model.model.to('cuda')
        for batch in tqdm(loader):
            outs = cudamodel.generate(input_ids=batch['source_ids'].cuda(),
                                      attention_mask=batch['source_mask'].cuda(),
                                      max_length=11)

            dec = [self.tokenizer.decode(ids[1:]) for ids in outs]
            outputs.extend(dec)
        preds = [-1 if x == "negative</s>" else 1 for x in outputs]
        ids = range(1, len(preds) + 1)
        out_dict = {
            'Id': ids,
            'Prediction': preds
        }
        out = pd.DataFrame(out_dict)
        out.to_csv('submission.csv', index=False)




if __name__ == '__main__':
    byt5 = ByT5_model()
    #byt5.train()
    byt5.generate_predictions()
