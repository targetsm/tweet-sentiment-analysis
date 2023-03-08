# Code adapted from https://colab.research.google.com/drive/1syXmhEQ5s7C59zU8RtHVru0wAvMXTSQ8

from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl


from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    get_linear_schedule_with_warmup
)

class T5FineTuner(pl.LightningModule):
    def __init__(self, hparams, get_dataset):
        super(T5FineTuner, self).__init__()
        # self.save_hyperparameters()
        self.myparams = hparams

        self.model = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(hparams.tokenizer_name_or_path)
        self.get_dataset = get_dataset

    def is_logger(self):
        return True  # self.trainer.proc_rank <= 0

    def forward(
            self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, labels=None
    ):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

    def _step(self, batch):
        labels = batch["target_ids"]
        labels[labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=labels,
            decoder_attention_mask=batch['target_mask']
        )

        loss = outputs[0]

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)

        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tensorboard_logs = {"avg_train_loss": avg_train_loss}
        # {"avg_train_loss": avg_train_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}
        return None

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.myparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.myparams.learning_rate, eps=self.myparams.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    # handled automatically by PyTorch Lightning
    # def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
    #   if self.trainer.use_tpu:
    #     xm.optimizer_step(optimizer)
    #   else:
    #     optimizer.step()
    #   optimizer.zero_grad()
    #   self.lr_scheduler.step()

    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.3f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}

        return tqdm_dict

    def train_dataloader(self):
        train_dataset = self.get_dataset(tokenizer=self.tokenizer, type_path="train", args=self.myparams)
        dataloader = DataLoader(train_dataset, batch_size=self.myparams.train_batch_size, drop_last=True, shuffle=True,
                                num_workers=0)
        t_total = (
                (len(dataloader.dataset) // (self.myparams.train_batch_size * max(1, self.myparams.n_gpu)))
                // self.myparams.gradient_accumulation_steps
                * float(self.myparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.myparams.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        val_dataset = self.get_dataset(tokenizer=self.tokenizer, type_path="test", args=self.myparams)
        return DataLoader(val_dataset, batch_size=self.myparams.eval_batch_size, num_workers=0, shuffle=False)