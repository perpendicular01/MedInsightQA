import pytorch_lightning as pl
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch import nn
import torch
import math
import functools
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, AutoTokenizer, AutoModel


class MCQAModel(pl.LightningModule):
    def __init__(self, model_name_or_path, args):
        super().__init__()
        self.args = args
        self.batch_size = self.args['batch_size']

        self.init_encoder_model(model_name_or_path)
        self.dropout = nn.Dropout(self.args['hidden_dropout_prob'])
        self.linear = nn.Linear(self.args['hidden_size'], 1)
        self.ce_loss = nn.CrossEntropyLoss()

        # Store val/test outputs
        self.val_outputs = {"logits": [], "labels": []}
        self.test_outputs = {"logits": [], "labels": []}

        self.save_hyperparameters()

    def init_encoder_model(self, model_name_or_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path)

    def prepare_dataset(self, train_dataset, val_dataset, test_dataset=None):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset if test_dataset else val_dataset

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_output = outputs[1]  # CLS token
        pooled_output = self.dropout(pooled_output)
        logits = self.linear(pooled_output)
        reshaped_logits = logits.view(-1, self.args['num_choices'])
        return reshaped_logits

    # -------------------------
    # Training Step
    # -------------------------
    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        for key in inputs:
            inputs[key] = inputs[key].to(self.args["device"])
        logits = self(**inputs)
        loss = self.ce_loss(logits, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
