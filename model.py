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

    # -------------------------
    # Validation Step
    # -------------------------
    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        for key in inputs:
            inputs[key] = inputs[key].to(self.args["device"])
        logits = self(**inputs)
        loss = self.ce_loss(logits, labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        self.val_outputs["logits"].append(logits.detach())
        self.val_outputs["labels"].append(labels.detach())
        return loss

    def on_validation_epoch_end(self):
        logits = torch.cat(self.val_outputs["logits"], dim=0)
        labels = torch.cat(self.val_outputs["labels"], dim=0)
        predictions = torch.argmax(logits, dim=-1)
        accuracy = (predictions == labels).float().mean()
        self.log("val_acc", accuracy, prog_bar=True)
        self.val_outputs = {"logits": [], "labels": []}

    # -------------------------
    # Test Step
    # -------------------------
    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        for key in inputs:
            inputs[key] = inputs[key].to(self.args["device"])
        logits = self(**inputs)

        # Compute loss only if labels exist
        if labels is not None:
            loss = self.ce_loss(logits, labels)
            self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            self.test_outputs["labels"].append(labels.detach())

        self.test_outputs["logits"].append(logits.detach())
        return logits

    def on_test_epoch_end(self):
        if self.test_outputs["labels"]:
            logits = torch.cat(self.test_outputs["logits"], dim=0)
            labels = torch.cat(self.test_outputs["labels"], dim=0)
            predictions = torch.argmax(logits, dim=-1)
            accuracy = (predictions == labels).float().mean()
            self.log("test_acc", accuracy, prog_bar=True)
        self.test_outputs = {"logits": [], "labels": []}

    # -------------------------
    # Optimizer and Scheduler
    # -------------------------
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.args['learning_rate'], eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=100,
            num_training_steps=(self.args['num_epochs'] + 1) * math.ceil(len(self.train_dataset) / self.args['batch_size'])
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    # -------------------------
    # DataLoader Helpers
    # -------------------------
    def process_batch(self, batch, tokenizer, max_len=32):
        expanded_batch = []
        labels = []

        for data_tuple in batch:
            if len(data_tuple) == 4:  # context + question + options + label
                context, question, options, label = data_tuple
            elif len(data_tuple) == 3:  # question + options + label (no context)
                context = None
                question, options, label = data_tuple
            elif len(data_tuple) == 2:  # question + options (test dataset)
                context = None
                question, options = data_tuple
                label = None
            else:
                raise ValueError(f"Unexpected data tuple length: {len(data_tuple)}")

            # Prepare question-option pairs
            question_option_pairs = [question + ' ' + option for option in options]

            if label is not None:
                labels.append(label)

            if context:
                contexts = [context] * len(options)
                expanded_batch.extend(zip(contexts, question_option_pairs))
            else:
                expanded_batch.extend(question_option_pairs)

        tokenized_batch = tokenizer.batch_encode_plus(
            expanded_batch,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt"
        )

        # Move everything to the correct device
        device = self.args["device"]
        tokenized_batch = {k: v.to(device) for k, v in tokenized_batch.items()}

        if labels:
            return tokenized_batch, torch.tensor(labels, device=device)
        else:
            return tokenized_batch, None

    def train_dataloader(self):
        train_sampler = RandomSampler(self.train_dataset)
        collate_fn = functools.partial(self.process_batch, tokenizer=self.tokenizer, max_len=self.args['max_len'])
        return DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=train_sampler, collate_fn=collate_fn)

    def val_dataloader(self):
        sampler = SequentialSampler(self.val_dataset)
        collate_fn = functools.partial(self.process_batch, tokenizer=self.tokenizer, max_len=self.args['max_len'])
        return DataLoader(self.val_dataset, batch_size=self.batch_size, sampler=sampler, collate_fn=collate_fn)

    def test_dataloader(self):
        sampler = SequentialSampler(self.test_dataset)
        collate_fn = functools.partial(self.process_batch, tokenizer=self.tokenizer, max_len=self.args['max_len'])
        return DataLoader(self.test_dataset, batch_size=self.batch_size, sampler=sampler, collate_fn=collate_fn)
