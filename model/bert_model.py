import torch.nn as nn
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)
from datasets import Dataset
import wandb

wandb.init(mode="disabled")


class Model(nn.Module):
    def __init__(self, model_type, layers_to_freeze=0, num_train_epochs=1):
        super(Model, self).__init__()

        self.training_args = TrainingArguments(
            report_to="none",
            output_dir="./results",
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir="./logs",
            evaluation_strategy="epoch",
        )
        self.id2label = {0: "NEGATIVE", 1: "POSITIVE"}
        self.label2id = {"NEGATIVE": 0, "POSITIVE": 1}

        self.trainer = None

        if model_type == "baseline":
            self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
            self.model = RobertaForSequenceClassification.from_pretrained(
                "roberta-base",
                num_labels=2,
                id2label=self.id2label,
                label2id=self.label2id,
            )

        for param in self.model.base_model.parameters():
            if layers_to_freeze > 0:
                param.requires_grad = False
                layers_to_freeze -= 1

    def apply_tokenizer(self, batch):
        return self.tokenizer(
            batch["text"],
            truncation=True,
            padding=True,
            max_length=100,
            add_special_tokens=True,
        )

    def train(self, train_df, dev_df):
        train_hf = Dataset.from_pandas(train_df)
        dev_hf = Dataset.from_pandas(dev_df)

        tokenized_train = train_hf.map(self.apply_tokenizer, batched=True)
        tokenized_dev = dev_hf.map(self.apply_tokenizer, batched=True)

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            tokenizer=self.tokenizer,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_dev,
            # data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )

        self.trainer.train()

    def inference(self, df):
        input_hf = Dataset.from_pandas(df)
        tokenized_input = input_hf.map(self.apply_tokenizer, batched=True)
        predictions = self.trainer.predict(tokenized_input)
        return predictions.predictions

    def evaluate_train(self, train_df):
        input_hf = Dataset.from_pandas(train_df)
        tokenized_input = input_hf.map(self.apply_tokenizer, batched=True)
        return self.trainer.evaluate(tokenized_input)

    def evaluate_dev(self):
        return self.trainer.evaluate()

    def compute_metrics(self, pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary"
        )
        acc = accuracy_score(labels, preds)
        return {
            "accuracy": acc,
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }

    def labels2file(self, p, outf_path):
        with open(outf_path, "w") as outf:
            for pi in p:
                outf.write(",".join([str(k) for k in pi]) + "\n")
