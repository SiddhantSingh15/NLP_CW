import torch
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)
import wandb

wandb.init(mode="disabled")


class Model:
    def __init__(self, model_type):
        self.cuda_available = torch.cuda.is_available()
        self.device = "cuda" if self.cuda_available else "cpu"
        print(f"Device: {self.device}")
        self.training_args = TrainingArguments(
            report_to="none",
            output_dir="./results",
            num_train_epochs=1,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir="./logs",
            evaluation_strategy="epoch",
        )

        self.trainer = None

        if model_type == "bert-base-cased":
            self.tokenizer = BertTokenizer.from_pretrained(model_type)
            self.model = BertForSequenceClassification.from_pretrained(
                model_type
            )
            self.model.to(self.device)

    def tokenize(self, batch):
        return self.tokenizer(batch["text"], padding=True, truncation=True)

    def train(self, train_df, dev_df):
        train_dataset = train_df.map(
            self.tokenize, batched=True, batch_size=len(train_df)
        )
        test_dataset = dev_df.map(
            self.tokenize, batched=True, batch_size=len(dev_df)
        )

        train_dataset.set_format(
            "torch", columns=["input_ids", "attention_mask", "label"]
        )
        test_dataset.set_format(
            "torch", columns=["input_ids", "attention_mask", "label"]
        )

        print(train_dataset.head())
        print(test_dataset.head())

        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=self.compute_metrics,
        )

        self.trainer.train()

    def evaluate(self):
        return self.model.evaluate()

    def compute_metrics(pred):
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
