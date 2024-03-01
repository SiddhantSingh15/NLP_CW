import torch
from simpletransformers.classification import (
    ClassificationModel,
    ClassificationArgs,
)
from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    precision_score,
    recall_score,
    accuracy_score,
)


class Model:
    def __init__(self, model_type):
        self.cuda_available = torch.cuda.is_available()
        self.device = "cuda" if self.cuda_available else "cpu"
        print(f"Device: {self.device}")

        baseline_model_args = ClassificationArgs(
            num_train_epochs=1,
            no_save=True,
            no_cache=True,
            overwrite_output_dir=True,
        )

        if model_type == "roberta-base":
            self.model = ClassificationModel(
                "roberta",
                'roberta-base',
                args=baseline_model_args,
                num_labels=2,
                use_cuda=self.cuda_available,
            )

    def train(self, train_df):
        self.model.train_model(train_df[["text", "label"]])

    def predict(self, dev_df):
        preds, _ = self.model.predict(dev_df.text.tolist())
        return preds

    def evaluation_metrics(self, true_labels, pred_labels):
        F1_score = f1_score(true_labels, pred_labels)
        precision_sc = precision_score(true_labels, pred_labels)
        conf_mat = confusion_matrix(true_labels, pred_labels)
        recall_sc = recall_score(true_labels, pred_labels)
        accuracy_sc = accuracy_score(true_labels, pred_labels)
        print(f"This is the positive f1 score: {F1_score}")
        print(f"This is the precison score: {precision_sc}")
        print(f"This is the recall score: {recall_sc}")
        print(f"This is the accuracy score: {accuracy_sc}")
        print(f"This is the confusion matrix:\n {conf_mat}")

    def labels2file(self, p, outf_path):
        with open(outf_path, "w") as outf:
            for pi in p:
                outf.write(",".join([str(k) for k in pi]) + "\n")
