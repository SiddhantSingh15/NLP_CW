from sklearn.metrics import precision_recall_fscore_support, accuracy_score


class EnsembleModel:
    def __init__(self, dev_dataset):
        self.dev_dataset = dev_dataset
        self.model_list = []
        self.train_set_list = []

    def add_model(self, model, train_set):
        self.model_list.append(model)
        self.train_set_list.append(train_set)

    def train(self):
        for model, train_set in zip(self.model_list, self.train_set_list):
            model.train(train_set, self.dev_dataset)

    def majority_vote(self, test_data):
        preds = [model.inference(test_data) for model in self.model_list]
        majority_voted = []
        for i in range(len(test_data)):
            votes = [pred[i].argmax(-1) for pred in preds]
            majority = max(set(votes), key=votes.count)
            majority_voted.append(majority)
        return majority_voted

    def compute_metrics(self, preds, test_labels):
        labels = test_labels
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
