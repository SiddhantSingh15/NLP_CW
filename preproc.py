import pandas as pd
from dont_patronize_me import DontPatronizeMe
import numpy as np
import os

os.chdir("/vol/bitbucket/ss5120/NLP_CW")


class PreProcessor:
    def __init__(self):
        self.data_path = "data/"
        self.trids = pd.read_csv(
            self.data_path + "train_semeval_parids-labels.csv"
        )
        self.teids = pd.read_csv(
            self.data_path + "dev_semeval_parids-labels.csv"
        )

        self.trids.par_id = self.trids.par_id.astype(str)
        self.teids.par_id = self.teids.par_id.astype(str)

        self.dpm = DontPatronizeMe(".", ".")
        self.dpm.load_task1()

        self.data = self.dpm.train_task1_df

        self.raw_train = self.rebuild_train(self.trids, self.data)
        self.raw_dev = self.rebuild_dev(self.teids, self.data)
        self.dev_labels = np.array(self.raw_dev['label'])

    def rebuild_train(self, trids, data):
        rows = []  # will contain par_id, label and text
        for idx in range(len(trids)):
            parid = trids.par_id[idx]
            # print(parid)
            # select row from original dataset to retrieve `text` and binary label
            keyword = data.loc[data.par_id == parid].keyword.values[0]
            text = data.loc[data.par_id == parid].text.values[0]
            label = data.loc[data.par_id == parid].label.values[0]
            rows.append(
                {
                    "par_id": parid,
                    "community": keyword,
                    "text": text,
                    "label": label,
                }
            )

        return pd.DataFrame(rows)

    def rebuild_dev(self, teids, data):
        rows = []  # will contain par_id, label and text
        for idx in range(len(teids)):
            parid = teids.par_id[idx]
            # print(parid)
            # select row from original dataset
            keyword = data.loc[data.par_id == parid].keyword.values[0]
            text = data.loc[data.par_id == parid].text.values[0]
            label = data.loc[data.par_id == parid].label.values[0]
            rows.append(
                {
                    "par_id": parid,
                    "community": keyword,
                    "text": text,
                    "label": label,
                }
            )

        return pd.DataFrame(rows)

    def downsample_neg(self):
        # downsample negative instances
        pcldf = self.raw_train[self.raw_train.label == 1]
        npos = len(pcldf)

        downsampled_df = pd.concat(
            [pcldf, self.raw_train[self.raw_train.label == 0][: npos * 2]]
        )

        return downsampled_df


if __name__ == "__main__":
    pp = PreProcessor()
    print(pp.raw_train.head())
    print(pp.raw_dev.head())
