from sklearn.datasets.base import Bunch
import csv
import numpy as np
import pandas as pd

def pre(file, label):
    df = pd.read_csv(file)
    temp = df.shape
    n_samples = temp[0]
    n_features = temp[1] - 1
    def load_dataset(file, nrows, ncols):
        with open(file) as csv_file:
            data_file = csv.reader(csv_file)
            n_samples = nrows
            n_features = ncols
            data = np.empty((n_samples, n_features))
            target = np.empty((n_samples,), dtype=np.int)
            next(csv_file)
            for i, sample in enumerate(data_file):
                if label == 1:
                    data[i] = np.asarray(sample[:-1], dtype=np.float64)
                    target[i] = np.asarray(sample[-label], dtype=np.int)
                else:
                    data[i] = np.asarray(sample[1:], dtype=np.float64)
                    target[i] = np.asarray(sample[0], dtype=np.int)

        return Bunch(data=data, target=target, target_names=set(target))

    md = load_dataset(file, n_samples, n_features)
    return md