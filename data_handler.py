import numpy as np
from os.path import abspath

class DataHandler():
    URL = ""

    def __init__(self, URL):
        self.URL = abspath(URL)

    def zscore_data(self, X):
        means = np.zeros((1, X.shape[1]))
        stds = np.zeros((1, X.shape[1]))
        for i in range(X.shape[1]):
            data_ = X[:, i]
            mean = np.mean(data_)
            means[:, i] = mean
            std = np.std(data_, ddof=1)
            stds[:, i] = std
        return means, stds

    def apply_zscore(self, means, stds, X):
        return (X - means)/stds
    
    def bin_features(self, X):
        XMean = np.mean(X, axis=0)
        for i in range(X.shape[1]):
            X_ = X[:, i]
            for j in range(len(X_)):
                if X_[j] > XMean[i]:
                    X_[j] = 1
                else:
                    X_[j] = 0
            X[:, i] = X_
        return X
    
    def class_feature(self, X):
        XMean = np.mean(X, axis=0)
        for i in range(X.shape[1]):
            XMean = np.mean(X[:, i], axis=0)
            unique_features = np.unique(X[:, i])
            means = []
            XMean_ = XMean / len(unique_features)
            X[:, i][X[:, i] > XMean_] = 1
            X[:, i][X[:, i] > XMean_] = 2
            X[:, i][X[:, i] < XMean_*3] = 3
        return X

    def parse_data_no_header(self):
        return np.genfromtxt(
            self.URL, delimiter=","
        )
    
    def parse_data(self):
        return np.loadtxt(
            self.URL, delimiter=",", skiprows=2
        )

    def shuffle_data(self, data, seed=0):
        np.random.seed(seed)
        # Shuffle Data
        np.random.shuffle(data)
        return data
    
    def split_data(self, data):
        # Create Arrays for Training vs Validation
        training_index = round(len(data)*2/3)
        train = data[0:training_index]
        validation_index = training_index+1
        validation = data[validation_index:]
        return train, validation

def get_xy(data, label_index=0):
    return data[:, label_index+1:], data[:, label_index]