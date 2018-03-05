import numpy as np
import pandas as pd


def create_data_subset(train_subset_size, valid_subset_size, test_subset_size):
    if train_subset_size + valid_subset_size + test_subset_size < 50000:

        train_x = pd.read_csv("./data/train_x.csv", header=None, nrows=train_subset_size)
        train_y = pd.read_csv("./data/train_y.csv", header=None, nrows=train_subset_size)
        valid_x = pd.read_csv("./data/train_x.csv", header=None, nrows=valid_subset_size, skiprows=train_subset_size)
        valid_y = pd.read_csv("./data/train_y.csv", header=None, nrows=valid_subset_size, skiprows=train_subset_size)
        test_x = pd.read_csv("./data/train_x.csv", header=None, nrows=train_subset_size, skiprows=train_subset_size + valid_subset_size)
        test_y = pd.read_csv("./data/train_y.csv", header=None, nrows=train_subset_size, skiprows=train_subset_size + valid_subset_size)
        print("files loaded")

        train_x.to_csv("./data_subset/train_x_{}.csv".format(train_subset_size), header=None, index=None)
        train_y.to_csv("./data_subset/train_y_{}.csv".format(train_subset_size), header=None, index=None)
        print("train set loaded")

        valid_x.to_csv("./data_subset/valid_x_{}.csv".format(valid_subset_size), header=None, index=None)
        valid_y.to_csv("./data_subset/valid_y_{}.csv".format(valid_subset_size), header=None, index=None)
        print("valid set loaded")

        test_x.to_csv("./data_subset/test_x_{}.csv".format(test_subset_size), header=None, index=None)
        test_y.to_csv("./data_subset/test_y_{}.csv".format(test_subset_size), header=None, index=None)
        print("test set written")

    else:
        print("number of data points too large")
        print("sum of each part larger than 50 000")

create_data_subset(2000, 1000, 500)