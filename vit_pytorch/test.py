import random
import numpy as np
from sklearn.metrics import confusion_matrix
import sklearn.model_selection
import seaborn as sns
import itertools
import spectral
import matplotlib.pyplot as plt
from scipy import io, misc
import os
import re
import torch

def train_test(train, test):
    # Use GPU ?
    train_gt = train
    test_gt = test
    p = train.shape[0] // 2
    q = train.shape[1] // 2
    for i in train.shape[0]:
        for j in train.shape[1]:
            if train[i][j] == 0:
                continue
            else:
                # train_gt[i-p, i + p][j-q, j + q] = test[i-p, i + p][j-q, j + q]
                test_gt[i-p, i + p][j-q, j + q] = 0


    return test_gt

