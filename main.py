from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score

from Pre import pre
from Exportproba import exportproba
from vote import vote
if __name__ == '__main__':

    iris = datasets.load_iris()
    wine = datasets.load_wine()
    wine2 = pre('./datasets/winequality-red.csv',1)
    handwrite = datasets.load_digits()
    glass = pre('./datasets/glass.csv',1)
    ######################################
    result = exportproba(iris)
    print(vote(result))
    result = exportproba(glass)
    print(vote(result))