from sklearn import datasets

from Exportproba import exportproba
from Pre import pre
from vote import vote

if __name__ == '__main__':

    iris = datasets.load_iris()
    wine = datasets.load_wine()
    wine2 = pre('./datasets/winequality-red.csv',1)
    handwrite = datasets.load_digits()
    glass = pre('./datasets/glass.csv',1)
    ######################################
    result = exportproba(glass)
    print(vote(result))