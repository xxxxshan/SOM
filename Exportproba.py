#algotithmes:
import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV#?
#...
def exportproba(dataset):
    X_train,X_test,Y_train,Y_test = train_test_split(dataset.data,dataset.target,
                                                    test_size = 0.3, random_state = 0)
    ss = MinMaxScaler()
    #ss = StandardScaler() # error when MultinomialNB()
    X_train = ss.fit_transform(X_train)
    X_test = ss.fit_transform(X_test)

    clfs = [SVC(kernel= 'linear',probability=True),MultinomialNB(),KNeighborsClassifier(),
       DecisionTreeClassifier(max_depth=len(dataset.target_names)),RandomForestClassifier(),
       GradientBoostingClassifier()]
    # replaced linearsvc with svc
    #probability = np.zeros([54,1])
    df = pd.DataFrame()
    names = ['SVC','MultinomialNB','KNeighborsClassifier',
         'DecisionTreeClassifier','RandomForestClassifier',
       'GradientBoostingClassifier']
    print(dataset.target_names)
    best = 0
    for model,name in zip(clfs,names):
        print('model: \n ')
        print(str(model))
        model.fit(X_train,Y_train)
        print('Accurary: ')
        accurary = model.score(X_test,Y_test)
        print(accurary)
        if accurary>=best:
            best = accurary
        print()
        #print("Decision function :\n{}".format(model.decision_function(X_test)[:3]))
        print("Predicted probabilities :\n{}".format(model.predict_proba(X_test)[:3]))
        print('-------------------------')
        print()
        proba = model.predict_proba(X_test)
        #probability = np.hstack((probability,proba))
        #print(probability[0,:])
        df[name] = list(proba)
    df['label'] = list(Y_test)
    print('Best accurary: ', best)
    return df