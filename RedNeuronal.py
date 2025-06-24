###MLP###
import pandas as pd
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


def MLP_HP(X1,y1,param_grid):
    from sklearn.model_selection import GridSearchCV
    import numpy as np

    X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.2, random_state=17,stratify=y1)
    grid_search = GridSearchCV(MLPClassifier(max_iter=1000, random_state=13), param_grid, cv=5,scoring='f1_macro')
    grid_search.fit(X_train, y_train)

    print(f"Mejores hiperparámetros: {grid_search.best_params_}")
    print(f"Mejor score: {grid_search.best_score_}")

    best_MLPTrain = grid_search.best_estimator_
    best_MLPTrain.fit(X_train,y_train)

    y_kknpred1=best_MLPTrain.predict(X_test)
    score = precision_score(y_kknpred1,y_test,average = None)
    print(score)

    best_MLP = grid_search.best_estimator_
    best_MLP.fit(X1,y1)

    return best_MLP, grid_search.best_score_,grid_search.best_params_