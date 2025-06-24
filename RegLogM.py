############################ REGRESION LOGISTICA MULTINOMIAL #####################

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import random
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score


def modelo_RL_HP(X_train,y_train,random_state,param_grid):
    """
    Funcion para realizar el Random forest con los mejores hiperparametros proporcionando un param_grid
    """
    model_rl = LogisticRegression(max_iter=500)

    grid_search = GridSearchCV(estimator=model_rl, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2,scoring='f1_macro')

    grid_search.fit(X_train, y_train)

    print("Mejores hiperparámetros:", grid_search.best_params_)
    best_model = grid_search.best_estimator_

    return best_model,grid_search.best_score_, grid_search.best_params_