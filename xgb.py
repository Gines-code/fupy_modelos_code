import xgboost as xgb
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import random
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score



"""
EJEMPLO HP
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
"""


def XGB_HP(X,y,param_grid):
    """
    Busqueda de hiperparametros para el modelo XGB
    """

    # Dividir el conjunto de datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Inicializar el modelo de Gradient Boosting
    model = GradientBoostingClassifier(random_state=42)

    # Realizar la búsqueda de hiperparámetros
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Mostrar los mejores hiperparámetros
    print("Mejores hiperparámetros encontrados:")
    print(grid_search.best_params_)


    # Utilizar el modelo con los mejores hiperparámetros para hacer predicciones
    best_model = grid_search.best_estimator_

    return best_model,grid_search.best_score_, grid_search.best_params_