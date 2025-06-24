##############SCRIPT CON UN RANDOM FOREST PARA PREDICCION################

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from sklearn.model_selection import GridSearchCV


"""
EJEMPLO DE HIPERPARAMETROS A PROPORCIONAR

param_dist = {
    'n_estimators': randint(50, 200),  # Valores entre 50 y 200
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': randint(2, 11),
    'min_samples_leaf': randint(1, 5),
    'max_features': ['auto', 'sqrt']
    }

"""

def modelo_RF(X_train,y_train,n_estimators,random_state):
    """
    Se crea un modelo random forest a partir de datos de entrenamiento y unos parametros determinados
    """
    # Crear el modelo Random Forest
    modelo_rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)  # Puedes ajustar los hiperparámetros como n_estimators

    # Entrenar el modelo
    modelo_rf.fit(X_train, y_train)

    return modelo_rf


def modelo_RF_HP(X_train,y_train,random_state,param_grid):
    """
    Funcion para realizar el Random forest con los mejores hiperparametros proporcionando un param_grid
    """
    rf = RandomForestClassifier(random_state=random_state)

    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2,scoring='f1_macro')

    grid_search.fit(X_train, y_train)

    print("Mejores hiperparámetros:", grid_search.best_params_)
    best_model = grid_search.best_estimator_

    return best_model,grid_search.best_score_, grid_search.best_params_


def rendimiento_rf(best_model,X_test,y_test):
    """
    A partir de un modelo y datos de test se comprueba el rendimiento de un Random Forest
    """
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

    # Predecir en el conjunto de prueba
    y_pred = best_model.predict(X_test)

    # 1. Precisión (accuracy)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.4f}')

    # 2. Matriz de confusión
    print("Matriz de Confusión:")
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(conf_matrix)

    # 3. Reporte de clasificación (precisión, recall, F1-score)
    print("Reporte de Clasificación:")
    clas_rep = classification_report(y_test, y_pred)
    print(clas_rep)

    return accuracy, conf_matrix, clas_rep


def plot_importancia_variables(model,X_train):
    import pandas as pd
    import matplotlib.pyplot as plt

    """
    Para evaluar las variables mas relevantes
    """

    # Obtener la importancia de las características
    feature_importances = model.feature_importances_

    # Crear un DataFrame con los nombres de las características y su importancia
    importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    # Mostrar las principales características
    print(importance_df)

    # Visualización
    importance_df.plot(kind='bar', x='Feature', y='Importance', title='Feature Importances')
    plt.show()

    return importance_df


def predict_proba(model,X):
    # Obtener las probabilidades predichas
    y_pred_proba = model.predict_proba(X)

    return y_pred_proba



