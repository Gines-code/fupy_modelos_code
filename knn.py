############### SCRIPT CON MODELO KNN ###############
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def standarizacion(X1):
    """
    Para estandarizar datos (necesario para algunos modelos)
    """
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X1 = pd.DataFrame(scaler.fit_transform(X1), columns=X1.columns)

    return X1


def knn_entrenamiento(X1,y1):
    """
    knn con hiperparametro para el valor k
    """

    import random
    # lista con numeros aleatorios para el random_state
    list_random = []
    n = 0
    while n < 28:
        n_rand = random.randint(0, 1000)
        list_random.append(n_rand)
        n = n+1
    print(list_random)

    #KNN Y CROSS VALIDATION
    ks = list(range(4, 30))
    accs = {}
    # Vamos recorriendo la rejilla con un bucle for...
    for k in ks:

        # Definimos el modelo con el valor de hiperparámetro correspondiente
        knn = KNeighborsClassifier(n_neighbors=k)

        metricas = []
        for i in list_random:
            X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.2, random_state=i,stratify=y1) 
            # Ajustamos a los datos de entrenamiento
            knn.fit(X_train, y_train)

            # Hacemos predicciones sobre los datos de test
            y_pred1 = knn.predict(X_test)

            # Evaluamos y guardamos la métrica correspondiente (en este caso accuracy)
            acc = metrics.accuracy_score(y_test, y_pred1)
            
            metricas.append(acc)
        # accs[k] = acc
            
        media = sum(metricas)/len(metricas)
        accs[k] = media

    print(accs)
    maximo_par = max(accs.items(), key=lambda x: x[1])
    print("El máximo valor es:", maximo_par)

    #se hace el modelo con el mejor K
    k_best = maximo_par[0]
    knn = KNeighborsClassifier(n_neighbors=k_best)
    knn.fit(X1,y1)

    return knn, maximo_par


def buscar_datos_test(df,X_test):
    """
    para encontrar los datos de test y hacer pruebas
    """
    indices_X1 = X_test.index
    # Buscar los índices de df1 en df2
    df2_encontrados = df.loc[indices_X1.intersection(df.index)]
    print(df2_encontrados[['local','visitante','local_encoded','visitante_encoded']])

    return df2_encontrados


def metricas_prec_score(knn,X_test,y_test):
    """
    metricas para precision y accuracy
        precision: dice el procentaje de aciertos por grupo
        accuracy: porcentaje de aciertos total
    """
    y_kknpred=knn.predict(X_test)
    score = precision_score(y_kknpred,y_test,average = None)
    print(score)
    acc = metrics.accuracy_score(y_test, y_kknpred)
    print(acc)

    return score,acc


"""
EJEMPLO HP

    param_grid = {
        'n_neighbors': np.arange(1, 31, 2),
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'leaf_size': [10, 20, 30, 40, 50],
        'p': [1, 2],
        'metric': ['minkowski'],
        'metric_params': [{'w':pesos}]
    }
"""


def KNN_HP(param_grid,X1,y1):
    from sklearn.model_selection import GridSearchCV
    import numpy as np

    X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.2, random_state=17,stratify=y1)
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, n_jobs=-1, verbose=1,scoring='f1_macro')
    grid_search.fit(X_train, y_train)

    print(f"Mejores hiperparámetros: {grid_search.best_params_}")
    print(f"Mejor score: {grid_search.best_score_}")

    best_knnTrain = grid_search.best_estimator_
    best_knnTrain.fit(X_train,y_train)

    y_kknpred1=best_knnTrain.predict(X_test)
    score = precision_score(y_kknpred1,y_test,average = None)
    print(score)

    best_knn = grid_search.best_estimator_
    best_knn.fit(X1,y1)

    return best_knn, grid_search.best_score_, grid_search.best_params_
