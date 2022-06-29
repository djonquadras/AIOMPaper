import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn import ensemble
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


def createDatase():
    # Importing Data
    data1 = pd.read_excel("Data/wMachine2018.xlsx", engine='openpyxl')
    data2 = pd.read_excel("Data/wMachine2019.xlsx", engine='openpyxl')
    data3 = pd.read_excel("Data/wMachine2020.xlsx", engine='openpyxl')
    data4 = pd.read_excel("Data/wMachine2021.xlsx", engine='openpyxl')
    data5 = pd.read_excel("Data/wMachine2022.xlsx", engine='openpyxl')
    data6 = pd.read_excel("Data/noMachine.xlsx", engine='openpyxl')
    data = pd.concat([data1, data2, data3, data4, data5, data6])

    # Fixing Data
    totalTime = data["tempo fila"] + data["tempo processamento"]
    data["totalTime"] = totalTime

    data['month'] = data['data chegada'].dt.month
    data['week'] = data['data chegada'].dt.week
    data['day'] = data['data chegada'].dt.day
    data['hour'] = data['data chegada'].dt.hour
    data['minute'] = data['data chegada'].dt.minute
    data['dayofweek'] = data['data chegada'].dt.dayofweek

    data = data[['classificadores', 'operador ocioso', 'tempo maquinas', 'equipamentos',
        'corta os graos', 'month', 'week', 'day', 'dayofweek', 'tempo fila', 'tempo processamento', 'totalTime']]

    data = data.rename(columns={'classificadores': "workers", "operador ocioso": "workersWaiting", "tempo maquinas": "machineProcTime",
                        'equipamentos': 'qntMachines', 'corta os graos': 'cutBean', 'tempo fila': 'queueTime', 'tempo processamento': 'procTime'})


    # Exporting to csv
    data.to_csv("Data/dataset.csv")

def createModels(X_train, y_train, X_test,y_test):
    vetor = [LinearRegression(),
        RandomForestRegressor(random_state=42,n_estimators=40),
        ensemble.GradientBoostingRegressor(),
        SVR(kernel = 'rbf'),
        DecisionTreeRegressor(random_state = 0)]

    names = ["Linear Regression",
            "Random Forest",
            "Gradient Boosting",
            "SVR",
            "Decision Tree"]

    models = list()
    i = 1

    for model in vetor:
        # Fit model
        model.fit(X_train, y_train)
        # Predict values of the test dataset
        y_pred = model.predict(X_test)
        # Calculate root mean squared error
        rmse = np.sqrt(mean_squared_error(y_pred, y_test))
        models.append(Models(model, rmse, names[i]))
        i +=1
    return models


class Models():
    def __init__(self, model, rmse, name):
        self.name = name
        self.rmse = rmse
        self.model = model

def printModels(models):
    rmse = []
    names = []
    for model in models:
        rmse.append(model.rmse)
        names.append(model.name)
    df_Summary = pd.DataFrame({'RMSE': rmse},
                            index = names)
    print(df_Summary)