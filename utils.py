import pandas as pd


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