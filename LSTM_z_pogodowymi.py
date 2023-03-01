import pandas as pd
import numpy as np
import math
import keras
import tensorflow as tf
import plotly.graph_objects as go

from matplotlib import pyplot as plt
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import LSTM, Dense


def build_model(optimizer):
    grid_model = Sequential()
    # pytnaie tylko czy liczbaw warst jest odpowiednia, czy funkcja aktywacji sie zgadza etc
    grid_model.add(LSTM(50, activation='relu',
                   return_sequences=True, input_shape=(LOOK_BACK, n_features)))
    grid_model.add(LSTM(50))
    grid_model.add(Dense(1))

    grid_model.compile(loss='mse', optimizer=optimizer)
    return grid_model


def createXY(dataset, n_past):
    dataX = []
    dataY = []
    for i in range(n_past, len(dataset)):
        dataX.append(dataset[i - n_past:i, 0:dataset.shape[1]])
        dataY.append(dataset[i, 0])
    return np.array(dataX), np.array(dataY)


data = pd.read_csv('Data/Scalanie2_2.csv', parse_dates=['Data'])

data.rename(
    columns={'(pył zawieszony PM2.5 [jednostka ug/m3])': 'PM2.5',
             '(pył zawieszony PM10 [jednostka ug/m3])': 'PM10',
             '(tlenek azotu [jednostka ug/m3])': 'NO',
             '(tlenki azotu [jednostka ug/m3])': 'NOs',
             '(tlenek węgla [jednostka ug/m3])': 'CO',
             '(benzen [jednostka ug/m3])': 'benzen',
             '(dwutlenek azotu [jednostka ug/m3])': 'NO2',
             'B00202A (2).Wynik': 'Kierunek wiatru',
             'Scalanie1.B00300S.Wynik': 'Temperatura powietrza',
             'Scalanie1.B00606S.Wynik': 'Opady',
             'Scalanie1.B00702A.Wynik': 'Predkosc wiatru',
             'Scalanie1.B00703A.Wynik': 'MAX Predkosc wiatru',
             'Scalanie1.B00802A.Wynik': 'Wilgotnosc powietrza'
             }, inplace=True
)

data.dropna(inplace=True)
# data_df = data
# data_df = data_df.reset_index()
# data_df = data_df.drop('index', axis=1)
# data_df.index = data_df['Data']

data = data.drop(['Scalanie1.Stacja', 'Scalanie1.Data',
                 'Scalanie1.Rodzaj'], axis=1)
data['Data'] = pd.to_datetime(data['Data'])
data = data.sort_values(by="Data")

data.set_axis(data['Data'], inplace=True)
data.drop(columns=['PM10', 'CO', 'NO2', 'benzen', 'NO', 'NOs', 'MAX Predkosc wiatru'], inplace=True)
data = data.drop('Data', axis=1)

split_percent = 0.80
split = int(split_percent * len(data['PM2.5']))

df_train = data[:split]
df_test = data[split:]

list = [i for i in data.columns if i != 'Data']
scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(df_train[list])
scaled_test = scaler.transform(df_test[list])

n_features = len(data.columns)
LOOK_BACK = 24
trainX, trainY = createXY(scaled_train, LOOK_BACK)
testX, testY = createXY(scaled_test, LOOK_BACK)
print("trainX Shape-- ", trainX.shape)
print("trainY Shape-- ", trainY.shape)
print("testX Shape-- ", testX.shape)
print("testY Shape-- ", testY.shape)

grid_model = KerasRegressor(
    build_fn=build_model, verbose=1, validation_data=(testX, testY))
parameters = {'batch_size': [16],  # tutaj mozna sie pobawic batch size
              'epochs': [10],  # tutaj mozna sie pobawic liczba epoch
              'optimizer': ['adam']}  # tutaj mozna dodac jakis optimizer, ale adam wszedzie raczej wygrywal

grid_search = GridSearchCV(estimator=grid_model,
                           param_grid=parameters,
                           cv=2)

grid_search = grid_search.fit(trainX, trainY)

print(grid_search.best_params_)
my_model = grid_search.best_estimator_.model
my_model.save('model')

prediction = my_model.predict(testX)
prediction_copies_array = np.repeat(prediction, n_features, axis=-1)
pred = scaler.inverse_transform(np.reshape(
    prediction_copies_array, (len(prediction), n_features)))[:, 0]

original_copies_array = np.repeat(testY, n_features, axis=-1)
original = scaler.inverse_transform(np.reshape(
    original_copies_array, (len(testY), n_features)))[:, 0]

testScore = math.sqrt(mean_squared_error(original, pred))
print('Test Score: %.2f RMSE' % testScore)

with open("metrics.txt", 'w') as outfile:
    outfile.write('Test Score: %.2f RMSE' % testScore)

df_test = df_test.reset_index()

trace1 = go.Scatter(
    y=trainY,
    mode='lines',
    name='Dane'
)
trace2 = go.Scatter(
    y=pred,
    x = df_test['Data'],
    mode='lines',
    name='Predykcja'
)
trace3 = go.Scatter(
    y=original,
    x=df_test['Data'],
    mode='lines',
    name='Rzeczywistość'
)
layout = go.Layout(
    title="Predykcja PM2.5",
    xaxis={'title': "Data"},
    yaxis={'title': "PM2.5[ug/m3]"}
)
fig2 = go.Figure(data=[trace2, trace3], layout=layout)
fig2.show()
