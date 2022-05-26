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
    grid_model.add(LSTM(50, activation='relu', return_sequences=True,input_shape=(LOOK_BACK, 10)))
    grid_model.add(LSTM(50))
    grid_model.add(Dense(1))

    grid_model.compile(loss = 'mse',optimizer = optimizer)
    return grid_model

def createXY(dataset,n_past):
    dataX = []
    dataY = []
    for i in range(n_past, len(dataset)):
            dataX.append(dataset[i - n_past:i, 0:dataset.shape[1]])
            dataY.append(dataset[i,0])
    return np.array(dataX),np.array(dataY)


data = pd.read_csv('Data/Scalanie2.csv', parse_dates=['Data'])

data.rename(columns = {'(pył zawieszony PM2.5 [jednostka ug/m3])':'PM2.5'
                    , '(pył zawieszony PM10 [jednostka ug/m3])':'PM10'
                    , '(tlenek azotu [jednostka ug/m3])':'NO'
                    , '(tlenki azotu [jednostka ug/m3])':'NOs'
                    , '(tlenek węgla [jednostka ug/m3])':'CO'
                    , '(benzen [jednostka ug/m3])':'benzen'
                    , '(dwutlenek azotu [jednostka ug/m3])':'NO2'
                    , 'Scalanie1.B00202A.Wynik':'Kierunek wiatru'
                    , 'Scalanie1.B00300S.Wynik':'Temperatura powietrza'
                    , 'Scalanie1.B00606S.Wynik':'Opady'
                    , 'Scalanie1.B00702A.Wynik':'Predkosc wiatru'
                    , 'Scalanie1.B00703A.Wynik':'MAX Predkosc wiatru'
                    , 'Scalanie1.B00802A.Wynik':'Wilgotnosc powietrza'
                    }, inplace = True)

data.dropna(inplace=True)
#data_df = data
# data_df = data_df.reset_index()
# data_df = data_df.drop('index', axis=1)
# data_df.index = data_df['Data']

data = data.drop(['Scalanie1.Stacja','Scalanie1.Data','Scalanie1.Rodzaj'], axis=1)
data['Data'] = pd.to_datetime(data['Data'])
data = data.sort_values(by="Data")

data.set_axis(data['Data'], inplace=True)
data.drop(columns=[ 'benzen', 'NO', 'NOs'], inplace=True)
data = data.drop('Data', axis=1)

split_percent = 0.80
split = int(split_percent*len(data['PM2.5']))

df_train = data[:split]
df_test = data[split:]


list = [i for i in data.columns if i != 'Data']
scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(df_train[list])
scaled_test = scaler.transform(df_test[list])

n_features = len(data.columns)
LOOK_BACK = 24
trainX, trainY = createXY(scaled_train,LOOK_BACK)
testX, testY = createXY(scaled_test,LOOK_BACK)
print("trainX Shape-- ",trainX.shape)
print("trainY Shape-- ",trainY.shape)
print("testX Shape-- ",testX.shape)
print("testY Shape-- ",testY.shape)

grid_model = KerasRegressor(build_fn=build_model,verbose=1,validation_data=(testX,testY))
parameters = {'batch_size' : [16,20,32,64],
              'epochs' : [8],
              'optimizer' : ['adam'] }


grid_search  = GridSearchCV(estimator = grid_model,
                            param_grid = parameters,
                            cv = 2)


grid_search = grid_search.fit(trainX, trainY)

print(grid_search.best_params_)
my_model=grid_search.best_estimator_.model

prediction=my_model.predict(testX)
prediction_copies_array = np.repeat(prediction,10, axis=-1)
pred=scaler.inverse_transform(np.reshape(prediction_copies_array,(len(prediction),10)))[:,0]

original_copies_array = np.repeat(testY,10, axis=-1)
original=scaler.inverse_transform(np.reshape(original_copies_array,(len(testY),10)))[:,0]

trace1 = go.Scatter(
    y = trainY,
    mode = 'lines',
    name = 'Dane'
)
trace2 = go.Scatter(
    y = pred,
    mode = 'lines',
    name = 'Predykcja'
)
trace3 = go.Scatter(
    y = original,
    mode='lines',
    name = 'Rzeczywistość'
)
layout = go.Layout(
    title = "Predykcja PM2.5",
    xaxis = {'title' : "Data"},
    yaxis = {'title' : "PM2.5[ug/m3]"}
)
fig = go.Figure(data=[ trace2, trace3], layout=layout)
fig.show()