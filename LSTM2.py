import pandas as pd
import numpy as np
import keras
import tensorflow as tf
import plotly.graph_objects as go

from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import LSTM, Dense


data = pd.read_csv('Data/Powietrze_2019-2022_Pilsudzkiego.csv', sep=';', parse_dates=['Data'])

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
data_df = data
# data_df = data_df.reset_index()
# data_df = data_df.drop('index', axis=1)
# data_df.index = data_df['Data']
# data_df = data_df.drop('Data', axis=1)

data['Data'] = pd.to_datetime(data['Data'])
data = data.sort_values(by="Data")

data.set_axis(data['Data'], inplace=True)
data.drop(columns=['CO', 'benzen', 'NO2', 'NO', 'NOs'], inplace=True)

PM25_data = data['PM2.5'].values
PM25_data = PM25_data.reshape((-1, 1))

PM10_data = data['PM10'].values
PM10_data = PM10_data.reshape((-1, 1))

split_percent = 0.80
split = int(split_percent*len(PM25_data))

PM25_train = PM25_data[:split]
PM25_test = PM25_data[split:]

PM10_train = PM10_data[:split]
PM10_test = PM10_data[split:]

date_train = data['Data'][:split]
date_test = data['Data'][split:]

look_back = 15

train_generator = TimeseriesGenerator(PM25_train, PM25_train, length=look_back, batch_size=20)
test_generator = TimeseriesGenerator(PM25_test, PM25_test, length=look_back, batch_size=1)

model = keras.models.load_model("model2")

prediction = model.predict(test_generator)
PM25_train = PM25_train.reshape((-1))
PM25_test = PM25_test.reshape((-1))
prediction = prediction.reshape((-1))

trace1 = go.Scatter(
    x = date_train,
    y = PM25_train,
    mode = 'lines',
    name = 'Dane'
)
trace2 = go.Scatter(
    x = date_test,
    y = prediction,
    mode = 'lines',
    name = 'Predykcja'
)
trace3 = go.Scatter(
    x = date_test,
    y = PM25_test,
    mode='lines',
    name = 'Rzeczywistość'
)
layout = go.Layout(
    title = "Predykcja PM2.5",
    xaxis = {'title' : "Data"},
    yaxis = {'title' : "PM2.5[ug/m3]"}
)
fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)
