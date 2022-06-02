import plotly.graph_objects as go
import requests
import json
from random import randrange
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np
import keras

def createXY(dataset, n_past):
    dataX = []
    dataY = []
    for i in range(n_past, len(dataset)):
        dataX.append(dataset[i - n_past:i, 0:dataset.shape[1]])
        dataY.append(dataset[i, 0])
    return np.array(dataX), np.array(dataY)


# TUTAJ POBIERAMY DANE Z AIRLY I POGODOWE

my_model = keras.models.load_model('model')

weather_url = 'https://api.open-meteo.com/v1/forecast?latitude=50.100&longitude=22.050&hourly=temperature_2m,relativehumidity_2m,precipitation,windspeed_10m,winddirection_10m&daily=windspeed_10m_max&timezone=Europe%2FBerlin&past_days=1'
airly_api_url = 'https://airapi.airly.eu/v2/measurements/installation?installationId=7491'

headers = {'Accept': 'application/json',
           'Accept-Encoding': 'gzip',
           'apikey': '07nP6PqsOPKPYMOq2L0D8qlFpcvYyY2f'
           }

weather_res = requests.get(weather_url)

airly_res = requests.get(airly_api_url, headers=headers)

history = airly_res.json()['history']

forecast = airly_res.json()['forecast']

weather_hourly = weather_res.json()['hourly']

weather_daily = weather_res.json()['daily']

data = []

i = 0
for val in history:
    if len(val['values']) > 0:
        data.append(       
            {
                "date": val['fromDateTime'],
                "pm25": val['values'][0]['value'], # same as in forecast below but throws IndexError: list index out of range
                #"pm25": randrange(10, 25),
                "temperature_2m": weather_hourly['temperature_2m'][i],
                "precipitation": weather_hourly['precipitation'][i],
                "windspeed_10m": weather_hourly['windspeed_10m'][i],
                "relativehumidity_2m": weather_hourly['relativehumidity_2m'][i],
                "winddirection_10m": weather_hourly['winddirection_10m'][i],
            }
    )
    i = i + 1

i = 24
for val in forecast:
    if len(val['values']) > 0:
        data.append(
            {
                "date": val['fromDateTime'],
                "pm25": val['values'][0]['value'],
                "temperature_2m": weather_hourly['temperature_2m'][i],
                "precipitation": weather_hourly['precipitation'][i],
                "windspeed_10m": weather_hourly['windspeed_10m'][i],
                "relativehumidity_2m": weather_hourly['relativehumidity_2m'][i],
                "winddirection_10m": weather_hourly['winddirection_10m'][i],
            }
    )
    i = i + 1

dataFrame = pd.read_json(json.dumps(data))
dataFrame.index = dataFrame['date']
dataFrame = dataFrame.drop('date', axis=1)

print(dataFrame)

LOOK_BACK = 24

list = [i for i in dataFrame.columns if i != 'date']
scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(dataFrame[list])
# scaled_test = scaler.transform(dataFrame[list])

testX, testY = createXY(scaled_train, LOOK_BACK)
# schedule.every().hour.do(geeks)
print(testX.shape)
# print(testY)
n_features = 6
prediction = my_model.predict(testX)
prediction_copies_array = np.repeat(prediction, n_features, axis=-1)
pred = scaler.inverse_transform(np.reshape(
    prediction_copies_array, (len(prediction), n_features)))[:, 0]
print(pred)

dataFrame = dataFrame.reset_index()

trace1 = go.Scatter(
    x=dataFrame['date'].head(24),
    y=dataFrame['pm25'].head(24),
    mode='lines',
    name='Dane'
)
trace2 = go.Scatter(
    x=dataFrame['date'].tail(25),
    y=pred,
    mode='lines',
    name='Nasza prognoza'
)
trace3 = go.Scatter(
    x=dataFrame['date'].tail(25),
    y=dataFrame['pm25'].tail(25),
    mode='lines',
    name='Prognoza Airly'
)

layout = go.Layout(
    title="Predykcja PM2.5",
    xaxis={'title': "Data"},
    yaxis={'title': "PM2.5[ug/m3]"}
)
fig3 = go.Figure(data=[trace1, trace2, trace3], layout=layout)
#fig3.show()
trace1 = go.Scatter(
    # x=df_train['Data'],
    # y=df_train['PM2.5'],
    mode='lines',
    name='Dane'
)
layout = go.Layout(
    title="Predykcja PM2.5",
    xaxis={'title': "Data"},
    yaxis={'title': "PM2.5[ug/m3]"}
)
fig4 = go.Figure(data=[trace1], layout=layout)
#
# fig3.show()
i = 1
for val in data:
    print(i, val['date'])
    i = i + 1

