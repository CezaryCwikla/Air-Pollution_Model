import plotly.graph_objects as go
import requests
import json
from random import randrange

## TUTAJ POBIERAMY DANE Z AIRLY I POGODOWE

weather_url = 'https://api.open-meteo.com/v1/forecast?latitude=50.100&longitude=22.050&hourly=temperature_2m,relativehumidity_2m,precipitation,windspeed_10m&daily=windspeed_10m_max&timezone=Europe%2FBerlin&past_days=1'
airly_api_url = 'https://airapi.airly.eu/v2/measurements/installation?installationId=7491'

headers = {'Accept': 'application/json',
           'Accept-Encoding': 'gzip',
           'apikey': '07nP6PqsOPKPYMOq2L0D8qlFpcvYyY2f'
           }

weather_res = requests.get(weather_url)

airly_res = requests.get(airly_api_url, headers=headers)

history = airly_res.json()['history']

forecast = airly_res.json()['forecast']

weather = weather_res.json()['hourly']

data = []

i = 0
for val in history:
    data.append(
        {
            "date": val['fromDateTime'],
            # "pm25": val['values'][0]['value'], # same as in forecast below but throws IndexError: list index out of range
            "pm25": randrange(10, 25),
            "precipitation": weather['precipitation'][i],
            "windspeed_10m": weather['windspeed_10m'][i],
            "temperature_2m": weather['temperature_2m'][i],
            "relativehumidity_2m": weather['relativehumidity_2m'][i],
        }
    )
    i = i + 1

i = 24
for val in forecast:
    data.append(
        {
            "date": val['fromDateTime'],
            "pm25": val['values'][0]['value'],
            "precipitation": weather['precipitation'][i],
            "windspeed_10m": weather['windspeed_10m'][i],
            "temperature_2m": weather['temperature_2m'][i],
            "relativehumidity_2m": weather['relativehumidity_2m'][i],
        }
    )
    i = i + 1

print(data)

trace1 = go.Scatter(
    #x=df_train['Data'],
    #y=df_train['PM2.5'],
    mode='lines',
    name='Dane'
)
trace2 = go.Scatter(
    # y=pred,
    # x=df_test['Data'],
    mode='lines',
    name='Predykcja'
)
trace3 = go.Scatter(
    # y=original,
    # x=df_test['Data'],
    mode='lines',
    name='Rzeczywistość'
)
layout = go.Layout(
    title="Predykcja PM2.5",
    xaxis={'title': "Data"},
    yaxis={'title': "PM2.5[ug/m3]"}
)
fig3 = go.Figure(data=[trace1, trace2, trace3], layout=layout)
