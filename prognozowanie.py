import plotly.graph_objects as go

## TUTAJ POBIERZEMY DANE Z AIRLY I POGODOWE




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
