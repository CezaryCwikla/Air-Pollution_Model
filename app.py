from flask import Flask, render_template
from LSTM2 import fig
from LSTM_z_pogodowymi2 import fig2, my_model
from prognozowanie import fig3
import pandas as pd
import json
import plotly
import plotly.express as px


app = Flask(__name__, template_folder='templates', static_url_path='/static')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/chart1')
def chart1():

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    header = "Wyniki uczenia bez danych pogodowych"
    description = """Wykres przedstawia proces uczenia sieci neuronowej LSTM jedynie przy pomocy danych o ilości zanieczyszczeń PM 2.5"""
    return render_template('wykres.html', graphJSON=graphJSON, header=header, description=description)


@app.route('/chart2')
def chart2():

    graphJSON = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
    header = "Wyniki uczenia przy wykorzystaniu danych pogodowych"
    description = """Wykres przedstawia proces uczenia sieci neuronowej LSTM wykorzystaniu danych dotyczących zanieczyszczeń PM 2.5 oraz danych pogodowych takich jak temperatura, wyligotnośc powietrza, czy siła i kierunek wiatru."""
    return render_template('wykres.html', graphJSON=graphJSON, header=header, description=description)

@app.route('/chart3')
def chart3():

    graphJSON = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)
    header = "Prognoza zanieczyszczenia"
    description = """Wykres przedstawia prognozę zanieczysczeń na podstawie utworzonego modelu w porówaniu do przewidywań platformy Airly"""
    return render_template('wykres.html', graphJSON=graphJSON, header=header, description=description)




