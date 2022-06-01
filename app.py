from flask import Flask, render_template
from LSTM2 import fig
from LSTM_z_pogodowymi2 import fig2, my_model
#from prognozowanie import fig3
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
    header = "Fruit in North America"
    description = """
    A academic study of the number of apples, oranges and bananas in the cities of
    San Francisco and Montreal would probably not come up with this chart.
    """
    return render_template('wykres.html', graphJSON=graphJSON, header=header, description=description)


@app.route('/chart2')
def chart2():

    graphJSON = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
    header = "Vegetables in Europe"
    description = """
    The rumor that vegetarians are having a hard time in London and Madrid can probably not be
    explained by this chart.
    """
    return render_template('wykres.html', graphJSON=graphJSON, header=header, description=description)

@app.route('/chart3')
def chart3():

    graphJSON = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)
    header = "Vegetables in Europe"
    description = """
    The rumor that vegetarians are having a hard time in London and Madrid can probably not be
    explained by this chart.
    """
    return render_template('wykres.html', graphJSON=graphJSON, header=header, description=description)

@app.route('/chart4')
def chart4():

    graphJSON = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
    header = "Vegetables in Europe"
    description = """
    The rumor that vegetarians are having a hard time in London and Madrid can probably not be
    explained by this chart.
    """
    return render_template('wykres.html', graphJSON=graphJSON, header=header, description=description)


@app.route('/about')
def about():

    return render_template('about.html')




