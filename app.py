from api import *
from flask import Flask, render_template
app = Flask(__name__)


@app.route('/')
def index():
    stocks = [Stock(symbol, symbol) for symbol in SYMBOLS]
    stocks_json = []
    for stock in stocks:
        stock_json = stock.json()
        stock_json["Row class"] = "bg-success"
        if stock_json["Prediction"] < 0:
            stock_json["Row class"] = "bg-danger"
        stocks_json.append(stock_json)
    return render_template("index.html", stocks_json=stocks_json, enumerate=enumerate)
