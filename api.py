import numpy as np
import pandas as pd

import joblib

import yfinance as yf
from catboost import CatBoostClassifier, CatBoostRegressor

from datetime import date

from pathlib import Path

from utils import get_features

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

DATA_PATH = "stocks/"
MODELS_PATH = "models/"
TARGET_WEEKDAY = "Friday"

SYMBOLS = "SPY, QQQ".split(", ")


class Stock(object):

    def __init__(self, name, symbol):
        self.name = name
        self.symbol = symbol
        self.data_path = Path.cwd() / DATA_PATH / (self.name + ".csv")
        self.model_path = Path.cwd() / MODELS_PATH / (self.name + ".pkl")
        self.today_str = str(date.today())

        self.__download_data()
        self.__find_fridays()
        self.__get_features()

    def __download_data(self):
        path = Path(self.data_path)
        data = yf.download(
            self.symbol, start="2002-01-01").drop(["Adj Close"], axis=1)
        data.columns = ["open", "high", "low", "close", "volume"]
        self.data = data

    def __get_features(self):
        features_data, X, y, y_bin = get_features(self.data.copy())
        self.features_data = features_data
        self.X = X
        self.y = y
        self.y_bin = y_bin
        print(X.shape)

    def __find_fridays(self):
        fridays_mask = (self.data.index.weekday_name == "Friday")
        fridays = self.data.index[fridays_mask]
        last_friday_date = fridays[-1]
        next_friday_date = last_friday_date + pd.Timedelta(7, unit='D')
        self.last_friday_date = last_friday_date
        self.next_friday_date = next_friday_date
        self.last_friday_price = self.data.loc[last_friday_date, "close"]

    def __predict(self):
        models_dict = joblib.load(self.model_path)
        X = self.features_data.loc[self.last_friday_date]
        return models_dict["sign"].predict(X.values), models_dict["abs"].predict(X.values)

    def fit(self, iters=6000, depth=7):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y_bin, random_state=0, test_size=0.2, shuffle=True)
        sign_model = CatBoostClassifier(
            depth=depth, iterations=iters, random_seed=0, eval_metric="Accuracy")
        sign_model.fit(X_train, y_train, use_best_model=True,
                       verbose=True, eval_set=(X_test, y_test))

        acc = accuracy_score(sign_model.predict(X_test), y_test)

        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, random_state=0, test_size=0.2, shuffle=True)
        abs_model = CatBoostRegressor(
            depth=depth, iterations=iters, random_seed=0, eval_metric="MAE")
        abs_model.fit(X_train, y_train, use_best_model=True,
                      verbose=True, eval_set=(X_test, y_test))

        _ = joblib.dump({"sign": sign_model, "abs": abs_model},
                        self.model_path, compress=9)
        return acc

    def json(self):
        sign_val, abs_val = self.__predict()
        if sign_val == 0.0:
            sign_val = -1.0

        json = {
            "Symbol": self.symbol,
            "Name": self.name,
            "Last friday": self.last_friday_date.strftime("%d %B %Y"),
            "Last friday price": round(self.last_friday_price, 5),
            "Next friday": self.next_friday_date.strftime("%d %B %Y"),
            "Prediction": round(sign_val*abs_val, 5),
        }
        return json
