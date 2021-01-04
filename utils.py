import numpy as np
import pandas as pd


def get_features(data, n_out=7):
    data['change_in_price'] = data['close'].diff()
    data["y"] = ((data["close"] - data["close"].shift(n_out)) /
                 data["close"].shift(n_out) * 100).shift(-n_out)
    data[["open", "high", "low", "close"]] = data[["open", "high", "low",
                                                   "close"]].transform(lambda x: x.ewm(span=n_out, adjust=False).mean())

    n = 14

    up_df, down_df = pd.DataFrame(data['change_in_price'].copy()), pd.DataFrame(
        data['change_in_price'].copy())

    up_df.loc['change_in_price'] = up_df.loc[(
        up_df['change_in_price'] < 0), 'change_in_price'] = 0

    down_df.loc['change_in_price'] = down_df.loc[(
        down_df['change_in_price'] > 0), 'change_in_price'] = 0

    down_df['change_in_price'] = down_df['change_in_price'].abs()

    ewma_up = up_df['change_in_price'].transform(
        lambda x: x.ewm(span=n).mean())
    ewma_down = down_df['change_in_price'].transform(
        lambda x: x.ewm(span=n).mean())

    relative_strength = ewma_up / ewma_down

    relative_strength_index = 100.0 - (100.0 / (1.0 + relative_strength))

    data['down_days'] = down_df['change_in_price']
    data['up_days'] = up_df['change_in_price']
    data['RSI'] = relative_strength_index

    n = 14

    low_14, high_14 = pd.DataFrame(
        data['low'].copy()), pd.DataFrame(data['high'].copy())

    low_14 = low_14['low'].transform(lambda x: x.rolling(window=n).min())
    high_14 = high_14['high'].transform(lambda x: x.rolling(window=n).max())

    k_percent = 100 * ((data['close'] - low_14) / (high_14 - low_14))

    data['low_14'] = low_14
    data['high_14'] = high_14
    data['k_percent'] = k_percent

    n = 14
    r_percent = ((high_14 - data['close']) / (high_14 - low_14)) * - 100
    data['r_percent'] = r_percent

    ema_26 = data['close'].transform(lambda x: x.ewm(span=26).mean())
    ema_12 = data['close'].transform(lambda x: x.ewm(span=12).mean())
    macd = ema_12 - ema_26

    ema_9_macd = macd.ewm(span=9).mean()

    data['MACD'] = macd
    data['MACD_EMA'] = ema_9_macd

    n = 9
    data['Price_Rate_Of_Change'] = data['close'].transform(
        lambda x: x.pct_change(periods=n))

    def obv(group):
        volume = group['volume']
        change = group['close'].diff()

        # intialize the previous OBV
        prev_obv = 0
        obv_values = []

        # calculate the On Balance Volume
        for i, j in zip(change, volume):

            if i > 0:
                current_obv = prev_obv + j
            elif i < 0:
                current_obv = prev_obv - j
            else:
                current_obv = prev_obv

            # OBV.append(current_OBV)
            prev_obv = current_obv
            obv_values.append(current_obv)

        # Return a panda series.
        return pd.Series(obv_values, index=group.index)

    data['On Balance Volume'] = obv(data)
    FCOLS = ['RSI', 
               'k_percent', 
               'r_percent', 
               'Price_Rate_Of_Change', 
               'MACD', 
               'On Balance Volume', 
               "open", 
               "high", 
               "low", 
               "close"]
    fd = data[FCOLS]
    X = data.dropna(axis=0)
    y_bin = np.uint(X["y"] > 0)
    y = np.abs(X["y"])
    return fd, X[FCOLS], y, y_bin
