from api import *

SYMBOLS = "AMD, AAPL, JPM, MU, NVDA, MNST".split(", ")

with open("Accuracy results.txt", "a") as f:
    accs = []
    for symbol in SYMBOLS:
        stock = Stock(symbol, symbol)
        acc = round(stock.fit(iters=6000, depth=7), 3)
        accs.append(acc)
        print(f"{symbol} accuracy {acc}")
        f.write(f"{symbol} accuracy {acc}\n")
    f.write(f"Mean accuracy {np.array(accs).mean()}")
    