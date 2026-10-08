"""Forecast GDXU daily candles with Kronos.

Needs the Kronos repo (https://github.com/shiyu-coder/Kronos) on disk and
network access to huggingface.co to download the model weights.

    python forecast_gdxu.py --kronos /home/user/shiyu-coder/kronos --days 20
"""
import argparse
import sys

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--kronos", default="/home/user/shiyu-coder/kronos")
parser.add_argument("--csv", default="data/GDXU_daily.csv")
parser.add_argument("--days", type=int, default=20)
parser.add_argument("--lookback", type=int, default=400)
parser.add_argument("--samples", type=int, default=5)
args = parser.parse_args()

sys.path.append(args.kronos)
from model import Kronos, KronosPredictor, KronosTokenizer  # noqa: E402

tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
predictor = KronosPredictor(model, tokenizer, max_context=512)

df = pd.read_csv(args.csv, parse_dates=["timestamps"]).tail(args.lookback).reset_index(drop=True)
future = pd.Series(pd.bdate_range(df["timestamps"].iloc[-1] + pd.Timedelta(days=1), periods=args.days))

pred = predictor.predict(
    df=df[["open", "high", "low", "close", "volume", "amount"]],
    x_timestamp=df["timestamps"],
    y_timestamp=future,
    pred_len=args.days,
    T=1.0,
    top_p=0.9,
    sample_count=args.samples,
    verbose=True,
)
pred.index = future.values

last = df["close"].iloc[-1]
end = pred["close"].iloc[-1]
print(pred[["open", "high", "low", "close"]].round(2))
print(f"\nLast close {last:.2f} -> forecast {end:.2f} in {args.days} trading days ({(end / last - 1):+.1%})")
pred.to_csv("data/GDXU_forecast.csv")
