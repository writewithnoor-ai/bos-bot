"""Shared helpers: data fetching, chart rendering, Telegram."""
import io
import os
import re
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import requests
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()
TG_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TG_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
TWELVE_KEY = os.environ.get("TWELVEDATA_API_KEY", "")
CANDLES = int(os.environ.get("CANDLES", "250"))
SETTLE_DELAY = float(os.environ.get("SETTLE_DELAY", "2"))  # let the exchange publish the closed bar



class Alert(BaseModel):
    secret: str
    symbol: str
    tf: str
    event: str = "forming"          # forming | touched | bos
    direction: str                  # BUY/SELL (setups) or BULLISH/BEARISH (bos)
    price: float
    origin: float | None = None     # fib 1.0
    ext: float | None = None        # fib 0.0
    origin_t: int | None = None     # ms timestamps of those bars
    ext_t: int | None = None
    entry_r: float | None = None
    zone_r: float | None = None
    ratios: list[float] = []


# TradingView interval -> (binance, twelvedata, seconds)
TF_MAP = {
    "1": ("1m", "1min", 60), "3": ("3m", "3min", 180), "5": ("5m", "5min", 300),
    "15": ("15m", "15min", 900), "30": ("30m", "30min", 1800),
    "60": ("1h", "1h", 3600), "120": ("2h", "2h", 7200), "240": ("4h", "4h", 14400),
    "D": ("1d", "1day", 86400), "1D": ("1d", "1day", 86400),
    "W": ("1w", "1week", 604800), "1W": ("1w", "1week", 604800),
}
CRYPTO_QUOTES = ("USDT", "USDC", "BUSD", "FDUSD")


def clean_symbol(s: str) -> str:
    s = s.split(":")[-1].upper()
    return re.sub(r"\.P$|PERP$", "", s)


def fetch_crypto(sym: str, tf: str, limit: int | None = None) -> pd.DataFrame:
    r = requests.get("https://data-api.binance.vision/api/v3/klines",
                     params={"symbol": sym, "interval": tf, "limit": limit or CANDLES}, timeout=15)
    r.raise_for_status()
    df = pd.DataFrame(r.json()).iloc[:, :6]
    df.columns = ["time", "Open", "High", "Low", "Close", "Volume"]
    df.index = pd.to_datetime(df.pop("time"), unit="ms")
    return df.astype(float)


def fetch_forex(sym: str, tf: str, limit: int | None = None) -> pd.DataFrame:
    if not TWELVE_KEY:
        raise RuntimeError("TWELVEDATA_API_KEY not set (needed for forex)")
    r = requests.get("https://api.twelvedata.com/time_series",
                     params={"symbol": f"{sym[:3]}/{sym[3:]}", "interval": tf,
                             "outputsize": limit or CANDLES, "timezone": "UTC",
                             "apikey": TWELVE_KEY}, timeout=15)
    r.raise_for_status()
    j = r.json()
    if "values" not in j:
        raise RuntimeError(f"Twelve Data error: {j}")
    df = pd.DataFrame(j["values"])
    df.index = pd.to_datetime(df.pop("datetime"))
    df = df.rename(columns=str.title).astype(float).sort_index()
    if "Volume" not in df:
        df["Volume"] = 0.0
    return df


def locate(df: pd.DataFrame, t_ms, secs):
    """Candle position nearest to a TradingView bar time, or None if off-screen."""
    if not t_ms:
        return None
    t = pd.to_datetime(t_ms, unit="ms")
    i = int(df.index.get_indexer([t], method="nearest")[0])
    return i if abs((df.index[i] - t).total_seconds()) <= secs else None


def lvl(a: Alert, r: float) -> float:
    return a.ext + r * (a.origin - a.ext)


def render_chart(df: pd.DataFrame, a: Alert, sym: str, tf_label: str, secs: int) -> bytes:
    bull = a.direction.upper() in ("BUY", "BULLISH")
    n = len(df)
    style = mpf.make_mpf_style(base_mpf_style="nightclouds",
                               marketcolors=mpf.make_marketcolors(
                                   up="#26a69a", down="#ef5350", inherit=True))
    title = f"{sym} {tf_label} | {a.direction.upper()} {a.event}"
    fig, axes = mpf.plot(df, type="candle", style=style, figsize=(10, 6),
                         volume=False, title=title, returnfig=True)
    ax = axes[0]
    ys = []
    if a.origin is not None and a.ext is not None:
        oi, ei = locate(df, a.origin_t, secs), locate(df, a.ext_t, secs)
        start = oi if oi is not None else 0
        for r in a.ratios:
            y = lvl(a, r)
            ys.append(y)
            if a.entry_r is not None and abs(r - a.entry_r) < 1e-9:
                col, lw = "#ffd54f", 1.6
            elif a.zone_r is not None and abs(r - a.zone_r) < 1e-9:
                col, lw = "#66bb6a", 1.6
            else:
                col, lw = "#9e9e9e", 0.8
            ax.hlines(y, start, n + 4, colors=col, linewidths=lw, linestyles="--")
            ax.text(n + 4.5, y, f"{r:g}", color=col, fontsize=8, va="center")
        if oi is not None:
            ax.scatter([oi], [a.origin], s=45, color="#2962ff", zorder=5)
        if ei is not None:
            ax.scatter([ei], [a.ext], s=45, color="#2962ff", zorder=5)
        if a.event == "touched" and a.entry_r is not None:
            ax.scatter([n - 1], [lvl(a, a.entry_r)], s=60, zorder=6,
                       color="#26a69a" if bull else "#ef5350")
    else:  # plain BOS alert
        ys.append(a.price)
        ax.hlines(a.price, 0, n + 4, colors="#2962ff", linewidths=1.2)
    lo, hi = min([df["Low"].min()] + ys), max([df["High"].max()] + ys)
    pad = (hi - lo) * 0.04
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlim(-1, n + 9)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def build_caption(a: Alert, sym: str, tf_label: str) -> str:
    bull = a.direction.upper() in ("BUY", "BULLISH")
    icon = "🟢" if bull else "🔴"
    head = f"{sym} · {tf_label}"
    if a.event == "bos":
        return f"{icon} {a.direction.title()} BOS\n{head}\nLevel: {a.price}"
    entry, zone = lvl(a, a.entry_r), lvl(a, a.zone_r)
    if a.event == "touched":
        return (f"{icon} {a.direction} setup: entry level touched\n{head}\n"
                f"Entry ({a.entry_r:g}): {entry:.6g}\nZone end ({a.zone_r:g}): {zone:.6g}\n"
                f"Price: {a.price}")
    return (f"{icon} {a.direction} setup forming\n{head}\n"
            f"Waiting for pullback to {a.entry_r:g}: {entry:.6g}\n"
            f"Zone end ({a.zone_r:g}): {zone:.6g}\n"
            f"Levels can shift until the entry level is touched.")


def tg(method: str, **kw):
    r = requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/{method}", timeout=30, **kw)
    r.raise_for_status()


