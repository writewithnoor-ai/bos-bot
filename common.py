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
    plan: dict | None = None        # entries / SL / TPs / lots


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
    n = limit or CANDLES
    resample = tf == "3min"  # Twelve Data has no 3min candles: build them from 1min
    r = requests.get("https://api.twelvedata.com/time_series",
                     params={"symbol": f"{sym[:3]}/{sym[3:]}",
                             "interval": "1min" if resample else tf,
                             "outputsize": n * 3 if resample else n, "timezone": "UTC",
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
    if resample:
        df = df.resample("3min").agg({"Open": "first", "High": "max", "Low": "min",
                                      "Close": "last", "Volume": "sum"}).dropna()
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
    left = 0
    if a.origin is not None and a.ext is not None:
        oi, ei = locate(df, a.origin_t, secs), locate(df, a.ext_t, secs)
        start = oi if oi is not None else 0
        left = max(0, start - 40)  # zoom in around the setup
        for r in a.ratios:
            y = lvl(a, r)
            ys.append(y)
            if a.entry_r is not None and abs(r - a.entry_r) < 1e-9:
                col, lw = "#ffd54f", 1.6
            elif a.zone_r is not None and abs(r - a.zone_r) < 1e-9:
                col, lw = "#66bb6a", 1.6
            else:
                col, lw = "#9e9e9e", 0.8
            ax.hlines(y, start, n + 8, colors=col, linewidths=lw, linestyles="--")
            ax.text(n + 8.4, y, f"{r:g}", color=col, fontsize=8, va="center")
        if oi is not None:
            ax.scatter([oi], [a.origin], s=45, color="#2962ff", zorder=5)
        if ei is not None:
            ax.scatter([ei], [a.ext], s=45, color="#2962ff", zorder=5)
        if a.event == "touched" and a.entry_r is not None:
            ax.scatter([n - 1], [lvl(a, a.entry_r)], s=60, zorder=6,
                       color="#26a69a" if bull else "#ef5350")
    else:  # plain BOS alert
        ys.append(a.price)
        ax.hlines(a.price, 0, n + 8, colors="#2962ff", linewidths=1.2)
    p = a.plan
    if p:
        d = 1 if bull else -1
        top = p["tp2"] if bull else p["e1"]
        bot = p["e1"] if bull else p["tp2"]
        ax.fill_between([n, n + 7], bot, top, color="#26a69a", alpha=0.18, linewidth=0)
        top = p["e1"] if bull else p["sl"]
        bot = p["sl"] if bull else p["e1"]
        ax.fill_between([n, n + 7], bot, top, color="#ef5350", alpha=0.18, linewidth=0)
        for name, price, col in (("entry 1", p["e1"], "#4da3ff"), ("entry 2", p["e2"], "#4da3ff"),
                                 ("SL", p["sl"], "#ef5350"), ("TP1", p["tp1"], "#26a69a"),
                                 ("TP2", p["tp2"], "#26a69a")):
            ys.append(price)
            ax.hlines(price, n, n + 7, colors=col, linewidths=1.8)
            ax.text(n + 0.3, price, name, color=col, fontsize=7, va="bottom")
    vis = df.iloc[left:]
    lo, hi = min([vis["Low"].min()] + ys), max([vis["High"].max()] + ys)
    pad = (hi - lo) * 0.04
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlim(left - 1, n + 14)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def plan_caption(a: Alert, icon: str, head: str) -> str:
    p = a.plan
    title = f"{a.direction} signal" + (" - Entry 1 touched" if a.event == "touched" else "")
    out = [f"{icon} {title}", head, "",
           f"Entry 1: {p['e1']:.6g}  |  {p['lots1']:g} lots",
           f"Entry 2: {p['e2']:.6g}  |  {p['lots2']:g} lots",
           f"Stop loss: {p['sl']:.6g}",
           f"TP1: {p['tp1']:.6g}",
           f"TP2: {p['tp2']:.6g}", "",
           f"Max loss (both entries filled, SL hit): {p['risk']:.0f} USDT",
           f"Loss if only Entry 1 fills: {p['risk1']:.0f} USDT",
           f"Profit if all closed at TP1: {p['pt1']:.0f} | at TP2: {p['pt2']:.0f} USDT"]
    if p.get("low_size"):
        out.append("Warning: size is below the minimum lot, check your risk.")
    if a.event != "touched":
        out.append("Levels may shift until Entry 1 is touched.")
    return "\n".join(out)


def build_caption(a: Alert, sym: str, tf_label: str) -> str:
    bull = a.direction.upper() in ("BUY", "BULLISH")
    icon = "🟢" if bull else "🔴"
    head = f"{sym} · {tf_label}"
    if a.event == "bos":
        return f"{icon} {a.direction.title()} BOS\n{head}\nLevel: {a.price}"
    if a.plan:
        return plan_caption(a, icon, head)
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
