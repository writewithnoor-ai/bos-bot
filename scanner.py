"""BOS Reversal Fib: free auto-alert scanner (no TradingView needed).

Python port of the indicator. On every run it downloads recent closed candles,
replays the indicator logic, and sends any NEW setup to Telegram with a chart.
  python scanner.py              one pass (used by GitHub Actions)
  python scanner.py --loop 60    keep running on your own PC, check every 60s
  python scanner.py --test       send the latest event per symbol, to test Telegram
"""
import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from common import (Alert, CANDLES, CRYPTO_QUOTES, TF_MAP, TG_CHAT_ID, build_caption,
                    clean_symbol, fetch_crypto, fetch_forex, render_chart, tg)

HERE = os.path.dirname(os.path.abspath(__file__))
CFG_PATH = os.path.join(HERE, "config.json")
STATE_PATH = os.path.join(HERE, "state.json")
HISTORY = 1000  # candles replayed each run


@dataclass
class Setup:
    bull: bool
    originP: float
    originB: int
    extP: float
    extB: int
    bosBar: int
    triggered: bool = False

    def px(self, r: float) -> float:
        return self.extP + r * (self.originP - self.extP)


def run_engine(df: pd.DataFrame, c: dict) -> list[dict]:
    """Bar-by-bar port of the Pine script. Returns events in order."""
    H, L, C = df["High"].values, df["Low"].values, df["Close"].values
    n, P = len(df), c["pivLen"]
    pc = np.roll(C, 1)
    tr = np.maximum.reduce([H - L, np.abs(H - pc), np.abs(L - pc)])
    tr[0] = H[0] - L[0]
    atr = pd.Series(tr).ewm(alpha=1 / c["atrLen"], adjust=False).mean().values

    ev: list[dict] = []
    pHi = pLo = None
    pHiDone = pLoDone = True
    lastDir, lastBar = 0, None
    cur: Setup | None = None

    def snap(kind, i, s: Setup, direction):
        return dict(type=kind, bar=i, dir=direction, price=float(C[i]), origin=s.originP,
                    ext=s.extP, originB=s.originB, extB=s.extB, atr=float(atr[i]))

    for i in range(n):
        # swing pivots, confirmed P bars after they form
        if i >= 2 * P:
            m = i - P
            if H[m] > H[m - P:m].max() and H[m] >= H[m + 1:i + 1].max():
                pHi, pHiDone = H[m], False
            if L[m] < L[m - P:m].min() and L[m] <= L[m + 1:i + 1].min():
                pLo, pLoDone = L[m], False

        # BOS on bar close
        bosDir, brk = 0, None
        if not pHiDone and C[i] > pHi:
            bosDir, brk, pHiDone = 1, pHi, True
        elif not pLoDone and C[i] < pLo:
            bosDir, brk, pLoDone = -1, pLo, True
        if bosDir and c["alertBos"]:
            ev.append(dict(type="bos", bar=i, dir="BULLISH" if bosDir > 0 else "BEARISH",
                           price=float(brk)))

        bos1Dir, bos1Bar = lastDir, lastBar
        if bosDir:
            lastDir, lastBar = bosDir, i
        flipBuy = bosDir == 1 and bos1Dir == -1 and bos1Bar is not None and i - bos1Bar <= c["maxGap"]
        flipSell = bosDir == -1 and bos1Dir == 1 and bos1Bar is not None and i - bos1Bar <= c["maxGap"]

        # manage the live setup
        if cur is not None:
            just = False
            if not cur.triggered:
                if cur.bull:
                    if H[i] > cur.extP:
                        cur.extP, cur.extB = H[i], i
                    elif L[i] <= cur.px(c["entryRatio"]):
                        cur.triggered = just = True
                else:
                    if L[i] < cur.extP:
                        cur.extP, cur.extB = L[i], i
                    elif H[i] >= cur.px(c["entryRatio"]):
                        cur.triggered = just = True
            if just and c["alertZone"]:
                ev.append(snap("touched", i, cur, "BUY" if cur.bull else "SELL"))
            buf = c["endBuf"] * atr[i]
            end = cur.px(c["zoneEnd"]) - buf if cur.bull else cur.px(c["zoneEnd"]) + buf
            broken = C[i] < end if cur.bull else C[i] > end
            if broken or i - cur.bosBar >= c["maxBars"]:
                cur = None

        # new setup on a reversal BOS
        for flip, bull, enabled in ((flipBuy, True, c["showBuy"]), (flipSell, False, c["showSell"])):
            if not (flip and enabled):
                continue
            lo = max(i - min(i - bos1Bar, 4999), 0)
            seg = range(lo, i + 1)
            if bull:
                orig = min(seg, key=lambda b: (L[b], -b))      # lowest low, latest on ties
                oB = orig
                ext = max(range(oB, i + 1), key=lambda b: (H[b], b))
                valid = H[ext] > L[oB]
                oP, eP = L[oB], H[ext]
            else:
                oB = max(seg, key=lambda b: (H[b], b))
                ext = min(range(oB, i + 1), key=lambda b: (L[b], -b))
                valid = H[oB] > L[ext]
                oP, eP = H[oB], L[ext]
            if valid and (c["minLegAtr"] == 0 or abs(oP - eP) >= c["minLegAtr"] * atr[i]):
                cur = Setup(bull, float(oP), int(oB), float(eP), int(ext), i)
                if c["alertForm"]:
                    ev.append(snap("forming", i, cur, "BUY" if bull else "SELL"))
    return ev


def forex_closed(now: float) -> bool:
    t = time.gmtime(now)
    return t.tm_wday == 5 or (t.tm_wday == 4 and t.tm_hour >= 22) or (t.tm_wday == 6 and t.tm_hour < 21)


def make_plan(ev: dict, sym: str, cfg: dict):
    """Entry 1 / Entry 2 / SL / TP1 / TP2 and lot sizes for a fixed max loss."""
    t, st = cfg["trade"], cfg["settings"]
    bull = ev["dir"] == "BUY"
    o, x = ev["origin"], ev["ext"]
    px = lambda r: x + r * (o - x)
    e1, e2 = px(t["entry1"]), px(t["entry2"])
    buf = st["endBuf"] * ev["atr"]
    sl = px(st["zoneEnd"]) - buf if bull else px(st["zoneEnd"]) + buf
    tp1, tp2 = px(t["tp1"]), px(t["tp2"])
    d = 1 if bull else -1
    r1, r2 = d * (e1 - sl), d * (e2 - sl)          # loss per unit at each entry
    if r1 <= 0 or r2 <= 0:
        return None
    w1, w2 = t["split"]
    k = t["risk_usd"] / (w1 * r1 + w2 * r2)         # units so that total loss = risk_usd
    cs = t["contract_size"].get(sym, t["contract_size"].get("default", 1))
    step = t["lot_step"]
    fl = lambda q: round(math.floor(q / cs / step + 1e-9) * step, 4)
    lots1, lots2 = fl(k * w1), fl(k * w2)
    q1, q2 = lots1 * cs, lots2 * cs
    return dict(e1=e1, e2=e2, sl=sl, tp1=tp1, tp2=tp2, r1=t["entry1"], r2=t["entry2"],
                t1=t["tp1"], t2=t["tp2"], lots1=lots1, lots2=lots2,
                risk=q1 * r1 + q2 * r2, risk1=q1 * r1,
                pt1=d * (q1 * (tp1 - e1) + q2 * (tp1 - e2)),
                pt2=d * (q1 * (tp2 - e1) + q2 * (tp2 - e2)),
                low_size=min(lots1, lots2) < t["min_lot"])


def send_event(df, ev, cfg, sym, tfk, bin_tf, secs):
    s = cfg["settings"]
    cut = df.iloc[:ev["bar"] + 1].tail(CANDLES)
    ms = lambda b: int(df.index[b].value // 1_000_000)
    if ev["type"] == "bos":
        a = Alert(secret="", symbol=sym, tf=tfk, event="bos", direction=ev["dir"], price=ev["price"])
    else:
        a = Alert(secret="", symbol=sym, tf=tfk, event=ev["type"], direction=ev["dir"],
                  price=ev["price"], origin=ev["origin"], ext=ev["ext"],
                  origin_t=ms(ev["originB"]), ext_t=ms(ev["extB"]), entry_r=s["entryRatio"],
                  zone_r=s["zoneEnd"], ratios=[r for r in s["ratios"] if r > 0],
                  plan=make_plan(ev, sym, cfg) if cfg.get("trade") else None)
    caption = build_caption(a, sym, bin_tf)
    try:
        img = render_chart(cut, a, sym, bin_tf, secs)
        tg("sendPhoto", data={"chat_id": TG_CHAT_ID, "caption": caption},
           files={"photo": ("chart.png", img)})
    except Exception as e:
        tg("sendMessage", data={"chat_id": TG_CHAT_ID, "text": f"{caption}\n\n(chart failed: {e})"})


def process(w, cfg, state, test=False):
    sym, tfk = clean_symbol(w["symbol"]), str(w["tf"]).upper()
    info = TF_MAP.get(tfk)
    if not info:
        print(f"skip {sym}: unsupported timeframe {tfk}")
        return
    bin_tf, td_tf, secs = info
    crypto = sym.endswith(CRYPTO_QUOTES)
    key = f"{sym}|{tfk}"
    now = time.time()
    expected = None if tfk in ("W", "1W") else int(now // secs) * secs - secs
    if not test:
        if expected is not None and state["last"].get(key) == expected:
            return  # nothing new since last check
        if not crypto and forex_closed(now):
            return
    df = fetch_crypto(sym, bin_tf, HISTORY) if crypto else fetch_forex(sym, td_tf, HISTORY)
    if df.index[-1] + pd.Timedelta(seconds=secs) > pd.Timestamp(now, unit="s"):
        df = df.iloc[:-1]  # drop the still-forming candle
    if len(df) < 60:
        print(f"skip {key}: not enough candles")
        return
    events = run_engine(df, cfg["settings"])
    n = len(df)

    if test:
        if events:
            send_event(df, events[-1], cfg, sym, tfk, bin_tf, secs)
            print(f"{key}: sent test alert ({events[-1]['type']})")
        else:
            print(f"{key}: no events in the last {n} candles")
        return

    sent, sent_set = state["sent"], set(state["sent"])
    seeded = key in state["init"]
    for e in events:
        k = f"{key}|{e['type']}|{e['dir']}|{int(df.index[e['bar']].value // 1_000_000)}"
        if k in sent_set:
            continue
        sent.append(k)
        sent_set.add(k)
        if seeded and (n - 1 - e["bar"]) * secs <= cfg.get("max_age_minutes", 90) * 60:
            try:
                send_event(df, e, cfg, sym, tfk, bin_tf, secs)
                print(f"{key}: sent {e['type']} {e['dir']}")
            except Exception as ex:
                sent.remove(k)
                sent_set.discard(k)  # retry next run
                print(f"{key}: send failed: {ex}", file=sys.stderr)
    if not seeded:
        state["init"].append(key)
        print(f"{key}: first run, existing history marked as seen (no alerts sent)")
    if expected is not None:
        state["last"][key] = expected
    state["sent"] = sent[-1500:]


def run_once(test=False):
    cfg = json.load(open(CFG_PATH))
    state = {"init": [], "last": {}, "sent": []}
    if os.path.exists(STATE_PATH) and not test:
        state.update(json.load(open(STATE_PATH)))
    for w in cfg["watch"]:
        try:
            process(w, cfg, state, test)
        except Exception as e:
            print(f"{w}: error: {e}", file=sys.stderr)
    if not test:
        json.dump(state, open(STATE_PATH, "w"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", type=int, metavar="SECONDS")
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    if args.loop:
        while True:
            run_once()
            time.sleep(args.loop)
    else:
        run_once(args.test)
