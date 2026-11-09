"""BOS Reversal Fib: free auto-alert scanner + level tracker (no TradingView needed).

On every run it (1) replays your indicator on recent closed candles and sends any NEW
signal to Telegram with a chart, and (2) follows each open signal on 1-minute candles and
messages you when Entry 1, Entry 2, TP1, TP2 or the stop loss is touched.
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
    plan = None
    if ev["type"] == "bos":
        a = Alert(secret="", symbol=sym, tf=tfk, event="bos", direction=ev["dir"], price=ev["price"])
    else:
        plan = make_plan(ev, sym, cfg) if cfg.get("trade") else None
        a = Alert(secret="", symbol=sym, tf=tfk, event=ev["type"], direction=ev["dir"],
                  price=ev["price"], origin=ev["origin"], ext=ev["ext"],
                  origin_t=ms(ev["originB"]), ext_t=ms(ev["extB"]), entry_r=s["entryRatio"],
                  zone_r=s["zoneEnd"], ratios=[r for r in s["ratios"] if r > 0], plan=plan)
    caption = build_caption(a, sym, bin_tf)
    try:
        img = render_chart(cut, a, sym, bin_tf, secs)
        tg("sendPhoto", data={"chat_id": TG_CHAT_ID, "caption": caption},
           files={"photo": ("chart.png", img)})
    except Exception as e:
        tg("sendMessage", data={"chat_id": TG_CHAT_ID, "text": f"{caption}\n\n(chart failed: {e})"})
    return plan


# ───────────────────────── level tracking ─────────────────────────
def fetch_candles(sym, tfk, limit):
    bin_tf, td_tf, _ = TF_MAP[tfk]
    return fetch_crypto(sym, bin_tf, limit) if sym.endswith(CRYPTO_QUOTES) else fetch_forex(sym, td_tf, limit)


def drop_forming(df, secs, now):
    if df.index[-1] + pd.Timedelta(seconds=secs) > pd.Timestamp(now, unit="s"):
        return df.iloc[:-1]
    return df


def register_trade(state, cfg, df, e, plan, sym, tfk, bin_tf, secs):
    setup_ms = int(df.index[e["bar"]].value // 1_000_000)
    tid = f"{sym}|{tfk}|{setup_ms}"
    if any(t["id"] == tid for t in state["trades"]):
        return
    state["trades"].append(dict(
        id=tid, sym=sym, bin_tf=bin_tf, source="crypto" if sym.endswith(CRYPTO_QUOTES) else "twelve",
        d=1 if e["dir"] == "BUY" else -1,
        e1=float(plan["e1"]), e2=float(plan["e2"]), sl=float(plan["sl"]),
        tp1=float(plan["tp1"]), tp2=float(plan["tp2"]),
        lots1=float(plan["lots1"]), lots2=float(plan["lots2"]),
        f1=False, f2=False, tp1_hit=False, closed=False,
        next_ms=setup_ms + secs * 1000,
        expire_ms=setup_ms + cfg["settings"]["maxBars"] * secs * 1000))


def simulate(tr, df):
    """Walk closed 1-minute candles in order. Returns ([(message, state_after)], final_state|None)."""
    d, cur, out = tr["d"], dict(tr), []
    f = lambda x: f"{x:.6g}"
    head = f"{'🟢' if d == 1 else '🔴'} {tr['sym']} · {tr['bin_tf']} {'BUY' if d == 1 else 'SELL'}"
    last_t = None

    def emit(text, t_ms, close=False):
        snap = dict(cur)
        snap["next_ms"], snap["closed"] = t_ms + 60_000, close
        out.append((f"{head}\n{text}", snap))

    levels = lambda: f"SL {f(cur['sl'])} | TP1 {f(cur['tp1'])} | TP2 {f(cur['tp2'])}"
    for ts, row in df.iterrows():
        t_ms = int(ts.value // 1_000_000)
        if t_ms < cur["next_ms"]:
            continue
        last_t = t_ms
        hi, lo = float(row["High"]), float(row["Low"])
        touch_e = (lambda p: lo <= p) if d == 1 else (lambda p: hi >= p)
        touch_t = (lambda p: hi >= p) if d == 1 else (lambda p: lo <= p)
        hit_sl = (lo <= cur["sl"]) if d == 1 else (hi >= cur["sl"])
        filled_now = False

        if not cur["f1"]:
            if t_ms >= cur["expire_ms"]:
                emit("⌛ Setup expired and Entry 1 was never touched. Cancel your pending orders.", t_ms, True)
                break
            if touch_e(cur["e1"]):
                cur["f1"] = filled_now = True
                emit(f"✅ Entry 1 touched @ {f(cur['e1'])} ({cur['lots1']:g} lots)\n"
                     f"Entry 2 pending @ {f(cur['e2'])} ({cur['lots2']:g} lots)\n{levels()}", t_ms)
            elif touch_t(cur["tp1"]):
                emit("⚠️ TP1 was reached before Entry 1 was touched. Cancel your pending orders, "
                     "this setup is missed.", t_ms, True)
                break
        if cur["f1"] and not cur["f2"] and not cur["tp1_hit"] and touch_e(cur["e2"]):
            cur["f2"] = filled_now = True
            tot = cur["lots1"] + cur["lots2"]
            avg = (cur["lots1"] * cur["e1"] + cur["lots2"] * cur["e2"]) / tot
            emit(f"✅ Entry 2 touched @ {f(cur['e2'])} ({cur['lots2']:g} lots)\n"
                 f"Both entries in: {tot:g} lots, average entry {f(avg)}\n{levels()}", t_ms)
        if not cur["f1"]:
            continue

        target = cur["tp2"] if cur["tp1_hit"] else cur["tp1"]
        tp_hit = (not filled_now) and touch_t(target)  # a target is judged from the next candle on
        if hit_sl and tp_hit:
            emit("⚠️ Stop loss and take profit were both reached inside the same 1-minute candle, "
                 "so the order is unclear. Check your chart. Tracking stopped for this trade.", t_ms, True)
            break
        if hit_sl:
            emit(f"🛑 Stop loss touched @ {f(cur['sl'])}", t_ms, True)
            break
        if tp_hit and not cur["tp1_hit"]:
            cur["tp1_hit"] = True
            note = "" if cur["f2"] else "\nEntry 2 was not touched: cancel that pending order."
            emit(f"🎯 TP1 touched @ {f(cur['tp1'])}\nNext: TP2 {f(cur['tp2'])} | SL {f(cur['sl'])}{note}", t_ms)
            tp_hit = touch_t(cur["tp2"])  # a big candle can reach TP2 as well
        if tp_hit and cur["tp1_hit"]:
            emit(f"🎯🎯 TP2 touched @ {f(cur['tp2'])}", t_ms, True)
            break

    if out and out[-1][1]["closed"]:
        return out, None
    final = dict(cur)
    if last_t is not None:
        final["next_ms"] = last_t + 60_000
    return out, final


def track_trades(state):
    now = time.time()
    for tr in state["trades"]:
        try:
            if tr.get("closed") or tr.get("source") not in ("crypto", "twelve"):
                tr["closed"] = True
                continue
            now_ms = int(now * 1000)
            if now_ms > tr["next_ms"] + 7 * 86400 * 1000:
                tr["closed"] = True  # stale, stop following
                continue
            if now_ms < tr["next_ms"] + 60_000:
                continue  # no new closed 1-minute candle yet
            if tr["source"] == "twelve" and forex_closed(now):
                continue
            need = min(max(int((now_ms - tr["next_ms"]) // 60_000) + 3, 5), 1000)
            df = drop_forming(fetch_candles(tr["sym"], "1", need), 60, now)
            events, final = simulate(tr, df)
            for msg, snap in events:
                tg("sendMessage", data={"chat_id": TG_CHAT_ID, "text": msg})
                tr.clear()
                tr.update(snap)
                print(f"{tr['id']}: {msg.splitlines()[1][:40]}")
            if final is not None:
                tr.clear()
                tr.update(final)
        except Exception as ex:
            print(f"track {tr.get('id')}: error: {ex}", file=sys.stderr)
    state["trades"] = [t for t in state["trades"] if not t.get("closed")]


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
                plan = send_event(df, e, cfg, sym, tfk, bin_tf, secs)
                print(f"{key}: sent {e['type']} {e['dir']}")
                if e["type"] == "forming" and plan and cfg.get("trade", {}).get("track", True):
                    register_trade(state, cfg, df, e, plan, sym, tfk, bin_tf, secs)
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
    state = {"init": [], "last": {}, "sent": [], "trades": []}
    if os.path.exists(STATE_PATH) and not test:
        state.update(json.load(open(STATE_PATH)))
    for w in cfg["watch"]:
        try:
            process(w, cfg, state, test)
        except Exception as e:
            print(f"{w}: error: {e}", file=sys.stderr)
    if not test:
        track_trades(state)
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
