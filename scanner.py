"""BOS Reversal Fib: free auto-alert scanner + trade tracker (no TradingView needed).

Each run:
  1. replays your indicator on recent closed candles and sends NEW signals (with chart),
  2. follows every open signal on 1-minute candles and messages you when
     Entry 1 / Entry 2 fill, TP1 / TP2 are hit, or the stop loss is hit.
  python scanner.py              one pass (GitHub Actions)
  python scanner.py --loop 60    keep running on your own PC
  python scanner.py --test       send the latest signal per symbol, to test Telegram
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

from common import (Alert, CANDLES, TF_MAP, TG_CHAT_ID, clean_symbol, fetch_any,
                    render_chart, build_caption, source_for, tg)

HERE = os.path.dirname(os.path.abspath(__file__))
CFG_PATH = os.path.join(HERE, "config.json")
STATE_PATH = os.path.join(HERE, "state.json")
HISTORY = 1000  # candles replayed each run


# ───────────────────────── indicator port ─────────────────────────
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
                    ext=s.extP, originB=s.originB, extB=s.extB, atr=float(atr[i]), setup=s.bosBar)

    for i in range(n):
        if i >= 2 * P:  # swing pivots, confirmed P bars after they form
            m = i - P
            if H[m] > H[m - P:m].max() and H[m] >= H[m + 1:i + 1].max():
                pHi, pHiDone = H[m], False
            if L[m] < L[m - P:m].min() and L[m] <= L[m + 1:i + 1].min():
                pLo, pLoDone = L[m], False

        bosDir, brk = 0, None  # BOS on bar close
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

        if cur is not None:  # manage the live setup
            just = False
            if not cur.triggered:
                if cur.bull:
                    if H[i] > cur.extP:
                        cur.extP, cur.extB = H[i], i
                        ev.append(snap("shift", i, cur, "BUY"))
                    elif L[i] <= cur.px(c["entryRatio"]):
                        cur.triggered = just = True
                else:
                    if L[i] < cur.extP:
                        cur.extP, cur.extB = L[i], i
                        ev.append(snap("shift", i, cur, "SELL"))
                    elif H[i] >= cur.px(c["entryRatio"]):
                        cur.triggered = just = True
            if just and c["alertZone"]:
                ev.append(snap("touched", i, cur, "BUY" if cur.bull else "SELL"))
            buf = c["endBuf"] * atr[i]
            end = cur.px(c["zoneEnd"]) - buf if cur.bull else cur.px(c["zoneEnd"]) + buf
            broken = C[i] < end if cur.bull else C[i] > end
            if broken or i - cur.bosBar >= c["maxBars"]:
                cur = None

        for flip, bull, enabled in ((flipBuy, True, c["showBuy"]), (flipSell, False, c["showSell"])):
            if not (flip and enabled):  # new setup on a reversal BOS
                continue
            lo = max(i - min(i - bos1Bar, 4999), 0)
            seg = range(lo, i + 1)
            if bull:
                oB = min(seg, key=lambda b: (L[b], -b))
                ext = max(range(oB, i + 1), key=lambda b: (H[b], b))
                valid, oP, eP = H[ext] > L[oB], L[oB], H[ext]
            else:
                oB = max(seg, key=lambda b: (H[b], b))
                ext = min(range(oB, i + 1), key=lambda b: (L[b], -b))
                valid, oP, eP = H[oB] > L[ext], H[oB], L[ext]
            if valid and (c["minLegAtr"] == 0 or abs(oP - eP) >= c["minLegAtr"] * atr[i]):
                cur = Setup(bull, float(oP), int(oB), float(eP), int(ext), i)
                if c["alertForm"]:
                    ev.append(snap("forming", i, cur, "BUY" if bull else "SELL"))
    return ev


# ───────────────────────── helpers ─────────────────────────
def forex_closed(now: float) -> bool:
    t = time.gmtime(now)
    return t.tm_wday == 5 or (t.tm_wday == 4 and t.tm_hour >= 22) or (t.tm_wday == 6 and t.tm_hour < 21)


def ms_at(df: pd.DataFrame, b: int) -> int:
    return int(df.index[b].value // 1_000_000)


def get_candles(sym: str, tfk: str, limit: int, cfg: dict) -> pd.DataFrame:
    df = fetch_any(sym, tfk, limit)
    off = cfg.get("offsets", {}).get(sym, 0.0)
    if off:  # manual price offset to line the feed up with your chart
        df = df.copy()
        df[["Open", "High", "Low", "Close"]] += off
    return df


def drop_forming(df: pd.DataFrame, secs: int, now: float) -> pd.DataFrame:
    if df.index[-1] + pd.Timedelta(seconds=secs) > pd.Timestamp(now, unit="s"):
        return df.iloc[:-1]
    return df


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
    r1, r2 = d * (e1 - sl), d * (e2 - sl)
    if r1 <= 0 or r2 <= 0:
        return None
    w1, w2 = t["split"]
    k = t["risk_usd"] / (w1 * r1 + w2 * r2)
    cs = t["contract_size"].get(sym, t["contract_size"].get("default", 1))
    step = t["lot_step"]
    fl = lambda q: round(math.floor(q / cs / step + 1e-9) * step, 4)
    lots1, lots2 = fl(k * w1), fl(k * w2)
    q1, q2 = lots1 * cs, lots2 * cs
    return dict(e1=e1, e2=e2, sl=sl, tp1=tp1, tp2=tp2, r1=t["entry1"], r2=t["entry2"],
                t1=t["tp1"], t2=t["tp2"], lots1=lots1, lots2=lots2, cs=cs,
                risk=q1 * r1 + q2 * r2, risk1=q1 * r1,
                pt1=d * (q1 * (tp1 - e1) + q2 * (tp1 - e2)),
                pt2=d * (q1 * (tp2 - e1) + q2 * (tp2 - e2)),
                low_size=min(lots1, lots2) < t["min_lot"])


def send_event(df, ev, cfg, sym, tfk, bin_tf, secs):
    """Send a signal (chart + text). Returns the trade plan, if any."""
    s = cfg["settings"]
    cut = df.iloc[:ev["bar"] + 1].tail(CANDLES)
    kind = "updated" if ev["type"] == "shift" else ev["type"]
    plan = None
    if ev["type"] == "bos":
        a = Alert(secret="", symbol=sym, tf=tfk, event="bos", direction=ev["dir"], price=ev["price"])
    else:
        plan = make_plan(ev, sym, cfg) if cfg.get("trade") else None
        a = Alert(secret="", symbol=sym, tf=tfk, event=kind, direction=ev["dir"],
                  price=ev["price"], origin=ev["origin"], ext=ev["ext"],
                  origin_t=ms_at(df, ev["originB"]), ext_t=ms_at(df, ev["extB"]),
                  entry_r=s["entryRatio"], zone_r=s["zoneEnd"],
                  ratios=[r for r in s["ratios"] if r > 0], plan=plan)
    caption = build_caption(a, sym, bin_tf)
    try:
        img = render_chart(cut, a, sym, bin_tf, secs)
        tg("sendPhoto", data={"chat_id": TG_CHAT_ID, "caption": caption},
           files={"photo": ("chart.png", img)})
    except Exception as e:
        tg("sendMessage", data={"chat_id": TG_CHAT_ID, "text": f"{caption}\n\n(chart failed: {e})"})
    return plan


# ───────────────────────── trade tracking ─────────────────────────
def register_trade(state, cfg, df, e, plan, sym, tfk, bin_tf, secs):
    setup_ms = ms_at(df, e["bar"])
    tid = f"{sym}|{tfk}|{setup_ms}"
    if any(t["id"] == tid for t in state["trades"]):
        return
    state["trades"].append(dict(
        id=tid, sym=sym, tfk=tfk, bin_tf=bin_tf, secs=secs, source=source_for(sym),
        d=1 if e["dir"] == "BUY" else -1, cs=float(plan["cs"]),
        e1=float(plan["e1"]), e2=float(plan["e2"]), sl=float(plan["sl"]),
        tp1=float(plan["tp1"]), tp2=float(plan["tp2"]),
        lots1=float(plan["lots1"]), lots2=float(plan["lots2"]),
        f1=False, f2=False, tp1_hit=False, pnl=0.0, closed=False,
        setup_ms=setup_ms, lvl_ms=setup_ms, next_ms=setup_ms + secs * 1000,
        expire_ms=setup_ms + cfg["settings"]["maxBars"] * secs * 1000))


def pnl_open(tr, price, frac=1.0):
    """P/L (USDT) of the filled quantity if closed at `price`; frac = share still open."""
    tot = 0.0
    for filled, lots, e in ((tr["f1"], tr["lots1"], tr["e1"]), (tr["f2"], tr["lots2"], tr["e2"])):
        if filled:
            tot += lots * tr["cs"] * tr["d"] * (price - e)
    return tot * frac


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
                emit("⌛ Setup expired and Entry 1 never filled. Cancel your pending orders.", t_ms, True)
                break
            if touch_e(cur["e1"]):
                cur["f1"] = filled_now = True
                emit(f"✅ Entry 1 filled @ {f(cur['e1'])} ({cur['lots1']:g} lots)\n"
                     f"Entry 2 still pending @ {f(cur['e2'])} ({cur['lots2']:g} lots)\n{levels()}", t_ms)
            elif touch_t(cur["tp1"]):
                emit("⚠️ TP1 was reached before Entry 1 filled. Cancel your pending orders, "
                     "this setup is missed.", t_ms, True)
                break
        if cur["f1"] and not cur["f2"] and not cur["tp1_hit"] and touch_e(cur["e2"]):
            cur["f2"] = filled_now = True
            tot = cur["lots1"] + cur["lots2"]
            avg = (cur["lots1"] * cur["e1"] + cur["lots2"] * cur["e2"]) / tot
            emit(f"✅ Entry 2 filled @ {f(cur['e2'])} ({cur['lots2']:g} lots)\n"
                 f"Both entries filled: {tot:g} lots, average entry {f(avg)}\n{levels()}", t_ms)
        if not cur["f1"]:
            continue

        target = cur["tp2"] if cur["tp1_hit"] else cur["tp1"]
        tp_hit = (not filled_now) and touch_t(target)  # TP is judged from the next candle on
        if hit_sl and tp_hit:
            emit("⚠️ Stop loss and take profit were both reached inside the same 1-minute candle, "
                 "so the order is unclear. Check your chart. Tracking stopped for this trade.", t_ms, True)
            break
        if hit_sl:
            loss = pnl_open(cur, cur["sl"], 0.5 if cur["tp1_hit"] else 1.0)
            total = cur["pnl"] + loss
            cur["pnl"] = total
            emit(f"🛑 Stop loss hit @ {f(cur['sl'])}\nThis part: {loss:+.0f} USDT | "
                 f"Trade total: {total:+.0f} USDT", t_ms, True)
            break
        if tp_hit and not cur["tp1_hit"]:
            cur["tp1_hit"] = True
            gain = pnl_open(cur, cur["tp1"], 0.5)
            cur["pnl"] += gain
            half = round((cur["lots1"] * cur["f1"] + cur["lots2"] * cur["f2"]) / 2, 2)
            note = "" if cur["f2"] else "\nEntry 2 never filled: cancel that pending order."
            emit(f"🎯 TP1 hit @ {f(cur['tp1'])}\nClose half your position ({half:g} lots). "
                 f"Profit so far: {gain:+.0f} USDT\nThe rest runs to TP2 ({f(cur['tp2'])}); "
                 f"stop loss stays at {f(cur['sl'])}.{note}", t_ms)
            tp_hit = touch_t(cur["tp2"])  # a big candle can reach TP2 as well
        if tp_hit and cur["tp1_hit"]:
            gain = pnl_open(cur, cur["tp2"], 0.5)
            cur["pnl"] += gain
            emit(f"🎯🎯 TP2 hit @ {f(cur['tp2'])}\nClose the rest. Trade total: {cur['pnl']:+.0f} USDT",
                 t_ms, True)
            break

    if out and out[-1][1]["closed"]:
        return out, None
    final = dict(cur)
    if last_t is not None:
        final["next_ms"] = last_t + 60_000
    return out, final


def track_trades(cfg, state):
    now = time.time()
    for tr in state["trades"]:
        try:
            if tr.get("closed"):
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
            df = drop_forming(get_candles(tr["sym"], "1", need, cfg), 60, now)
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


# ───────────────────────── scanning ─────────────────────────
def process(w, cfg, state, test=False):
    sym, tfk = clean_symbol(w["symbol"]), str(w["tf"]).upper()
    info = TF_MAP.get(tfk)
    if not info:
        print(f"skip {sym}: unsupported timeframe {tfk}")
        return
    bin_tf, td_tf, secs = info
    key = f"{sym}|{tfk}"
    now = time.time()
    expected = None if tfk in ("W", "1W") else int(now // secs) * secs - secs
    if not test:
        if expected is not None and state["last"].get(key) == expected:
            return  # nothing new since last check
        if source_for(sym) == "twelve" and forex_closed(now):
            return
    df = drop_forming(get_candles(sym, tfk, HISTORY, cfg), secs, now)
    if len(df) < 60:
        print(f"skip {key}: not enough candles")
        return
    events = run_engine(df, cfg["settings"])
    n = len(df)

    if test:
        real = [e for e in events if e["type"] != "shift"]
        if real:
            send_event(df, real[-1], cfg, sym, tfk, bin_tf, secs)
            print(f"{key}: sent test alert ({real[-1]['type']})")
        else:
            print(f"{key}: no events in the last {n} candles")
        return

    max_age = cfg.get("max_age_minutes", 90) * 60
    track = cfg.get("trade", {}).get("track", True)
    sent, sent_set = state["sent"], set(state["sent"])
    seeded = key in state["init"]
    for e in events:
        if e["type"] == "shift":
            continue
        k = f"{key}|{e['type']}|{e['dir']}|{ms_at(df, e['bar'])}"
        if k in sent_set:
            continue
        sent.append(k)
        sent_set.add(k)
        if seeded and (n - 1 - e["bar"]) * secs <= max_age:
            try:
                plan = send_event(df, e, cfg, sym, tfk, bin_tf, secs)
                print(f"{key}: sent {e['type']} {e['dir']}")
                if e["type"] == "forming" and plan and track:
                    register_trade(state, cfg, df, e, plan, sym, tfk, bin_tf, secs)
            except Exception as ex:
                sent.remove(k)
                sent_set.discard(k)  # retry next run
                print(f"{key}: send failed: {ex}", file=sys.stderr)

    # the indicator moved a pending setup: tell the user to replace the orders
    last_shift = {}
    for e in events:
        if e["type"] == "shift":
            last_shift[e["setup"]] = e
    for setup_bar, e in last_shift.items():
        tr = next((t for t in state["trades"] if t["id"] == f"{key}|{ms_at(df, setup_bar)}"), None)
        if not tr or tr["f1"] or tr.get("closed") or ms_at(df, e["bar"]) <= tr["lvl_ms"]:
            continue
        newp = make_plan(e, sym, cfg)
        if not newp:
            continue
        thresh = cfg["trade"].get("shift_notify", 0.25) * abs(tr["e1"] - tr["sl"])
        if abs(newp["e1"] - tr["e1"]) < thresh or (n - 1 - e["bar"]) * secs > max_age:
            continue
        try:
            send_event(df, e, cfg, sym, tfk, bin_tf, secs)
            for fld in ("e1", "e2", "sl", "tp1", "tp2", "lots1", "lots2"):
                tr[fld] = newp[fld]
            tr["lvl_ms"] = ms_at(df, e["bar"])
            print(f"{key}: sent updated levels")
        except Exception as ex:
            print(f"{key}: update failed: {ex}", file=sys.stderr)

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
        track_trades(cfg, state)
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
