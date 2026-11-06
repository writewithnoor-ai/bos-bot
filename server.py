"""Optional: TradingView webhook receiver -> chart screenshot -> Telegram."""
import os
import time

import pandas as pd
from fastapi import FastAPI, HTTPException

from common import (Alert, CRYPTO_QUOTES, SETTLE_DELAY, TF_MAP, TG_CHAT_ID, build_caption,
                    clean_symbol, fetch_crypto, fetch_forex, render_chart, tg)

SECRET = os.environ["WEBHOOK_SECRET"]
app = FastAPI()


@app.post("/webhook")
def webhook(a: Alert):
    if a.secret != SECRET:
        raise HTTPException(401, "bad secret")
    sym = clean_symbol(a.symbol)
    info = TF_MAP.get(a.tf.upper())
    if not info:
        raise HTTPException(400, f"unsupported timeframe {a.tf}")
    bin_tf, td_tf, secs = info
    caption = build_caption(a, sym, bin_tf)
    try:
        time.sleep(SETTLE_DELAY)
        df = fetch_crypto(sym, bin_tf) if sym.endswith(CRYPTO_QUOTES) else fetch_forex(sym, td_tf)
        now = pd.Timestamp.now(tz="UTC").tz_localize(None)
        if df.index[-1] + pd.Timedelta(seconds=secs) > now:
            df = df.iloc[:-1]
        img = render_chart(df, a, sym, bin_tf, secs)
        tg("sendPhoto", data={"chat_id": TG_CHAT_ID, "caption": caption},
           files={"photo": ("chart.png", img)})
    except Exception as e:
        tg("sendMessage", data={"chat_id": TG_CHAT_ID, "text": f"{caption}\n\n(chart failed: {e})"})
    return {"ok": True}


@app.get("/health")
def health():
    return {"ok": True}
