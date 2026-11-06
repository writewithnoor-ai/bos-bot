# BOS Reversal Fib: free auto-alert bot (Telegram)

`scanner.py` is a Python copy of your indicator. It checks the market itself, so TradingView webhooks / paid plan are NOT needed.

## Setup A: GitHub Actions (free, runs without your PC)
1. Create a Telegram bot with @BotFather, copy the token. Message your bot once, then open
   https://api.telegram.org/bot<TOKEN>/getUpdates and copy the chat id.
2. (Forex only) free key from twelvedata.com.
3. Create a new GitHub repo (PUBLIC = unlimited free minutes) and upload every file in this folder,
   including the `.github/workflows/scan.yml` file and `config.json`.
4. Repo -> Settings -> Secrets and variables -> Actions -> New repository secret. Add:
   TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, TWELVEDATA_API_KEY
5. Repo -> Settings -> Actions -> General -> Workflow permissions -> "Read and write permissions" -> Save.
6. Actions tab -> "scan" -> Run workflow. The first run only records current history (no alerts).
   From then on it runs about every 5 minutes and sends new setups.
Private repo: you get 2,000 free minutes/month, so change the cron in scan.yml to `*/30 * * * *`.

## Setup B: your own PC (free)
pip install -r requirements.txt
set TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID (and TWELVEDATA_API_KEY) in a .env file
python scanner.py --test        # sends the latest past setup per symbol, to confirm Telegram works
python scanner.py --loop 60     # checks every 60 seconds while the PC is on

## Change what it watches
Edit `config.json`: `watch` (symbol + timeframe: 1, 5, 15, 30, 60, 240, D) and `settings`
(same names and defaults as your indicator inputs). Forex 15m timeframe or lower on the free
Twelve Data plan is limited to 800 requests/day, so keep the list short.

## Optional: webhook mode
`server.py` still works if you later get TradingView webhooks (see pine_patch.md).
