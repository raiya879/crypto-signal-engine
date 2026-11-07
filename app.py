import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import json
import os
import time
from datetime import datetime

st.set_page_config(page_title="Crypto Quant Signal Engine", page_icon="⚡", layout="wide")

exchange = ccxt.kucoinfutures({'enableRateLimit': True, 'timeout': 10000})
LOG_FILE = "top20_signals_history.json"

DEFAULT_TOP_20 = [
    "BTC", "ETH", "SOL", "XRP", "DOGE", 
    "ADA", "BNB", "SUI", "AVAX", "LINK", 
    "NEAR", "APT", "DOT", "LTC", "SHIB", 
    "PEPE", "UNI", "ICP", "RENDER", "FET"
]

def load_signals():
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_signal(record):
    signals = load_signals()
    signals.append(record)
    with open(LOG_FILE, "w") as f:
        json.dump(signals, f, indent=4)

def update_signals_db(signals_list):
    with open(LOG_FILE, "w") as f:
        json.dump(signals_list, f, indent=4)

def fetch_and_analyze_tf(symbol, timeframe):
    try:
        bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=220)
        if not bars or len(bars) < 30:
            return None
            
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # EMA
        ema_len = 200 if len(df) >= 200 else 50
        df['EMA_TREND'] = df['close'].ewm(span=ema_len, adjust=False).mean()

        # RSI (14)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0.0).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/14, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        df['RSI'] = 100 - (100 / (1 + rs))

        # ATR (14) & ADX (14)
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = tr.ewm(alpha=1/14, adjust=False).mean()

        up_move = df['high'].diff()
        down_move = -df['low'].diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_di = 100 * (pd.Series(plus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean() / df['ATR'].replace(0, np.nan))
        minus_di = 100 * (pd.Series(minus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean() / df['ATR'].replace(0, np.nan))
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        df['ADX'] = dx.ewm(alpha=1/14, adjust=False).mean()
        
        df = df.dropna(subset=['close'])
        if df.empty:
            return None
            
        latest = df.iloc[-1]
        ema_val = float(latest['EMA_TREND']) if pd.notna(latest['EMA_TREND']) else float(latest['close'])
        trend = "BULLISH" if float(latest['close']) > ema_val else "BEARISH"
        
        rsi_val = float(latest['RSI']) if pd.notna(latest['RSI']) else 50.0
        adx_val = float(latest['ADX']) if pd.notna(latest['ADX']) else 0.0
        atr_val = float(latest['ATR']) if pd.notna(latest['ATR']) else (float(latest['close']) * 0.015)
        
        return {
            "close": float(latest['close']),
            "rsi": rsi_val,
            "adx": adx_val,
            "atr": atr_val,
            "trend": trend
        }
    except Exception:
        return None

def scan_single_coin(coin):
    symbol = f"{coin}/USDT:USDT"
    tf_4h = fetch_and_analyze_tf(symbol, '4h')
    tf_1h = fetch_and_analyze_tf(symbol, '1h')
    tf_15m = fetch_and_analyze_tf(symbol, '15m')
    
    if not tf_4h or not tf_1h or not tf_15m:
        return None

    try:
        funding_rate = float(exchange.fetch_funding_rate(symbol).get('fundingRate', 0)) * 100
    except Exception:
        funding_rate = 0.01

    price = tf_15m['close']
    atr = tf_1h['atr']
    verdict = "WAIT"

    if tf_15m['adx'] >= 20 and tf_1h['adx'] >= 20:
        if tf_4h['trend'] == "BULLISH" and tf_1h['trend'] == "BULLISH":
            if 36 <= tf_15m['rsi'] <= 58 and funding_rate < 0.04:
                verdict = "LONG"
        elif tf_4h['trend'] == "BEARISH" and tf_1h['trend'] == "BEARISH":
            if 42 <= tf_15m['rsi'] <= 64 and funding_rate > -0.04:
                verdict = "SHORT"

    sl = round(price - (1.5 * atr), 4) if verdict == "LONG" else round(price + (1.5 * atr), 4)
    tp1 = round(price + (1.5 * atr), 4) if verdict == "LONG" else round(price - (1.5 * atr), 4)
    tp2 = round(price + (3.0 * atr), 4) if verdict == "LONG" else round(price - (3.0 * atr), 4)

    return {
        "coin": coin, "symbol": symbol, "price": price,
        "verdict": verdict, "sl": sl, "tp1": tp1, "tp2": tp2,
        "rsi": round(tf_15m['rsi'], 1), "adx": round(tf_15m['adx'], 1)
    }

def update_history():
    signals = load_signals()
    if not signals: 
        return
    for sig in signals:
        if sig['status'] in ["FULL WIN (TP2)", "PARTIAL WIN (TP1->BE)", "LOSS (SL Hit)"]:
            continue
        try:
            candles = exchange.fetch_ohlcv(sig['symbol'], timeframe='5m', since=sig['timestamp_ms'], limit=500)
            if not candles: 
                continue
            current_price = candles[-1][4]
            side, entry, sl, tp1, tp2 = sig['side'], sig['entry'], sig['sl'], sig['tp1'], sig['tp2']
            tp1_hit = sig.get('tp1_hit', False)
            status = "RUNNING"
            for c in candles:
                high, low = c[2], c[3]
                if side == "LONG":
                    active_sl = entry if tp1_hit else sl
                    if low <= active_sl:
                        status = "PARTIAL WIN (TP1->BE)" if tp1_hit else "LOSS (SL Hit)"
                        break
                    if high >= tp2:
                        status = "FULL WIN (TP2)"; break
                    if high >= tp1: tp1_hit = True
                elif side == "SHORT":
                    active_sl = entry if tp1_hit else sl
                    if high >= active_sl:
                        status = "PARTIAL WIN (TP1->BE)" if tp1_hit else "LOSS (SL Hit)"
                        break
                    if low <= tp2:
                        status = "FULL WIN (TP2)"; break
                    if low <= tp1: tp1_hit = True

            if status == "FULL WIN (TP2)":
                pnl = abs(tp2 - entry) / entry * 100
            elif status == "PARTIAL WIN (TP1->BE)":
                pnl = (abs(tp1 - entry) / entry * 100) / 2
            elif status == "LOSS (SL Hit)":
                pnl = - (abs(entry - sl) / entry * 100)
            else:
                pnl = (current_price - entry) / entry * 100 if side == "LONG" else (entry - current_price) / entry * 100

            sig['status'] = status
            sig['tp1_hit'] = tp1_hit
            sig['pnl_percent'] = round(pnl, 2)
        except Exception:
            pass
    update_signals_db(signals)

# UI Layout
st.title("⚡ Crypto Futures Confluence Scanner")
st.caption("Top 20 Institutional Multi-Timeframe Engine & Signal Tracker")

tab1, tab2 = st.tabs(["🚀 Active Market Signals", "📊 Historical Track Record"])

with tab1:
    if st.button("🔄 Scan Top 20 Now", use_container_width=True):
        status_box = st.empty()
        status_box.info("🔍 Connecting to KuCoin Futures & Scanning Top 20...")
        
        existing = load_signals()
        active_coins = [s['coin'] for s in existing if s.get('status') == "RUNNING"]
        found = []
        
        for i, coin in enumerate(DEFAULT_TOP_20):
            status_box.text(f"Scanning [{i+1}/20]: {coin}...")
            res = scan_single_coin(coin)
            if res and res['verdict'] in ["LONG", "SHORT"]:
                found.append(res)
                if coin not in active_coins:
                    rec = {
                        "id": len(existing) + 1, "coin": coin, "symbol": res['symbol'],
                        "side": res['verdict'], "entry": res['price'], "sl": res['sl'],
                        "tp1": res['tp1'], "tp2": res['tp2'],
                        "timestamp_ms": exchange.milliseconds(),
                        "time_str": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "status": "RUNNING", "tp1_hit": False, "pnl_percent": 0.0
                    }
                    save_signal(rec)
                    existing.append(rec)
                    active_coins.append(coin)
            time.sleep(0.1)
            
        status_box.empty()
        st.session_state['last_scan'] = found
        st.session_state['scan_completed'] = True

    found = st.session_state.get('last_scan', [])
    if found:
        for s in found:
            color = "#00e676" if s['verdict'] == "LONG" else "#ff5252"
            st.markdown(f"""
            <div style="padding:15px; border-radius:10px; border-left:6px solid {color}; background-color:#1e1e24; margin-bottom:12px;">
                <h3 style="margin:0; color:white;">{s['coin']} — <span style="color:{color};">{s['verdict']}</span> @ ${s['price']}</h3>
                <p style="margin:5px 0; color:#b0b0b0;"><b>Target 1:</b> ${s['tp1']} | <b>Target 2:</b> ${s['tp2']} | <b>Stop Loss:</b> ${s['sl']}</p>
                <small style="color:#808080;">RSI (15m): {s['rsi']} | ADX (15m): {s['adx']}</small>
            </div>
            """, unsafe_allow_html=True)
    elif st.session_state.get('scan_completed'):
        st.info("⏸️ All 20 coins are currently filtered out (Choppy market or conflicting trends). No setups right now.")

with tab2:
    if st.button("🔄 Refresh Outcomes & Win Rate"):
        with st.spinner("Replaying candles..."):
            update_history()

    signals = load_signals()
    if signals:
        df = pd.DataFrame(signals)
        closed = df[df['status'].isin(["FULL WIN (TP2)", "PARTIAL WIN (TP1->BE)", "LOSS (SL Hit)"])]
        wins = len(closed[closed['status'].str.contains("WIN")])
        total_closed = len(closed)
        win_rate = round((wins / total_closed * 100), 1) if total_closed > 0 else 0.0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Historical Win Rate", f"{win_rate}%")
        m2.metric("Total Wins", wins)
        m3.metric("Total Losses", total_closed - wins)
        m4.metric("Active (Running)", len(df[df['status'] == "RUNNING"]))

        display_df = df[['id', 'time_str', 'coin', 'side', 'entry', 'tp1', 'tp2', 'sl', 'status', 'pnl_percent']].copy()
        display_df.columns = ['ID', 'Time', 'Coin', 'Type', 'Entry ($)', 'TP1 ($)', 'TP2 ($)', 'SL ($)', 'Status', 'PnL % (1x)']
        st.dataframe(display_df, use_container_width=True)
    else:
        st.write("No recorded signals yet.")
