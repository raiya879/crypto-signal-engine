import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import json
import os
import time
from datetime import datetime

st.set_page_config(page_title="Crypto Quant Signal Engine V2", page_icon="⚡", layout="wide")

exchange = ccxt.kucoinfutures({'enableRateLimit': True, 'timeout': 10000})
LOG_FILE = "top20_signals_history.json"

DEFAULT_TOP_20 = [
    "BTC", "ETH", "SOL", "XRP", "DOGE", 
    "ADA", "BNB", "SUI", "AVAX", "LINK", 
    "NEAR", "APT", "DOT", "LTC", "SHIB", 
    "PEPE", "UNI", "ICP", "RENDER", "FET"
]

def format_price(val):
    if val is None:
        return "0.0000"
    val = float(val)
    if val < 0.001:
        return f"{val:.8f}"
    elif val < 1:
        return f"{val:.5f}"
    else:
        return f"{val:.4f}"

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

def is_in_cooldown(coin, existing_signals, cooldown_hours=4):
    now_ms = exchange.milliseconds()
    cooldown_ms = cooldown_hours * 3600 * 1000
    for s in reversed(existing_signals):
        if s.get('coin') == coin and s.get('status') == "LOSS (SL Hit)":
            if now_ms - s.get('timestamp_ms', 0) < cooldown_ms:
                return True
            break
    return False

def fetch_and_analyze_tf(symbol, timeframe):
    try:
        bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=220)
        if not bars or len(bars) < 55:
            return None
            
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # EMAs (20, 50, 200)
        df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()
        ema_len = 200 if len(df) >= 200 else 100
        df['EMA200'] = df['close'].ewm(span=ema_len, adjust=False).mean()

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

        # Volume Moving Average (20)
        df['VOL_MA20'] = df['volume'].rolling(20).mean()
        
        df = df.dropna(subset=['close'])
        if len(df) < 5:
            return None
            
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        close_val = float(latest['close'])
        open_val = float(latest['open'])
        ema20_val = float(latest['EMA20'])
        ema50_val = float(latest['EMA50'])
        ema200_val = float(latest['EMA200']) if pd.notna(latest['EMA200']) else close_val
        
        if close_val > ema50_val and ema50_val >= ema200_val * 0.998:
            trend = "BULLISH"
        elif close_val < ema50_val and ema50_val <= ema200_val * 1.002:
            trend = "BEARISH"
        else:
            trend = "NEUTRAL"
        
        rsi_val = float(latest['RSI']) if pd.notna(latest['RSI']) else 50.0
        adx_val = float(latest['ADX']) if pd.notna(latest['ADX']) else 0.0
        atr_val = float(latest['ATR']) if pd.notna(latest['ATR']) else (close_val * 0.015)
        
        vol_ma = float(latest['VOL_MA20']) if pd.notna(latest['VOL_MA20']) and latest['VOL_MA20'] > 0 else 1.0
        rvol = float(latest['volume']) / vol_ma

        # Check last closed candle or current forming candle direction
        bull_candle = (close_val > open_val) or (float(prev['close']) > float(prev['open']))
        bear_candle = (close_val < open_val) or (float(prev['close']) < float(prev['open']))

        swing_low = float(df['low'].tail(12).min())
        swing_high = float(df['high'].tail(12).max())
        
        return {
            "close": close_val,
            "ema20": ema20_val,
            "ema50": ema50_val,
            "ema200": ema200_val,
            "rsi": rsi_val,
            "adx": adx_val,
            "atr": atr_val,
            "rvol": rvol,
            "trend": trend,
            "bull_candle": bull_candle,
            "bear_candle": bear_candle,
            "swing_low": swing_low,
            "swing_high": swing_high
        }
    except Exception:
        return None

def get_btc_regime():
    tf_1h = fetch_and_analyze_tf("BTC/USDT:USDT", '1h')
    tf_15m = fetch_and_analyze_tf("BTC/USDT:USDT", '15m')
    if not tf_1h or not tf_15m:
        return "NEUTRAL", 50.0
    if tf_15m['close'] > tf_15m['ema20'] and tf_15m['rsi'] > 55 and tf_1h['close'] > tf_1h['ema50']:
        return "STRONG_BULL", round(tf_15m['rsi'], 1)
    elif tf_15m['close'] < tf_15m['ema20'] and tf_15m['rsi'] < 45 and tf_1h['close'] < tf_1h['ema50']:
        return "STRONG_BEAR", round(tf_15m['rsi'], 1)
    return "NEUTRAL", round(tf_15m['rsi'], 1)

def scan_single_coin(coin, btc_regime):
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
    atr_1h = tf_1h['atr']
    atr_15m = tf_15m['atr']
    verdict = "WAIT"

    # Overextension Check: Do not chase if price is too far (> 2.2x ATR) from 15m EMA20
    dist_from_ema20 = abs(price - tf_15m['ema20'])
    not_overextended = dist_from_ema20 <= (2.2 * atr_15m)

    if tf_15m['adx'] >= 20 and tf_1h['adx'] >= 20 and not_overextended:
        # LONG CONFLUENCE
        if (tf_4h['trend'] == "BULLISH" and tf_1h['trend'] == "BULLISH" and 
            price >= tf_15m['ema20'] * 0.998 and tf_15m['ema20'] >= tf_15m['ema50'] * 0.998 and
            tf_15m['bull_candle'] and btc_regime != "STRONG_BEAR"):
            if 40 <= tf_15m['rsi'] <= 62 and funding_rate < 0.04:
                verdict = "LONG"
                
        # SHORT CONFLUENCE
        elif (tf_4h['trend'] == "BEARISH" and tf_1h['trend'] == "BEARISH" and 
              price <= tf_15m['ema20'] * 1.002 and tf_15m['ema20'] <= tf_15m['ema50'] * 1.002 and
              tf_15m['bear_candle'] and btc_regime != "STRONG_BULL"):
            if 38 <= tf_15m['rsi'] <= 60 and funding_rate > -0.04:
                verdict = "SHORT"

    # Confidence Score Calculation (0 - 100%)
    score = 60
    if tf_15m['adx'] >= 25 and tf_1h['adx'] >= 25:
        score += 10
    if tf_15m['rvol'] >= 1.15:
        score += 12
    elif tf_15m['rvol'] >= 0.85:
        score += 5
    if (verdict == "LONG" and btc_regime == "STRONG_BULL") or (verdict == "SHORT" and btc_regime == "STRONG_BEAR"):
        score += 10
    if (verdict == "LONG" and tf_15m['ema20'] > tf_15m['ema50'] > tf_15m['ema200']) or \
       (verdict == "SHORT" and tf_15m['ema20'] < tf_15m['ema50'] < tf_15m['ema200']):
        score += 8
    score = min(99, score)

    # Structure + ATR Smart Stop Loss & Targets
    decimals = 8 if price < 0.001 else (5 if price < 1 else 4)
    if verdict == "LONG":
        struct_sl = tf_15m['swing_low'] - (0.25 * atr_15m)
        atr_sl = price - (1.5 * atr_1h)
        raw_sl = max(struct_sl, atr_sl)
        if (price - raw_sl) < (0.8 * atr_15m):
            raw_sl = price - (1.2 * atr_1h)
        risk = abs(price - raw_sl)
        sl = round(raw_sl, decimals)
        tp1 = round(price + (1.5 * risk), decimals)
        tp2 = round(price + (3.0 * risk), decimals)
    else:
        struct_sl = tf_15m['swing_high'] + (0.25 * atr_15m)
        atr_sl = price + (1.5 * atr_1h)
        raw_sl = min(struct_sl, atr_sl)
        if (raw_sl - price) < (0.8 * atr_15m):
            raw_sl = price + (1.2 * atr_1h)
        risk = abs(raw_sl - price)
        sl = round(raw_sl, decimals)
        tp1 = round(price - (1.5 * risk), decimals)
        tp2 = round(price - (3.0 * risk), decimals)

    sl_pct = round((abs(price - sl) / price) * 100, 2) if price > 0 else 1.5

    return {
        "coin": coin, "symbol": symbol, "price": price,
        "verdict": verdict, "sl": sl, "tp1": tp1, "tp2": tp2,
        "rsi": round(tf_15m['rsi'], 1), "adx": round(tf_15m['adx'], 1),
        "rvol": round(tf_15m['rvol'], 2), "score": score, "sl_pct": sl_pct
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

# Sidebar: Risk & Filter Settings
with st.sidebar:
    st.header("⚙️ Risk & Filter Settings")
    acc_balance = st.number_input("Account Balance ($)", min_value=10.0, value=100.0, step=10.0)
    risk_pct = st.slider("Risk per Trade (%)", min_value=0.5, max_value=5.0, value=1.5, step=0.5)
    max_signals = st.slider("Max Top Signals per Scan", min_value=1, max_value=5, value=3)
    min_score = st.slider("Minimum Confidence Score (%)", min_value=60, max_value=90, value=70, step=5)
    st.caption("🛡️ **Active Pro Guards:**\n- BTC Market Regime Filter\n- EMA 20/50/200 Alignment\n- Candle Close & RVOL Check\n- 4H Loss Cooldown Lock")

# Main UI Layout
st.title("⚡ Crypto Futures Confluence Scanner V2")
st.caption("Institutional Multi-Timeframe Engine • BTC Regime Guard • Smart Structure Stops")

tab1, tab2 = st.tabs(["🚀 Active Market Signals", "📊 Historical Track Record"])

with tab1:
    if st.button("🔄 Scan Top 20 Now (Pro V2 Engine)", use_container_width=True):
        status_box = st.empty()
        status_box.info("🔍 Checking BTC Market Regime & Connecting to KuCoin Futures...")
        
        btc_regime, btc_rsi = get_btc_regime()
        st.session_state['btc_regime'] = (btc_regime, btc_rsi)
        
        existing = load_signals()
        active_coins = [s['coin'] for s in existing if s.get('status') == "RUNNING"]
        candidates = []
        
        for i, coin in enumerate(DEFAULT_TOP_20):
            status_box.text(f"BTC Regime: {btc_regime} | Scanning [{i+1}/20]: {coin}...")
            if is_in_cooldown(coin, existing):
                continue
            res = scan_single_coin(coin, btc_regime)
            if res and res['verdict'] in ["LONG", "SHORT"] and res['score'] >= min_score:
                candidates.append(res)
            time.sleep(0.1)
            
        # Sort by highest confidence score & ADX, then pick Top N
        candidates.sort(key=lambda x: (x['score'], x['adx']), reverse=True)
        found = candidates[:max_signals]
        
        for res in found:
            coin = res['coin']
            if coin not in active_coins:
                rec = {
                    "id": len(existing) + 1, "coin": coin, "symbol": res['symbol'],
                    "side": res['verdict'], "entry": res['price'], "sl": res['sl'],
                    "tp1": res['tp1'], "tp2": res['tp2'],
                    "score": res['score'],
                    "timestamp_ms": exchange.milliseconds(),
                    "time_str": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "status": "RUNNING", "tp1_hit": False, "pnl_percent": 0.0
                }
                save_signal(rec)
                existing.append(rec)
                active_coins.append(coin)
            
        status_box.empty()
        st.session_state['last_scan'] = found
        st.session_state['scan_completed'] = True

    if 'btc_regime' in st.session_state:
        b_reg, b_rsi = st.session_state['btc_regime']
        reg_color = "#00e676" if b_reg == "STRONG_BULL" else ("#ff5252" if b_reg == "STRONG_BEAR" else "#ffb74d")
        st.markdown(f"**🌐 BTC Market Regime:** <span style='color:{reg_color}; font-weight:bold;'>{b_reg}</span> (BTC 15m RSI: {b_rsi})", unsafe_allow_html=True)

    found = st.session_state.get('last_scan', [])
    if found:
        risk_amount = acc_balance * (risk_pct / 100.0)
        for s in found:
            color = "#00e676" if s['verdict'] == "LONG" else "#ff5252"
            p_str = format_price(s['price'])
            tp1_str = format_price(s['tp1'])
            tp2_str = format_price(s['tp2'])
            sl_str = format_price(s['sl'])
            
            sl_pct = max(s.get('sl_pct', 1.5), 0.2)
            pos_size_usdt = round(risk_amount / (sl_pct / 100.0), 1)
            rec_lev = max(1, min(20, int(100 / (sl_pct * 2.5))))
            
            st.markdown(f"""
            <div style="padding:16px; border-radius:10px; border-left:6px solid {color}; background-color:#1e1e24; margin-bottom:12px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <h3 style="margin:0; color:white;">{s['coin']} — <span style="color:{color};">{s['verdict']}</span> @ ${p_str}</h3>
                    <span style="background-color:#2b2b36; color:#ffd54f; padding:4px 10px; border-radius:6px; font-weight:bold;">
                        🔥 Confidence: {s['score']}%
                    </span>
                </div>
                <p style="margin:8px 0; color:#e0e0e0; font-size:15px;">
                    <b>🎯 TP 1 (1.5R):</b> ${tp1_str} &nbsp;|&nbsp; <b>🚀 TP 2 (3.0R):</b> ${tp2_str} &nbsp;|&nbsp; <b>🛑 Smart SL:</b> ${sl_str} (-{sl_pct}%)
                </p>
                <p style="margin:4px 0; color:#80cbc4; font-size:13.5px;">
                    💰 <b>Risk (${round(risk_amount, 2)}):</b> Position Size: <b>${pos_size_usdt} USDT</b> &nbsp;|&nbsp; Suggested Max Leverage: <b>{rec_lev}x</b>
                </p>
                <small style="color:#9e9e9e;">RSI (15m): {s['rsi']} &nbsp;|&nbsp; ADX (15m): {s['adx']} &nbsp;|&nbsp; Volume (RVOL): {s['rvol']}x</small>
            </div>
            """, unsafe_allow_html=True)
    elif st.session_state.get('scan_completed'):
        st.info("⏸️ No high-probability setups passed all V2 Institutional Filters right now. Capital preservation mode active!")

with tab2:
    if st.button("🔄 Refresh Outcomes & Win Rate"):
        with st.spinner("Replaying candles..."):
            update_history()

    signals = load_signals()
    if signals:
        df = pd.DataFrame(signals)
        if 'score' not in df.columns:
            df['score'] = 75
        closed = df[df['status'].isin(["FULL WIN (TP2)", "PARTIAL WIN (TP1->BE)", "LOSS (SL Hit)"])]
        wins = len(closed[closed['status'].str.contains("WIN")])
        total_closed = len(closed)
        win_rate = round((wins / total_closed * 100), 1) if total_closed > 0 else 0.0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Historical Win Rate", f"{win_rate}%")
        m2.metric("Total Wins", wins)
        m3.metric("Total Losses", total_closed - wins)
        m4.metric("Active (Running)", len(df[df['status'] == "RUNNING"]))

        display_df = df[['id', 'time_str', 'coin', 'side', 'score', 'entry', 'tp1', 'tp2', 'sl', 'status', 'pnl_percent']].copy()
        for col in ['entry', 'tp1', 'tp2', 'sl']:
            display_df[col] = display_df[col].apply(format_price)
        display_df.columns = ['ID', 'Time', 'Coin', 'Type', 'Score %', 'Entry ($)', 'TP1 ($)', 'TP2 ($)', 'SL ($)', 'Status', 'PnL % (1x)']
        st.dataframe(display_df, use_container_width=True)
    else:
        st.write("No recorded signals yet.")
