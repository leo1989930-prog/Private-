import os
import json
import re
import time
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import numpy as np
import requests
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

from fugle_marketdata import RestClient

# =========================================================
# ⚙️ 基礎設定與常數
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

TAIPEI_TZ = ZoneInfo("Asia/Taipei")

PRICE_CACHE_TTL = 15
SR_CACHE_TTL = 300
INTRADAY_CACHE_TTL = 15

DEFAULT_STOCK = "6770"

GEMINI_MODELS = [
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro"
]

# =========================================================
# 🖥️ Streamlit 頁面設定 (必須為第一個 Streamlit 指令)
# =========================================================

st.set_page_config(
    page_title="台股閃電智慧決策實戰系統_YL_V4.2",
    layout="wide"
)

# =========================================================
# 🔒 密碼驗證模組（頁面第一道關卡）
# =========================================================

def check_password():
    """檢查使用者密碼驗證狀態"""
    # 從 Secrets 讀取 APP_PASSWORD，若未設定則預設為 123456
    APP_PASSWORD = st.secrets.get("APP_PASSWORD", "123456")

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.title("🔒 系統存取驗證")
        st.markdown("---")
        
        pwd_input = st.text_input("請輸入系統密碼以繼續：", type="password")
        
        if st.button("🔓 登入系統", use_container_width=True):
            if pwd_input == APP_PASSWORD:
                st.session_state.authenticated = True
                st.success("✅ 驗證成功，正在載入系統...")
                st.rerun()
            else:
                st.error("❌ 密碼錯誤，請重新輸入！")
        return False

    return True

# 執行驗證，未通過驗證者直接中斷後續程式碼渲染
if not check_password():
    st.stop()

# =========================================================
# 🎨 RWD UI 樣式設定
# =========================================================

st.markdown("""
<style>
body, p, div, span, label, input {
    font-family: '微軟正黑體', 'Microsoft JhengHei', sans-serif;
}
.stApp {
    background-color: #E5E5E5;
}
[data-testid="stSidebar"] {
    background-color: #1E1E1E !important;
    border-right: 6px solid #FFD700 !important;
}
[data-testid="stSidebar"],
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] label {
    color: #FFFFFF !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #FFFFFF !important;
    border-left: 6px solid #FFD700;
    padding-left: 8px;
}
h1, h2, h3 {
    color: #000000 !important;
    font-weight: 900 !important;
    letter-spacing: 1px;
    border-left: 8px solid #FFD700;
    padding-left: 10px;
}

@media (max-width: 768px) {
    h1 { font-size: 1.4rem !important; border-left-width: 5px !important; }
    h2 { font-size: 1.2rem !important; }
    h3 { font-size: 1.05rem !important; }
    .action-card { padding: 12px !important; }
}

hr {
    border: 0;
    height: 3px;
    background-color: #333333;
    margin: 1.2em 0;
}
.highlight-tag {
    background-color: #FFD700;
    color: #000;
    padding: 3px 8px;
    border-radius: 6px;
    font-weight: 900;
    box-shadow: 2px 2px 0px rgba(0,0,0,0.7);
}
.stButton > button {
    background-color: #000000 !important;
    border: 2px solid #000000 !important;
    color: #FFFFFF !important;
    border-radius: 10px !important;
    font-weight: 900 !important;
    letter-spacing: 1px;
}
.stButton > button:hover {
    background-color: #FFD700 !important;
    color: #000000 !important;
    transform: translateY(-2px);
    box-shadow: 4px 4px 0px rgba(0,0,0,0.8) !important;
}
.action-card {
    background-color: #1E1E1E;
    border-radius: 10px;
    padding: 16px;
    margin-top: 15px;
    color: #FFFFFF;
    border-left: 8px solid #FFD700;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# 📦 Secrets 讀取與套件檢測
# =========================================================

try:
    import twstock
    HAS_TWSTOCK = True
except ImportError:
    HAS_TWSTOCK = False

FUGLE_API_KEY = st.secrets.get("FUGLE_API_KEY", "").strip()
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "").strip()
TG_BOT_TOKEN = st.secrets.get("TG_BOT_TOKEN", "").strip()
TG_CHAT_ID = st.secrets.get("TG_CHAT_ID", "").strip()
GEMINI_AVAILABLE = bool(GEMINI_API_KEY)

# =========================================================
# 🧠 Session State 管理
# =========================================================

if "strategy_result" not in st.session_state:
    st.session_state.strategy_result = None

if "last_stock_code" not in st.session_state:
    st.session_state.last_stock_code = None

if "gemini_model_used" not in st.session_state:
    st.session_state.gemini_model_used = None

if "search_history" not in st.session_state:
    st.session_state.search_history = []

if "strategy_logs" not in st.session_state:
    st.session_state.strategy_logs = []

# =========================================================
# 🕒 工具函數與外部服務
# =========================================================

def get_tw_now():
    return datetime.now(TAIPEI_TZ)

def get_market_session():
    now = get_tw_now()
    if now.weekday() >= 5:
        return "休市"
    hour, minute = now.hour, now.minute
    if hour < 9:
        return "盤前"
    if hour < 13 or (hour == 13 and minute <= 30):
        return "盤中"
    return "盤後"

def send_telegram_notify(token, chat_id, message):
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}
        resp = requests.post(url, json=payload, timeout=5)
        return resp.status_code == 200
    except Exception as e:
        logging.error(f"Telegram 傳送失敗: {e}")
        return False

# =========================================================
# 📜 數據清理與抓取
# =========================================================

def update_history(stock_label):
    history = st.session_state.search_history
    if stock_label in history:
        history.remove(stock_label)
    history.insert(0, stock_label)
    st.session_state.search_history = history[:10]

def parse_stock_code(user_input):
    if not user_input:
        return None
    match = re.search(r"(\d{4,6})", user_input.strip())
    return match.group(1) if match else None

def normalize_yf_columns(df):
    if df is None or df.empty:
        return df
    if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(0)
    return df

def get_yf_symbols(symbol):
    if HAS_TWSTOCK and symbol in twstock.codes:
        market = getattr(twstock.codes[symbol], "market", "")
        if "上櫃" in market:
            return [f"{symbol}.TWO"]
        elif "上市" in market:
            return [f"{symbol}.TW"]
    return [f"{symbol}.TW", f"{symbol}.TWO"]

@st.cache_data(ttl=86400)
def get_stock_name(symbol):
    if not symbol: return "未知標的"
    if HAS_TWSTOCK:
        try:
            if symbol in twstock.codes:
                return twstock.codes[symbol].name
        except Exception:
            pass
    for yf_symbol in get_yf_symbols(symbol):
        try:
            ticker = yf.Ticker(yf_symbol)
            info = ticker.info
            stock_name = info.get("shortName") or info.get("longName")
            if stock_name: return stock_name
        except Exception:
            continue
    return "未知標的"

@st.cache_data(ttl=PRICE_CACHE_TTL)
def get_realtime_price(symbol):
    if FUGLE_API_KEY:
        try:
            client = RestClient(api_key=FUGLE_API_KEY)
            quote = client.stock.intraday.quote(symbol=symbol)
            last_price = quote.get("lastPrice")
            if last_price is not None and float(last_price) > 0:
                return float(last_price), "Fugle 即時報價"
        except Exception as e:
            logging.warning(f"Fugle 報價取得失敗: {e}")
    for yf_symbol in get_yf_symbols(symbol):
        try:
            df = yf.download(yf_symbol, period="1d", interval="1m", progress=False)
            if not df.empty:
                df = normalize_yf_columns(df)
                price = float(df["Close"].iloc[-1])
                return price, "YFinance 1m 備援"
        except Exception:
            continue
    return None, "無法取得報價"

@st.cache_data(ttl=INTRADAY_CACHE_TTL)
def get_intraday_orb_and_df(symbol):
    orb_data = {"today_high": None, "today_low": None, "orb_high": None, "orb_low": None, "df": pd.DataFrame()}
    for yf_symbol in get_yf_symbols(symbol):
        try:
            df = yf.download(yf_symbol, period="1d", interval="1m", progress=False)
            if df.empty: continue
            df = normalize_yf_columns(df)
            
            if df.index.tz is not None:
                df.index = df.index.tz_convert(TAIPEI_TZ)
            else:
                df.index = df.index.tz_localize("UTC").tz_convert(TAIPEI_TZ)

            orb_data["today_high"] = round(float(df["High"].max()), 2)
            orb_data["today_low"] = round(float(df["Low"].min()), 2)
            
            df_orb = df.between_time("09:00", "09:05")
            if not df_orb.empty:
                orb_data["orb_high"] = round(float(df_orb["High"].max()), 2)
                orb_data["orb_low"] = round(float(df_orb["Low"].min()), 2)
                
            if "Volume" in df.columns and "Close" in df.columns:
                cum_vol = df["Volume"].cumsum()
                cum_vol_price = (df["Close"] * df["Volume"]).cumsum()
                df["VWAP"] = np.where(cum_vol > 0, cum_vol_price / cum_vol, df["Close"])
            
            orb_data["df"] = df
            return orb_data
        except Exception as e:
            logging.error(f"即時 K 線計算異常: {e}")
            continue
    return orb_data

@st.cache_data(ttl=SR_CACHE_TTL)
def get_market_levels(symbol):
    result = {"week_high": None, "week_low": None, "month_high": None, "month_low": None}
    for yf_symbol in get_yf_symbols(symbol):
        try:
            df = yf.download(yf_symbol, period="6mo", interval="1d", progress=False)
            if df.empty: continue
            df = normalize_yf_columns(df)
            week_df = df.tail(5)
            result["week_high"] = round(float(week_df["High"].max()), 2)
            result["week_low"] = round(float(week_df["Low"].min()), 2)
            month_df = df.tail(20)
            result["month_high"] = round(float(month_df["High"].max()), 2)
            result["month_low"] = round(float(month_df["Low"].min()), 2)
            return result
        except Exception:
            continue
    return result

# =========================================================
# 📈 繪圖與量化核心
# =========================================================

def plot_intraday_chart(df, symbol, stock_name, orb_high, orb_low):
    if df.empty:
        st.info("尚無盤中 1m 即時圖表資料。")
        return
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name="1m K線", increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
    ))
    if "VWAP" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['VWAP'], mode='lines', name='VWAP 均價',
            line=dict(color='#FFA726', width=1.5)
        ))
    if orb_high:
        fig.add_hline(y=orb_high, line_dash="dash", line_color="#FFD700", annotation_text=f"ORB High ({orb_high})", annotation_position="top right")
    if orb_low:
        fig.add_hline(y=orb_low, line_dash="dash", line_color="#AB47BC", annotation_text=f"ORB Low ({orb_low})", annotation_position="bottom right")
    fig.update_layout(
        title=f"📈 {stock_name} ({symbol}) 1m 盤中即時走勢與 ORB 攻防線",
        yaxis_title="價格 (元)", xaxis_title="時間", template="plotly_dark",
        height=420, margin=dict(l=15, r=15, t=35, b=15), xaxis_rangeslider_visible=False
    )
    st.plotly_chart(fig, use_container_width=True)

def calculate_quant_score(price, night_chg, foreign_oi, day_high, day_low, week_high, week_low, orb_high, orb_low):
    score = 0
    breakdown_items = []
    if night_chg > 0:
        score += 1; breakdown_items.append("夜盤上漲 (+1)")
    elif night_chg < 0:
        score -= 1; breakdown_items.append("夜盤下跌 (-1)")
    if foreign_oi > 0:
        score += 1; breakdown_items.append("外資淨多單 (+1)")
    elif foreign_oi < 0:
        score -= 1; breakdown_items.append("外資淨空單 (-1)")
    if day_high and price >= day_high:
        score += 2; breakdown_items.append("現價突破當日高點 (+2)")
    elif day_low and price <= day_low:
        score -= 2; breakdown_items.append("現價跌破當日低點 (-2)")
    if week_high and price >= week_high:
        score += 3; breakdown_items.append("現價突破 5 日高點 (+3)")
    elif week_low and price <= week_low:
        score -= 3; breakdown_items.append("現價跌破 5 日低點 (-3)")

    orb_status = "Range (區間震盪)"
    if orb_high and price > orb_high:
        score += 2; orb_status = "Breakout (多頭突破)"; breakdown_items.append("突破 ORB 高點 (+2)")
    elif orb_low and price < orb_low:
        score -= 2; orb_status = "Breakdown (空頭跌破)"; breakdown_items.append("跌破 ORB 低點 (-2)")

    if score >= 4: rating = "🟢 強勢偏多"
    elif 1 <= score <= 3: rating = "🟡 偏多觀察"
    elif score == 0: rating = "⚪ 中性震盪"
    elif -3 <= score <= -1: rating = "🟠 偏空觀察"
    else: rating = "🔴 強勢偏空"

    return {"score": score, "rating": rating, "orb_status": orb_status, "breakdown": breakdown_items}

def calculate_position_size(total_capital, max_risk_pct, entry_price, stop_loss_price, target_price):
    if entry_price <= 0 or stop_loss_price <= 0 or entry_price == stop_loss_price: return None
    max_risk_amount = total_capital * (max_risk_pct / 100.0)
    risk_per_share = abs(entry_price - stop_loss_price)
    risk_per_lot = risk_per_share * 1000
    if risk_per_lot <= 0: return None
    suggested_lots = int(max_risk_amount // risk_per_lot)
    reward_per_share = abs(target_price - entry_price) if target_price > 0 else 0
    rr_ratio = round(reward_per_share / risk_per_share, 2) if risk_per_share > 0 else 0
    return {"max_risk_amount": round(max_risk_amount, 0), "risk_per_lot": round(risk_per_lot, 0), "suggested_lots": suggested_lots, "rr_ratio": rr_ratio}

def call_gemini_safe(prompt_text):
    if not GEMINI_AVAILABLE: return None
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt_text}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 1000, "responseMimeType": "application/json"}
    }
    for model_name in GEMINI_MODELS:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={GEMINI_API_KEY}"
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=8)
            if response.status_code == 200:
                data = response.json()
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        raw_json_str = parts[0].get("text", "").strip()
                        try:
                            parsed_data = json.loads(raw_json_str)
                            st.session_state.gemini_model_used = model_name
                            return parsed_data
                        except json.JSONDecodeError:
                            pass
            elif response.status_code in [429, 503]:
                time.sleep(1)
        except Exception:
            continue
    return None

def generate_strategy(stock_name, symbol, price, night_chg, foreign_oi, levels, intraday, strategy_type, time_context, pos_calc):
    day_high, day_low = intraday.get("today_high"), intraday.get("today_low")
    orb_high, orb_low = intraday.get("orb_high"), intraday.get("orb_low")
    week_high, week_low = levels.get("week_high"), levels.get("week_low")

    quant = calculate_quant_score(price, night_chg, foreign_oi, day_high, day_low, week_high, week_low, orb_high, orb_low)
    score_breakdown_str = "\n".join([f"- {item}" for item in quant['breakdown']]) if quant['breakdown'] else "- 無明確加減分項目"
    
    pos_info_str = ""
    if pos_calc:
        pos_info_str = f"""
---
## 🛡️ 風控與建議下單部位
- **單筆最大授權虧損額：** ${pos_calc['max_risk_amount']:,.0f} 元
- **每張承受風險額：** ${pos_calc['risk_per_lot']:,.0f} 元
- **建議最大下單數量：** **{pos_calc['suggested_lots']}** 張 {'(⚠️ 風險超出授權上限，不建議進場)' if pos_calc['suggested_lots'] == 0 else ''}
- **風報比 (R/R Ratio)：** **{pos_calc['rr_ratio']}** {'(🟢 風報比優良)' if pos_calc['rr_ratio'] >= 1.5 else '(⚠️ 風報比偏低，注意控制入場)'}
"""

    rule_report = f"""
# ⚡ {strategy_type} 戰術推演（{time_context}）

## 🎯 綜合量化評分判讀
- **當前評級：** {quant['rating']} (總分：**{quant['score']}** 分)
- **ORB 戰術狀態：** `{quant['orb_status']}`
- **得分明細項目：**
{score_breakdown_str}
{pos_info_str}
---
## 📊 數據結構
- **參考標的：** {stock_name} ({symbol})
- **現價：** {price} 元
- **開盤 ORB 區間 (09:00-09:05)：** High: **{orb_high or 'N/A'}** / Low: **{orb_low or 'N/A'}**
- **當日高低價：** High: **{day_high or 'N/A'}** / Low: **{day_low or 'N/A'}**
- **近 5 日高低價：** High: **{week_high or 'N/A'}** / Low: **{week_low or 'N/A'}**
"""

    prompt = f"""
你是一位冷酷精準的台股交易專家。請務必輸出 JSON 格式，格式如下：
{{
  "action_signal": "強烈多頭 / 偏多佈局 / 中性觀望 / 偏空反彈賣 / 強烈空頭",
  "tactical_actions": ["攻防動作 1 (包含具體價位與條件)", "攻防動作 2", "攻防動作 3"],
  "risk_warning": "停損警示與勝率提示說明"
}}
【量化數據輸入】
- 股票：{stock_name} ({symbol})，現價：{price}
- 策略類型：{strategy_type} ({time_context})
- 量化總分：{quant['score']} 分 ({quant['rating']})
- ORB 狀態：{quant['orb_status']} (High: {orb_high}, Low: {orb_low})
- 建議張數：{pos_calc['suggested_lots'] if pos_calc else 'N/A'} 張，風報比：{pos_calc['rr_ratio'] if pos_calc else 'N/A'}
"""
    ai_json = call_gemini_safe(prompt)
    return rule_report, ai_json, quant

# =========================================================
# 🚀 UI 側邊欄設定
# =========================================================

st.title("⚡ 台股閃電智慧決策實戰系統_YL_V4.2 (TG推播版)")

st.sidebar.header("📊 參數與戰術設定")
if st.sidebar.button("🔄 離線/刷新報價", use_container_width=True):
    st.cache_data.clear()
    st.sidebar.success("快取已清空，數據即時刷新中！")

input_val = st.sidebar.text_input("輸入股票代號", value=DEFAULT_STOCK)
quick_select = st.sidebar.selectbox("📜 快速選擇歷史紀錄", ["-- 請選擇 --"] + st.session_state.search_history)
user_input = quick_select if quick_select != "-- 請選擇 --" else input_val
symbol = parse_stock_code(user_input)

if not symbol:
    st.error("⚠️ 請輸入正確的台股股票代號")
    st.stop()

if st.session_state.last_stock_code != symbol:
    st.session_state.strategy_result = None
    st.session_state.last_stock_code = symbol

stock_name = get_stock_name(symbol)
current_price, price_source = get_realtime_price(symbol)
quote_time = get_tw_now().strftime("%Y-%m-%d %H:%M:%S")

intraday_data = get_intraday_orb_and_df(symbol)
market_levels = get_market_levels(symbol)

default_price = round(current_price, 2) if current_price else 0.0
current_price_input = st.sidebar.number_input("即時股價 (元)", value=default_price, step=0.05)
night_change = st.sidebar.number_input("夜盤漲跌點數", value=0, step=1)
foreign_oi = st.sidebar.number_input("外資未平倉淨口數", value=0, step=100)

st.sidebar.markdown("---")
st.sidebar.header("🛡️ 風控與部位計算器")
total_capital = st.sidebar.number_input("交易總資金 (NTD)", value=1000000, step=100000)
max_risk_pct = st.sidebar.slider("單筆最大承受損失 (%)", min_value=0.5, max_value=5.0, value=1.0, step=0.5)

default_sl = intraday_data.get('orb_low') or market_levels.get('week_low') or (current_price_input * 0.98)
default_tp = intraday_data.get('orb_high') or market_levels.get('week_high') or (current_price_input * 1.03)

stop_loss_input = st.sidebar.number_input("預設停損價 (元)", value=float(round(default_sl, 2) if default_sl else 0.0), step=0.1)
target_price_input = st.sidebar.number_input("預設目標價 (元)", value=float(round(default_tp, 2) if default_tp else 0.0), step=0.1)

pos_calc = calculate_position_size(total_capital, max_risk_pct, current_price_input, stop_loss_input, target_price_input)
if pos_calc and pos_calc['suggested_lots'] == 0:
    st.sidebar.error("⚠️ 風險過高：單筆潛在虧損超過授權上限，建議縮小停損距離或放棄交易！")

st.sidebar.markdown("---")
st.sidebar.header("📲 外部整合 (Telegram)")
user_tg_token = st.sidebar.text_input("TG Bot Token (選填)", value=TG_BOT_TOKEN, type="password")
user_tg_chat_id = st.sidebar.text_input("TG Chat ID (選填)", value=TG_CHAT_ID, type="password")
enable_tg_push = st.sidebar.checkbox("✅ 產出戰術後自動推播至 Telegram", value=bool(TG_BOT_TOKEN and TG_CHAT_ID))

# =========================================================
# 🚀 雙分頁系統
# =========================================================

tab_main, tab_history = st.tabs(["⚡ 即時決策中心", "📚 戰術歷史日誌 (回測區)"])

with tab_main:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"💡 {stock_name} ({symbol})")
        st.markdown(f"**即時報價：** <span class='highlight-tag'>{current_price_input} 元</span>", unsafe_allow_html=True)
        st.markdown(f"**報價來源：** {price_source}")
        st.caption(f"最後更新時間：{quote_time}")
    with col2:
        st.subheader("⚖️ 即時與多週期價位看板")
        st.markdown(f"""
        - **09:00-09:05 ORB 高/低：** `{intraday_data.get('orb_high', 'N/A')}` / `{intraday_data.get('orb_low', 'N/A')}`
        - **盤中即時高/低 (1m)：** `{intraday_data.get('today_high', 'N/A')}` / `{intraday_data.get('today_low', 'N/A')}`
        - **近 5 日高/低 (日K)：** `{market_levels.get('week_high', 'N/A')}` / `{market_levels.get('week_low', 'N/A')}`
        """)

    plot_intraday_chart(
        df=intraday_data.get("df", pd.DataFrame()), symbol=symbol, stock_name=stock_name,
        orb_high=intraday_data.get("orb_high"), orb_low=intraday_data.get("orb_low")
    )
    st.markdown("---")

    st.subheader("🤖 Rule Engine / AI 聯合戰術推演")
    market_session = get_market_session()
    st.info(f"系統台北時間：{get_tw_now().strftime('%Y-%m-%d %H:%M:%S')} ｜ 市場狀態：{market_session}")

    c1, c2, c3 = st.columns(3)
    btn_day_pre = c1.button("📅 當沖盤前", use_container_width=True)
    btn_day_in = c2.button("⚡ 當沖盤中", use_container_width=True)
    btn_day_post = c3.button("📊 當沖盤後", use_container_width=True)
    
    c4, c5, c6 = st.columns(3)
    btn_swing = c4.button("📈 波段戰術", use_container_width=True)
    btn_week = c5.button("📅 週線規劃", use_container_width=True)
    btn_month = c6.button("📆 月線佈局", use_container_width=True)

    active_strategy, time_context = None, None
    if btn_day_pre: active_strategy, time_context = "當沖", "盤前"
    elif btn_day_in: active_strategy, time_context = "當沖", "盤中"
    elif btn_day_post: active_strategy, time_context = "當沖", "盤後"
    elif btn_swing: active_strategy, time_context = "波段", "日線"
    elif btn_week: active_strategy, time_context = "每週", "週線"
    elif btn_month: active_strategy, time_context = "每月", "月線"

    if active_strategy:
        update_history(f"{stock_name} ({symbol})")
        with st.spinner("⚡ 運算量化模型與生成戰術中..."):
            rule_report, ai_json, quant_data = generate_strategy(
                stock_name, symbol, current_price_input, night_change, foreign_oi,
                market_levels, intraday_data, active_strategy, time_context, pos_calc
            )
            st.session_state.strategy_result = (rule_report, ai_json)

            log_time = get_tw_now().strftime("%Y-%m-%d %H:%M:%S")
            ai_signal = ai_json.get("action_signal", "中性觀望") if ai_json else "中性觀望"
            st.session_state.strategy_logs.insert(0, {
                "時間": log_time,
                "標的": f"{stock_name}({symbol})",
                "策略類型": active_strategy,
                "量化評分": f"{quant_data['score']} ({quant_data['rating']})",
                "AI 核心訊號": ai_signal
            })

            if enable_tg_push and user_tg_token and user_tg_chat_id:
                msg = f"""
<b>⚡【閃電戰術推播】</b>
📈 <b>標的:</b> {stock_name} ({symbol})
💰 <b>現價:</b> {current_price_input}
🎯 <b>策略:</b> {active_strategy}
📊 <b>量化評分:</b> {quant_data['score']} 分
🤖 <b>核心訊號:</b> {ai_signal}
                """
                if send_telegram_notify(user_tg_token, user_tg_chat_id, msg.strip()):
                    st.toast("✅ 戰術已成功推播至 Telegram！", icon="📲")
                else:
                    st.toast("⚠️ Telegram 推播失敗，請檢查 Token 與 Chat ID", icon="❌")

    if st.session_state.strategy_result:
        rule_report, ai_json = st.session_state.strategy_result
        st.success("✅ 戰術推演完成")
        st.markdown(rule_report)
        st.markdown("---")
        st.markdown(f"### 🤖 AI 閃電戰術卡片 (`{st.session_state.gemini_model_used or '離線/Rule Engine 模式'}`)")
        
        if ai_json:
            signal = ai_json.get("action_signal", "中性觀望")
            actions = ai_json.get("tactical_actions", [])
            risk_warn = ai_json.get("risk_warning", "")
            st.markdown(f"""
            <div class="action-card">
                <h3 style="color: #FFD700; margin-top:0;">🎯 AI 核心戰術訊號：{signal}</h3>
                <h4>⚡ 核心攻防動作：</h4>
                <ul>{''.join([f'<li>{action}</li>' for action in actions])}</ul>
                <p style="color: #FF8C00; font-weight: bold; margin-bottom: 0;">⚠️ 風控警示：{risk_warn}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("AI 戰術解讀模組處於離線狀態或未設定 GEMINI_API_KEY，已自動呈現 Rule Engine 數據。")

with tab_history:
    st.subheader("📚 戰術歷史日誌 (Session)")
    st.markdown("在此分頁可以回顧您本次開啟網頁以來的所有戰術評分紀錄，方便進行盤後覆盤檢討。")
    if len(st.session_state.strategy_logs) > 0:
        log_df = pd.DataFrame(st.session_state.strategy_logs)
        st.dataframe(log_df, use_container_width=True, hide_index=True)
    else:
        st.info("目前尚無戰術紀錄。請先於「即時決策中心」執行一次戰術推演。")
