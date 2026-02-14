# app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from joblib import dump, load
import matplotlib.pyplot as plt
from datetime import timedelta

st.set_page_config(page_title="📈 Edgewalker Stock App", layout="wide")

# -----------------------------
# Sidebar Settings
# -----------------------------
popular_symbols = [
    "AAPL", "MSFT", "GOOG", "AMZN", "TSLA",
    "META", "NVDA", "BRK-B", "JPM", "V",
    "UNH", "HD", "PG", "MA", "DIS", "HPQ", "HPE",
    "NFLX", "PYPL", "ADBE", "KO", "NKE"
]

st.sidebar.header("Stock Settings")
symbol = st.sidebar.selectbox("Select Stock Symbol", popular_symbols, index=2)
forecast_days = st.sidebar.slider("Forecast Days (1–14)", 1, 7)
retrain = st.sidebar.checkbox("Retrain Model", value=False)

MODEL_FILE = "Stock_Predictions_Model.keras"
SCALER_FILE = "Stock_Predictions_Model.scaler"

# -----------------------------
# Load Historical Data
# -----------------------------
@st.cache_data(ttl=3600)
def get_data(sym):
    try:
        df = yf.download(sym, period="2y", auto_adjust=True, progress=False)
        if not df.empty:
            df.reset_index(inplace=True)
            df['Date'] = pd.to_datetime(df['Date'])
            return df
    except:
        pass
    return pd.DataFrame()

df = get_data(symbol)
if df.empty:
    st.error("Could not fetch data. Check your symbol or connection.")
    st.stop()

# -----------------------------
# Load or Train Model
# -----------------------------
@st.cache_resource
def load_or_train_model(df, retrain=False):
    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE) and not retrain:
        model = load_model(MODEL_FILE)
        scaler = load(SCALER_FILE)
        return model, scaler

    series = df['Close'].values.reshape(-1,1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series)
    X, y = [], []
    window = 100
    for i in range(window, len(scaled)):
        X.append(scaled[i-window:i,0])
        y.append(scaled[i,0])
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    model = Sequential([
        LSTM(50, input_shape=(X.shape[1],1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=2, batch_size=16, verbose=0)
    model.save(MODEL_FILE)
    dump(scaler, SCALER_FILE)
    return model, scaler

model, scaler = load_or_train_model(df, retrain=retrain)

# -----------------------------
# Forecast Function
# -----------------------------
def forecast(model, scaler, series, days):
    window = 100
    if len(series) < window:
        raise ValueError(f"Need at least {window} historical points")
    last_window = series[-window:].values.reshape(-1,1)
    last_scaled = scaler.transform(last_window)
    seq = last_scaled.copy()
    preds_scaled = []
    for _ in range(days):
        p = model.predict(seq.reshape(1, window, 1), verbose=0)[0,0]
        preds_scaled.append(p)
        seq = np.roll(seq, -1)
        seq[-1,0] = p
    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1,1)).flatten()
    future_dates = pd.date_range(series.index[-1] + timedelta(days=1), periods=days)
    return preds, future_dates

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview","🔮 Forecast","💡 Trading Signal","⚡ About"])

# -----------------------------
# Tab 1: Real-Time Candlestick + Volume + MAs
# -----------------------------
with tab1:
    st.subheader(f"{symbol} — Last 10 Rows")
    df_display = df.tail(10)[["Date","Open","High","Low","Close","Volume"]].copy()
    df_display["Volume"] = df_display["Volume"].map(lambda x: f"{int(x):,}")
    st.dataframe(df_display, height=300)

# -----------------------------
# Tab 2: Forecast + Moving Averages + Table + ±2σ Confidence
# -----------------------------
with tab2:
    st.subheader(f"{symbol} — Forecast with MA50, MA100, MA200")
    series = df.set_index('Date')['Close']

    try:
        # Forecast
        preds, future_dates = forecast(model, scaler, series, forecast_days)

        # Correctly create forecast DataFrame
        df_forecast = pd.DataFrame({
            'Day': list(range(1, forecast_days + 1)),
            'Forecast': preds
        })

        # Moving averages
        ma50 = series.rolling(50).mean()
        ma100 = series.rolling(100).mean()
        ma200 = series.rolling(200).mean()

        # ±2σ confidence
        recent_100 = series.tail(100)
        returns_std = recent_100.pct_change().dropna().std() * recent_100.iloc[-1]
        upper = preds + 2*returns_std
        lower = preds - 2*returns_std

        # Plot historical + forecast
        plt.figure(figsize=(12,6))
        plt.plot(series.index, series.values, label='Close', color='green')
        plt.plot(ma50.index, ma50.values, label='MA50', color='orange', linestyle='--')
        plt.plot(ma100.index, ma100.values, label='MA100', color='red', linestyle='--')
        plt.plot(ma200.index, ma200.values, label='MA200', color='blue', linestyle='--')
        plt.plot(future_dates, preds, label='Forecast', color='#FF5733', linestyle='-', marker='o')
        plt.fill_between(future_dates, lower, upper, color='orange', alpha=0.2, label='Confidence ±2σ')
        plt.title(f"{symbol} — Close, MA50/100/200 + Forecast")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        st.pyplot(plt.gcf())
        plt.close()

        # Display forecast table
        st.markdown("### Forecast Table")
        st.dataframe(df_forecast.style.format({"Forecast": "${:,.2f}"}), height=300)

    except Exception as e:
        st.error(f"Forecast error: {e}")


# -----------------------------
# Tab 3: Trading Signal
# -----------------------------
with tab3:
    latest = float(series.iloc[-1])
    pred1, _ = forecast(model, scaler, series, 1)
    pred_price = float(pred1[-1])
    threshold = 0.002
    signal = "HOLD 🟡"
    if pred_price > latest*(1+threshold): signal="BUY 🟢"
    elif pred_price < latest*(1-threshold): signal="SELL 🔴"
    st.subheader("Trading Signal")
    st.metric(label=f"Latest Price: ${latest:.2f}", value=signal, delta=f"Predicted: ${pred_price:.2f}")

# -----------------------------
# Tab 4: About
# -----------------------------
with tab4:
    st.markdown("""
    <div style='text-align:center; margin-top:30px;'>
        <h2 style='color:#00B4D8;'>⚡ The Edgewalker — Powered by Mr Smb ⚡</h2>
        <p style='color:#AAA; font-size:18px;'>Where <strong style='color:#FFD700;'>prediction</strong> meets <strong style='color:#FFD700;'>precision</strong>.</p>
        <p style='color:#888;'>Powered by: <strong>SMB Group of Investments Pvt. Ltd</strong></p>
        <p style='color:#777;'>Created by <strong>Mr Smb — The Edgewalker 🜂</strong></p>
        <p style='color:#666;'>Version 3.0 — Edgewalker Pulse Edition</p>
    </div>
    """, unsafe_allow_html=True)
