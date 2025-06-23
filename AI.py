import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info and warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from datetime import timedelta
import yfinance as yf
import plotly.graph_objs as go
import traceback
import logging
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, GRU, Dense, Input, Conv1D, MaxPooling1D, Dropout, Reshape
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import requests

# Configure Streamlit
st.set_page_config(page_title="֎AI Crypto Price Predictor",
                   layout="wide",
                   page_icon="📈")

# Custom CSS styling
st.markdown("""
<style>
    /* Tab styling */
    button[data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] > p {
        font-size: 18px;
        font-weight: bold;
    }
    
    /* Header styling */
    h1 {
        font-size: 27px !important;
    }
    
    h2 {
        font-size: 24px !important;
    }
    
    h3 {
        font-size: 20px !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        padding: 20px;
    }
    
    /* Metric cards */
    [data-testid="metric-container"] {
        border: 1px solid #e1e4e8;
        border-radius: 8px;
        padding: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Clickable image styling */
    .clickable-image {
        cursor: pointer;
        margin-bottom: 10px;
        width: 100%;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    .clickable-image:hover {
        transform: scale(1.01);
    }
</style>
""",
            unsafe_allow_html=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
COINS = {
    "Bitcoin (BTC)": "BTC-USD",
    "Ethereum (ETH)": "ETH-USD",
    "Binance Coin (BNB)": "BNB-USD",
    "Cardano (ADA)": "ADA-USD",
    "Solana (SOL)": "SOL-USD",
    "Ripple (XRP)": "XRP-USD",
    "Dogecoin (DOGE)": "DOGE-USD",
    "Polkadot (DOT)": "DOT-USD",
    "Litecoin (LTC)": "LTC-USD",
    "Avalanche (AVAX)": "AVAX-USD"
}

MODEL_TYPES = {
    "MLP Neural Network": "MLP",
    "GRU Recurrent Network": "GRU",
    "LSTM Recurrent Network": "LSTM",
    "CNN-LSTM Hybrid": "CNN-LSTM"
}

INTERVALS = {"Hourly": "1h", "Daily": "1d", "Weekly": "1wk", "Monthly": "1mo"}

MODEL_DIR = "models"
SCALER_DIR = "scalers"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SCALER_DIR, exist_ok=True)

# --- CSS for ticker ---
st.markdown("""
    <style>
    .ticker-container {
        background-color: #000000;
        color: #ffffff;
        padding: 5px 0;
        border-radius: 8px;
        overflow: hidden;
        white-space: nowrap;
        font-family: monospace;
        font-size: 20px;
        margin-bottom: 20px;
    }
    .ticker-content {
        display: inline-block;
        padding-left: 100%;
        animation: ticker-slide 20s linear infinite;
    }
    @keyframes ticker-slide {
        0% { transform: translateX(0%); }
        100% { transform: translateX(-100%); }
    }
    .coin {
        display: inline-block;
        margin: 0 25px;
        padding: 6px 10px;
        background-color: transparent;
        border-radius: 6px;
    }
    </style>
""",
            unsafe_allow_html=True)

# Ticker display
ticker_placeholder = st.empty()

def fetch_prices():
    result = ""
    for symbol, yf_symbol in COINS.items():
        try:
            data = yf.Ticker(yf_symbol).history(period="1d", interval="1m")
            latest_price = data["Close"].dropna().iloc[-1]
            result += f'<span class="coin">{symbol}: ${latest_price:,.2f}</span>'
        except:
            result += f'<span class="coin">{symbol}: N/A</span>'
    return result

# Only update ticker once per run
ticker_html = f'<div class="ticker-container"><div class="ticker-content">{fetch_prices()}</div></div>'
ticker_placeholder.markdown(ticker_html, unsafe_allow_html=True)

# Helper functions
def handle_error(context, e, show_traceback=False):
    """Handle and display errors gracefully"""
    error_msg = f"⚠️ Error in {context}: {str(e)}"
    if show_traceback:
        error_msg += f"\n\nTraceback:\n{traceback.format_exc()}"
    st.error(error_msg)
    logger.error(error_msg)
    return None

@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_live_price(symbol):
    """Fetch current price using yfinance"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period='1d')
        if not data.empty:
            return data['Close'].iloc[-1]
        return None
    except Exception as e:
        handle_error(f"fetching live price for {symbol}", e)
        return None

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_historical_data(symbol, interval, period="963d"):
    """Fetch historical data using yfinance"""
    try:
        ticker = yf.Ticker(symbol)

        # Adjust period based on interval
        if interval == "1h":
            period = "36d"  # Max 730 days for 1h data
        elif interval == "1d":
            period = "369d"  # ~2 years
        elif interval == "1wk":
            period = "3690d"  # ~10 years
        elif interval == "1mo":
            period = "3690d"  # ~10 years

        data = ticker.history(period=period, interval=interval)

        if data.empty:
            raise ValueError("No data returned from yfinance")

        # Clean and format data
        data = data[['Close']].rename(columns={'Close': 'Price'})
        data.index.name = 'Date'
        data.dropna(inplace=True)

        return data
    except Exception as e:
        handle_error(f"fetching historical data for {symbol}", e)
        return None

def prepare_data(data, window=60, scaler=None):
    """Prepare data for model training/prediction"""
    try:
        if scaler is None:
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled = scaler.fit_transform(data)
        else:
            scaled = scaler.transform(data)

        x, y = [], []
        for i in range(window, len(scaled)):
            x.append(scaled[i - window:i])
            y.append(scaled[i])

        x, y = np.array(x), np.array(y)
        return x, y, scaler
    except Exception as e:
        handle_error("preparing data", e)
        return None, None, None

def build_model(model_type, input_shape, neurons=50, dropout=0.2, learning_rate=0.001):
    """Build the selected model architecture"""
    try:
        model = Sequential()

        if model_type == "LSTM":
            model.add(Input(shape=input_shape))
            model.add(LSTM(neurons, return_sequences=True))
            model.add(Dropout(dropout))
            model.add(LSTM(neurons))
            model.add(Dropout(dropout))
            model.add(Dense(1))

        elif model_type == "GRU":
            model.add(Input(shape=input_shape))
            model.add(GRU(neurons, return_sequences=True))
            model.add(Dropout(dropout))
            model.add(GRU(neurons))
            model.add(Dropout(dropout))
            model.add(Dense(1))

        elif model_type == "MLP":
            model.add(Input(shape=(input_shape[0], )))
            model.add(Dense(neurons * 2, activation='relu'))
            model.add(Dropout(dropout))
            model.add(Dense(neurons, activation='relu'))
            model.add(Dense(1))

        elif model_type == "CNN-LSTM":
            model.add(Input(shape=input_shape))
            model.add(Reshape((input_shape[0], input_shape[1])))
            model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'))
            model.add(MaxPooling1D(pool_size=2))
            model.add(LSTM(neurons, return_sequences=True))
            model.add(Dropout(dropout))
            model.add(LSTM(neurons))
            model.add(Dropout(dropout))
            model.add(Dense(1))

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        model.compile(optimizer=Adam(learning_rate=learning_rate),
                      loss='mean_squared_error',
                      metrics=['mae'])
        return model
    except Exception as e:
        handle_error(f"building {model_type} model", e, show_traceback=True)
        return None

def train_model(coin, model_type, interval, epochs=8, batch_size=32, neurons=50, dropout=0.2, lr=0.001):
    """Train the selected model"""
    try:
        symbol = COINS[coin]
        df = fetch_historical_data(symbol, INTERVALS[interval])

        if df is None or df.empty:
            st.error(f"Failed to fetch data for {coin}")
            return None, None, None, None, None, None

        scaler_path = os.path.join(SCALER_DIR, f"{coin}_{interval}_scaler.save")
        scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

        x, y, scaler = prepare_data(df.values, scaler=scaler)
        if x is None or y is None:
            st.error(f"Failed to prepare data for {coin}")
            return None, None, None, None, None, None

        # Adjust input shape based on model type
        if model_type == "MLP":
            x = x.reshape(x.shape[0], -1)
            input_shape = (x.shape[1], )
        elif model_type == "CNN-LSTM":
            input_shape = (x.shape[1], x.shape[2])
        else:
            input_shape = x.shape[1:]

        model = build_model(model_type, input_shape, neurons, dropout, lr)
        if model is None:
            st.error(f"Failed to build model for {coin}")
            return None, None, None, None, None, None

        with st.spinner(f"Training {model_type} model for {coin}..."):
            history = model.fit(x,
                              y,
                              epochs=epochs,
                              batch_size=batch_size,
                              verbose=0,
                              validation_split=0.1)

        model_path = os.path.join(MODEL_DIR, f"{coin}_{model_type}_{interval}.keras")
        joblib.dump(scaler, scaler_path)
        model.save(model_path)

        return model, df, x, y, model_path, scaler_path, history
    except Exception as e:
        handle_error(f"training {model_type} model for {coin}",
                     e,
                     show_traceback=True)
        return None, None, None, None, None, None, None

def predict_future(model, last_seq, steps_pred, model_type, scaler):
    """Generate future predictions"""
    try:
        predicted = []
        input_seq = last_seq.copy()

        for _ in range(steps_pred):
            if model_type == "MLP":
                pred_scaled = model.predict(input_seq.reshape(1, -1), verbose=0)[0][0]
                input_seq = np.roll(input_seq, -1)
                input_seq[-1] = pred_scaled
            elif model_type == "CNN-LSTM":
                pred_scaled = model.predict(input_seq[np.newaxis, ...], verbose=0)[0][0]
                input_seq = np.roll(input_seq, -1, axis=0)
                input_seq[-1] = pred_scaled
            else:  # LSTM/GRU
                pred_scaled = model.predict(input_seq[np.newaxis, ...], verbose=0)[0][0]
                input_seq = np.vstack([input_seq[1:], [pred_scaled]])

            predicted.append(pred_scaled)

        predicted = np.array(predicted).reshape(-1, 1)
        predicted_prices = scaler.inverse_transform(predicted)
        return predicted_prices.flatten().tolist()
    except Exception as e:
        handle_error(f"predicting future prices with {model_type}",
                     e,
                     show_traceback=True)
        return None

def evaluate_model(model, x, y, model_type):
    """Evaluate model performance"""
    try:
        if model_type == "MLP":
            x = x.reshape(x.shape[0], -1)
        preds = model.predict(x, verbose=0)
        mse = mean_squared_error(y, preds)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y - preds))
        return mse, rmse, mae
    except Exception as e:
        handle_error("evaluating model", e)
        return None, None, None

def create_prediction_plot(historical_df, future_dates, predicted_prices, title):
    """Create interactive plot with historical and predicted data"""
    fig = go.Figure()

    # Historical data
    fig.add_trace(
        go.Scatter(x=historical_df.index,
                   y=historical_df.iloc[:, 0],
                   mode='lines',
                   name='Historical',
                   line=dict(color='#1f77b4', width=2),
                   hovertemplate=
                   'Date: %{x|%Y-%m-%d}<br>Price: $%{y:.4f}<extra></extra>'))

    # Predicted data
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=predicted_prices,
            mode='lines+markers',
            name='Predicted',
            line=dict(color='#ff7f0e', width=2, dash='dot'),
            marker=dict(size=6),
            hovertemplate=
            'Date: %{x|%Y-%m-%d}<br>Predicted: $%{y:.4f}<extra></extra>'))

    # Current price marker
    last_price = historical_df.iloc[-1, 0]
    fig.add_trace(
        go.Scatter(x=[historical_df.index[-1]],
                   y=[last_price],
                   mode='markers',
                   name='Current Price',
                   marker=dict(color='#2ca02c', size=10),
                   hovertemplate=f'Current: ${last_price:.4f}<extra></extra>'))

    fig.update_layout(title=title,
                      xaxis_title="Date",
                      yaxis_title="Price (USD)",
                      height=600,
                      template="plotly_white",
                      hovermode="x unified",
                      legend=dict(orientation="h",
                                  yanchor="bottom",
                                  y=1.02,
                                  xanchor="right",
                                  x=1))

    return fig

# Streamlit UI
def main():
    # Add clickable image at the top
    st.markdown(
        f'<a href="https://four.meme/token/0x0b27a4b8faad66586d67a9bffbbcd61e2e5ff87a?code=5L95FU43AQ3M" target="_blank">'
        f'<img src="https://i.postimg.cc/9fHWGDg8/ik804.jpg" class="clickable-image" alt="Clickable Image">'
        f'</a>',
        unsafe_allow_html=True
    )
    
    st.title("📈 ֎AI CRYPTO PRICE PREDICTION")
    st.caption("Powered by ֎SMA.Deep Learning Models with AI Data")

    # Initialize session state for coin selection
    if 'selected_coin' not in st.session_state:
        st.session_state.selected_coin = "Bitcoin (BTC)"

    # Sidebar with live price and info
    with st.sidebar:
        st.header("⛁ Coin Information")

        selected_coin_for_live = st.selectbox(
            "Select Coin",
            list(COINS.keys()),
            index=list(COINS.keys()).index(st.session_state.selected_coin),
            key="live_coin_select")

        # Live price display
        price_placeholder = st.empty()
        price = fetch_live_price(COINS[selected_coin_for_live])

        if price is not None:
            price_placeholder.metric(
                label=f"Current {selected_coin_for_live.split('(')[0].strip()} Price",
                value=f"${price:,.4f}",
                delta=f"{price - fetch_live_price(COINS[selected_coin_for_live]) or 0:.4f}"
            )
        else:
            price_placeholder.warning("Live price temporarily unavailable")

        st.markdown("---")
        st.markdown("### ℹ️ About This App")
        st.markdown("""
        This program uses deep learning models to predict cryptocurrency prices.

        **Features:**
        - Real-time price ticker for top cryptocurrencies
        - Multiple AI model architectures (MLP, GRU, LSTM, CNN-LSTM)
        - Automatic model training when needed
        - Interactive price prediction visualizations
        - Performance metrics for model evaluation

        **Data Source**: yfinance (Yahoo Finance)
        """)

    # Main tabs
    tab_info, tab_predict = st.tabs(["📝 App Description", "🔮 Price Prediction"])

    with tab_info:
        st.header("📝 How It Works")
        st.markdown("""
        **You can donate after profit earn on this USDT BNB Chain Address 0x1469686F6116aBb4B3e963Eb3B55E1F6233a5e35**
        
        This program uses deep learning models to predict cryptocurrency prices.

        **How it works?**

        The program automatically trains AI models when needed for prediction.

        1. **Price Prediction**
        - Go to the Price Prediction tab
        - Select cryptocurrency, AI model & time period
        - Click "Predict Future Prices" button
        - The system will automatically train a model if needed
        - View your price predictions

        **WOW**.... You are in the **🔮 future** now.

        **Available Models:**
        - **MLP Neural Network**: Basic feedforward neural network
        - **GRU Recurrent Network**: Gated Recurrent Unit for sequence data
        - **LSTM Recurrent Network**: Long Short-Term Memory for time series
        - **CNN-LSTM Hybrid**: Combines convolutional and LSTM layers

        **Time Intervals:**
        - Hourly
        - Daily
        - Weekly
        - Monthly
        """)

    with tab_predict:
        st.header("🔮 Price Prediction")

        col1, col2, col3 = st.columns([3, 2, 2])
        with col1:
            coin_pred = st.selectbox("Select Cryptocurrency",
                                     list(COINS.keys()),
                                     index=list(COINS.keys()).index(
                                         st.session_state.selected_coin),
                                     key="predict_coin_select")
            st.session_state.selected_coin = coin_pred

        with col2:
            model_display_name = st.selectbox("Model Type",
                                              options=list(MODEL_TYPES.keys()),
                                              key="predict_model")
            model_type_code = MODEL_TYPES[model_display_name]

        with col3:
            interval_display_name = st.selectbox("Time Interval",
                                                 options=list(INTERVALS.keys()),
                                                 key="predict_interval")
            interval_code = INTERVALS[interval_display_name]

        col1, col2 = st.columns([2, 1])
        with col1:
            steps_pred = st.slider(
                "Steps to Predict",
                min_value=1,
                max_value=100,
                value=30,
                help="Number of future time periods to predict")

        with col2:
            st.markdown("")
            st.markdown("")
            auto_train = st.checkbox(
                "Auto-train if needed",
                value=True,
                help="Automatically train model if not found")

        def run_prediction():
            """Run the prediction pipeline"""
            try:
                model_path = os.path.join(
                    MODEL_DIR,
                    f"{coin_pred}_{model_type_code}_{interval_code}.keras")
                scaler_path = os.path.join(
                    SCALER_DIR, f"{coin_pred}_{interval_code}_scaler.save")

                # Check if model exists or auto-train is enabled
                if not os.path.exists(model_path) and auto_train:
                    with st.spinner("Model not found. Training new model..."):
                        result = train_model(coin_pred,
                                             model_type_code,
                                             interval_display_name,
                                             epochs=8,
                                             batch_size=32,
                                             neurons=50,
                                             dropout=0.2,
                                             lr=0.001)

                        if result is None:
                            st.error("Failed to train model")
                            return

                        model, df, x, y, model_path, scaler_path, _ = result

                if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                    st.error("Model not found. Please enable auto-train or try again.")
                    return

                # Load model and scaler
                model = load_model(model_path)
                model.compile(optimizer=Adam(learning_rate=0.001),
                              loss='mean_squared_error',
                              metrics=['mae'])
                scaler = joblib.load(scaler_path)

                # Fetch historical data
                df = fetch_historical_data(COINS[coin_pred], interval_code)
                if df is None or df.empty:
                    st.error("Failed to fetch historical data")
                    return

                # Prepare data for prediction
                x, y, _ = prepare_data(df.values, scaler=scaler)
                if x is None or y is None:
                    st.error("Failed to prepare data for prediction")
                    return

                # Generate predictions
                last_seq = x[-1]
                predicted_prices = predict_future(model, last_seq, steps_pred,
                                                  model_type_code, scaler)

                if predicted_prices is None:
                    st.error("Prediction failed")
                    return

                # Generate future dates
                last_date = df.index[-1]
                delta_map = {
                    "1h": timedelta(hours=1),
                    "1d": timedelta(days=1),
                    "1wk": timedelta(weeks=1),
                    "1mo": timedelta(days=30)
                }
                delta = delta_map.get(interval_code, timedelta(days=1))
                future_dates = [
                    last_date + delta * (i + 1) for i in range(steps_pred)
                ]

                # Create results dataframe
                df_future = pd.DataFrame({
                    "Date": future_dates,
                    "Predicted Price": predicted_prices
                }).set_index("Date")

                # Display results
                st.subheader(f"Prediction Results for {coin_pred}")

                # Create plot and data columns
                col1, col2 = st.columns([2, 2])

                with col2:
                    fig = create_prediction_plot(
                        df, future_dates, predicted_prices,
                        f"{coin_pred} Price Prediction ({model_display_name}, {interval_display_name})"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col1:
                    st.markdown("### Prediction Values")
                    st.dataframe(df_future.style.format(
                        {"Predicted Price": "${:,.4f}"}),
                                 height=600)

                # Show performance metrics on historical data
                st.subheader("Model Performance on Historical Data")
                mse, rmse, mae = evaluate_model(model, x, y, model_type_code)
                if mse is not None:
                    col1, col2, col3 = st.columns(3)
                    col1.metric("MSE", f"{mse:.6f}")
                    col2.metric("RMSE", f"{rmse:.6f}")
                    col3.metric("MAE", f"{mae:.6f}")

            except Exception as e:
                handle_error("running prediction", e, show_traceback=True)

        if st.button("🔮 Predict Future Prices", use_container_width=True):
            with st.spinner("Generating predictions..."):
                run_prediction()

if __name__ == "__main__":
    main()
