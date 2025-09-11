import streamlit as st
import pandas as pd
import yfinance as yf
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date

# Load the saved model
loaded_model = joblib.load('volatility_model.joblib')

# Define the prediction function
def predict_volatility(ticker, start_date, end_date):
    try:
        new_data = yf.download(ticker, start=start_date, end=end_date, progress=False)

        if new_data.empty:
            st.error(f"Could not download data for {ticker}. Please check the ticker symbol or date range.")
            return None, None
        
        # Simplify column names and clean data
        new_data.columns = [col[0] for col in new_data.columns]
        new_data.dropna(inplace=True)

        # Engineer the same features
        new_data['Log_Returns'] = np.log(new_data['Close'] / new_data['Close'].shift(1))
        new_data['Volatility_21d'] = new_data['Log_Returns'].rolling(window=21).std() * np.sqrt(252)
        new_data['SMA_50'] = new_data['Close'].rolling(window=50).mean()
        new_data['SMA_200'] = new_data['Close'].rolling(window=200).mean()
        new_data['Momentum_50'] = (new_data['Close'] / new_data['Close'].rolling(window=50).mean()) - 1
        new_data['Momentum_200'] = (new_data['Close'] / new_data['Close'].rolling(window=200).mean()) - 1
        delta = new_data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        RS = gain / loss
        new_data['RSI_14'] = 100 - (100 / (1 + RS))
        new_data['Lag_Volatility_1'] = new_data['Volatility_21d'].shift(1)
        new_data['Lag_Volatility_5'] = new_data['Volatility_21d'].shift(5)
        
        new_data.dropna(inplace=True)

        # Select the most recent row of data for prediction
        latest_data = new_data.iloc[-1].drop(['Close', 'High', 'Low', 'Open', 'Volume', 'Log_Returns', 'Volatility_21d'])
        latest_features = pd.DataFrame(latest_data).transpose()
        
        # Make a prediction
        prediction = loaded_model.predict(latest_features)
        
        return prediction[0], new_data
        
    except Exception as e:
        st.error(f"Error making prediction. Please check the ticker symbol. Error: {e}")
        return None, None

# --- Streamlit App Interface ---
st.title("Stock Volatility Predictor")
st.write("Enter a stock ticker and select a date range to predict its future volatility.")

# User inputs
col1, col2 = st.columns(2)
with col1:
    ticker_input = st.text_input("Enter Ticker Symbol", "SPY").upper()
with col2:
    today = date.today()
    start_date = st.date_input("Start Date", today - pd.Timedelta(days=365))
    end_date = st.date_input("End Date", today)

if st.button("Predict"):
    if ticker_input:
        with st.spinner("Predicting..."):
            predicted_volatility, historical_data = predict_volatility(ticker_input, start_date, end_date)
            
            if predicted_volatility is not None:
                # Display the prediction and a confidence interval
                st.markdown(f"**Predicted Volatility for {ticker_input}:** `{predicted_volatility:.4f}`")
                
                # Calculate a simple confidence interval based on our ~4% error from backtesting
                lower_bound = predicted_volatility * 0.96
                upper_bound = predicted_volatility * 1.04
                st.info(f"We are confident the actual volatility will be in the range of **{lower_bound:.4f} to {upper_bound:.4f}**.")
                
                # Add a simple risk assessment
                if predicted_volatility < 0.15:
                    st.success("This suggests a low volatility/low-risk environment.")
                elif predicted_volatility < 0.30:
                    st.warning("This suggests a moderate volatility/moderate-risk environment.")
                else:
                    st.error("This suggests a high volatility/high-risk environment.")

                st.markdown("---")
                
                # Plot the historical data
                st.subheader(f"Historical Data for {ticker_input}")
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
                
                # Price Plot
                ax1.plot(historical_data['Close'], label='Close Price')
                ax1.set_title(f"Historical Close Price")
                ax1.set_ylabel("Price")
                ax1.grid(True)
                ax1.legend()
                
                # Volatility Plot
                ax2.plot(historical_data['Volatility_21d'], color='orange', label='21-Day Volatility')
                ax2.set_title("Historical Volatility")
                ax2.set_xlabel("Date")
                ax2.set_ylabel("Volatility")
                ax2.grid(True)
                ax2.legend()
                
                st.pyplot(fig)

                st.markdown("---")
                st.subheader("Recent Data Table")
                st.dataframe(historical_data.tail())

                st.markdown("---")
                st.subheader("Model Feature Importance")
                
                # Get and plot feature importances
                feature_importances = loaded_model.feature_importances_
                feature_names = historical_data.drop(columns=['Close', 'High', 'Low', 'Open', 'Volume', 'Log_Returns', 'Volatility_21d']).columns
                importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False)
                
                fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
                sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax_imp)
                ax_imp.set_title('Model Feature Importance')
                ax_imp.set_xlabel('Importance')
                ax_imp.set_ylabel('Feature')
                
                st.pyplot(fig_imp)
    else:
        st.warning("Please enter a ticker symbol.")