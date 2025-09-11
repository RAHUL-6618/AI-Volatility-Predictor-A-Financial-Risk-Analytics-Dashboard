# AI Volatility Predictor: A Financial Risk Analytics Dashboard

This project is a comprehensive data science solution that predicts the **future volatility of financial assets**.  
It showcases a **full-stack data science workflow**, from data acquisition and rigorous model building to deployment in an interactive web application.

---

## 📌 Overview
The goal of this project is to provide a reliable forecast of a stock's **short-term volatility**, a key measure of risk for investors and analysts.  
The solution uses **historical market data** to train a machine learning model, which is then served through a **user-friendly web interface**.

### Core Components
- **Data Engineering**: Sourcing, cleaning, and transforming raw financial data.  
- **Machine Learning**: Training a robust regression model to make predictions.  
- **Model Validation**: Proving the model's reliability with professional-grade evaluation techniques.  
- **Deployment**: Creating an interactive dashboard to make the model accessible to anyone.  

---

## 🚀 Key Features
- **Real-time Predictions**: Get an instant volatility forecast for any stock by simply entering its ticker symbol.  
- **Interactive Dashboard**: Built with Streamlit, the app visualizes historical price movements, volatility trends, and a clear risk assessment.  
- **Advanced Methodology**: The model uses techniques like **XGBoost**, **hyperparameter tuning**, and a **rigorous backtesting strategy**.  
- **Model Transparency**: Displays the most important features influencing the model's predictions.  

---

## 📊 Methodology & Mathematical Concepts

This project's model works by identifying patterns in historical data.  
We engineered several features based on **financial and mathematical concepts**.

### 1. Logarithmic Returns
We calculated the daily log returns \( r_t \):

\[
r_t = \ln \left(\frac{P_t}{P_{t-1}}\right)
\]

Where:
- \( P_t \) = Price on day *t*

---

### 2. Historical Volatility
Our target variable is **annualized historical volatility** \( \sigma_P \):

\[
\sigma_P = \sqrt{\frac{1}{n-1} \sum_{i=1}^n (r_i - \bar{r})^2} \times \sqrt{252}
\]

Where:
- \( n = 21 \) (rolling window in days)  
- \( \bar{r} \) = Mean daily log return  
- 252 ≈ trading days in a year  

---

### 3. Relative Strength Index (RSI)
Momentum oscillator used to measure speed & change of price movements:

\[
RSI = 100 - \frac{100}{1 + RS}
\]

Where:

\[
RS = \frac{\text{Average Gain}}{\text{Average Loss}}
\]

---

### 4. Model Performance
We evaluated the model using **R-squared (\(R^2\))**:

\[
R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}
\]

Our model achieved:  
\[
R^2 = 0.9471
\]  
➡️ Explains **94% of the volatility**.

---

## 📂 Project Structure

