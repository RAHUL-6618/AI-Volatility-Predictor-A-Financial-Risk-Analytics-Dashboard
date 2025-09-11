# AI Volatility Predictor: A Financial Risk Analytics Dashboard

This project is a comprehensive **data science solution** that predicts the **future volatility of financial assets**.  
It showcases a **full-stack data science workflow**, from data acquisition and rigorous model building to deployment in an interactive web application.

---

##  Overview
The goal of this project is to provide a reliable forecast of a stock's **short-term volatility**, a key measure of risk for investors and analysts.  
The solution uses **historical market data** to train a machine learning model, which is then served through a **user-friendly web interface**.

###  Core Components
- **Data Engineering** → Sourcing, cleaning, and transforming raw financial data  
- **Machine Learning** → Training a robust regression model to make predictions  
- **Model Validation & Backtesting** → Proving the model's reliability with rigorous evaluation across multiple time periods  
- **Deployment** → Creating an interactive dashboard to make the model accessible to anyone  

---

###  Key Features
- **Real-time Predictions** → Instant volatility forecast for any stock by entering its ticker symbol  
- **Interactive Dashboard** → Built with Streamlit, visualizing historical price movements, volatility trends, and clear risk assessments  
- **Advanced Methodology** → Uses **XGBoost**, hyperparameter tuning, and a rigorous backtesting strategy  
- **Model Transparency** → Displays the most important features influencing the model’s predictions  
- **Walk-Forward Backtesting** → Validates the model over multiple rolling time windows to simulate real-world performance  
 

---


## 📐 Methodology & Mathematical Concepts

The model works by identifying patterns in historical data. Several features were engineered using **financial and mathematical concepts**.

---

### 1. Logarithmic Returns
Daily log returns \( r_t \):

\[
r_t = \ln \left(\frac{P_t}{P_{t-1}}\right)
\]

Where:  
- \( P_t \) = Price on day *t*

---

### 2. Historical Volatility
The target variable is **annualized historical volatility** \( \sigma_P \):

\[
\sigma_P = \sqrt{\frac{1}{n-1} \sum_{i=1}^n (r_i - \bar{r})^2} \times \sqrt{252}
\]

Where:  
- \( n = 21 \) (rolling window in days)  
- \( \bar{r} \) = Mean daily log return  
- 252 ≈ trading days in a year  

---

### 3. Relative Strength Index (RSI)
Momentum oscillator used to measure the speed & change of price movements:

\[
RSI = 100 - \frac{100}{1 + RS}
\]

Where:

\[
RS = \frac{\text{Average Gain}}{\text{Average Loss}}
\]

---

### 4. Model Performance & Rigorous Backtesting

We evaluated the model using **R-squared (\(R^2\))**, which measures how much of the variance in the target variable is predictable from the features.

\[
R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}
\]

Where:  
- \( y_i \) = Actual volatility  
- \( \hat{y}_i \) = Predicted volatility  
- \( \bar{y} \) = Mean actual volatility  

---

#### Backtesting Process
Unlike a single train/test split, **backtesting** simulates how the model would have performed over multiple rolling periods in the past.  
This is represented mathematically as a sequence of train/test splits that "walk forward" in time.

For each split \( k \in \{1, …, K\} \):

- **Train**:  
\[
D^{train}_k = \{(x_t, y_t) \mid t \in [T_0, T_k]\}
\]

- **Test**:  
\[
D^{test}_k = \{(x_t, y_t) \mid t \in [T_k, T'_k]\}
\]

This process yields a series of performance scores.  
Our model achieved a **robust average \( R^2 = 0.9492 \)** from backtesting.

---

---

## 📂 Project Structure

```text
AI-Volatility-Predictor-A-Financial-Risk-Analytics-Dashboard/
│── app.py                  # Streamlit interactive dashboard
│── volatility_model.joblib # Saved ML model
│── README.md               # Project overview
|── Tarin.ipynb
```

---

## How to Run the App

Follow these steps to get the AI Volatility Predictor running locally:
### 1. Clone the repository:

```bash
git clone https://github.com/RAHUL-6618/AI-Volatility-Predictor-A-Financial-Risk-Analytics-Dashboard.git
cd AI-Volatility-Predictor-A-Financial-Risk-Analytics-Dashboard
```
### 2. Install Dependencies
```bash
# Make sure you have Python 3.8+ installed
pip install pandas yfinance joblib numpy scikit-learn xgboost streamlit matplotlib seaborn
```

Note: Ensure volatility_model.joblib is in the same directory as app.py.
### 3. Run the App
```bash
# Launch the Streamlit application
streamlit run app.py
```
