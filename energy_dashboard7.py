import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error

# Load the dataset
def load_data(uploaded_file):
    data = pd.read_csv(uploaded_file)
    return data

# Preprocess and clean the data
def preprocess_data(data):
    # Automatically detect time column if exists
    time_column = None
    for column in data.columns:
        if 'time' in column.lower() or 'date' in column.lower():
            time_column = column
            break
            
    if time_column:
        data[time_column] = pd.to_datetime(data[time_column], errors='coerce')
        data.set_index(time_column, inplace=True)
        data = data.dropna()  # Drop rows with missing values
    else:
        st.error("No time column found in the dataset.")
        return None
    
    return data

# ARIMA model prediction
def arima_model(data, target_column):
    model = ARIMA(data[target_column], order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=30)
    return forecast

# LSTM model prediction
def lstm_model(data, target_column):
    data = data[[target_column]]
    data = data.values.reshape(-1, 1)
    
    # Prepare the dataset for LSTM (60 time steps as input)
    X = []
    y = []
    for i in range(60, len(data)):
        X.append(data[i-60:i, 0])
        y.append(data[i, 0])
    X, y = np.array(X), np.array(y)
    
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    model.fit(X, y, epochs=10, batch_size=32)
    forecast = model.predict(X[-30:])
    
    return forecast

# Streamlit UI
st.title('Energy Consumption Forecast Dashboard')

# Sidebar: File Upload
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    data = load_data(uploaded_file)
    data = preprocess_data(data)
    
    if data is not None:
        # Automatically detect target column (numeric)
        target_column = None
        for column in data.columns:
            if data[column].dtype in [np.int64, np.float64]:
                target_column = column
                break
                
        if target_column is None:
            st.error("No numeric column found for forecasting.")
        else:
            # Train RandomForest model for feature importance
            features = data.drop(columns=[target_column])
            target = data[target_column]
            
            # Handling categorical columns if any
            label_encoder = LabelEncoder()
            for column in features.select_dtypes(include=['object']).columns:
                features[column] = label_encoder.fit_transform(features[column])
            
            features = pd.get_dummies(features)  # Convert categorical features to dummies
            
            X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
            
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            importances = model.feature_importances_
            feature_names = features.columns
            
            # Display Feature Importances
            fig = px.bar(x=importances, y=feature_names, orientation='h', labels={'x': 'Feature Importance', 'y': 'Features'})
            st.plotly_chart(fig)

            # Display Monthly, Hourly, and Daily Consumption
            st.header("Energy Consumption Overview")
            
            # Filter only numeric columns for resampling
            numeric_data = data.select_dtypes(include=[np.number])

            st.subheader("Monthly Energy Consumption")
            monthly_data = numeric_data.resample('M').mean()  # Only numeric columns
            monthly_fig = px.line(monthly_data, x=monthly_data.index, y=target_column, title="Monthly Energy Consumption")
            st.plotly_chart(monthly_fig)

            st.subheader("Hourly Energy Consumption")
            hourly_data = numeric_data.resample('H').mean()  # Only numeric columns
            hourly_fig = px.line(hourly_data, x=hourly_data.index, y=target_column, title="Hourly Energy Consumption")
            st.plotly_chart(hourly_fig)

            st.subheader("Daily Energy Consumption")
            daily_data = numeric_data.resample('D').mean()  # Only numeric columns
            daily_fig = px.line(daily_data, x=daily_data.index, y=target_column, title="Daily Energy Consumption")
            st.plotly_chart(daily_fig)

            # ARIMA and LSTM Predictions
            st.header("Energy Consumption Forecast")
            
            # ARIMA Prediction
            arima_forecast = arima_model(data, target_column)
            arima_dates = pd.date_range(start=data.index[-1], periods=30, freq='D')
            arima_fig = px.line(x=arima_dates, y=arima_forecast, title="ARIMA Forecasted Energy Consumption")
            st.plotly_chart(arima_fig)

            # LSTM Prediction
            lstm_forecast = lstm_model(data, target_column)
            lstm_dates = pd.date_range(start=data.index[-1], periods=30, freq='D')
            lstm_fig = px.line(x=lstm_dates, y=lstm_forecast.flatten(), title="LSTM Forecasted Energy Consumption")
            st.plotly_chart(lstm_fig)
            
            # Recommendations and Alerts section after predictions
            st.header("Recommendations and Alerts")

            # Create columns for Recommendations and Alerts, both in the same row
            col1, col2 = st.columns([2, 2])

            with col1:
                st.subheader("Recommendations")
                st.write("1. Consider adjusting energy usage during peak hours to reduce costs.")
                st.write("2. Use smart appliances to optimize energy consumption.")
                st.write("3. Monitor energy usage regularly to spot any unusual patterns.")
            
            with col2:
                st.subheader("Alerts")
                st.write("⚠️ Alert: High energy consumption detected, consider reducing appliance usage during peak hours.")
                st.write("⚠️ Alert: Abnormal consumption trend detected, recommend checking for potential issues with appliances.")
                st.write("⚠️ Alert: Energy usage is approaching critical levels, please take immediate action.")
else:
    st.write("Please upload a CSV file to get started.")
