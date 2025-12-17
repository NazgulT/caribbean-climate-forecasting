"""
Time Series Forecasting for Caribbean Region Precipitation 
Trains LSTM, XGBoost, SARIMA, and Facebook Prophet models on historical precipitation data dated Jan 1980 - {Current Month} 2025
"""

import pandas as pd
import numpy as np
import requests
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

# Stats evaluation
#from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
#from statsmodels.tsa.seasonal import seasonal_decompose
#from statsmodels.tsa.stattools import adfuller


# ML libraries
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt


# LSTM - Tensorflow/Keras
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.metrics import RootMeanSquaredError
from keras.optimizers import Adam
from keras.models import load_model
from keras import layers


from utils import load_caribbean_weather

# Configuration variables
TEST_SIZE = 36

def load_and_prepare_data():
    print("\nLoading data...")
    # Load the dataset
    df = load_caribbean_weather()[['precip_log']]

    return df

def create_lag_features(df, target_col, lags=[1, 2]):
    """Create lagged features for SARIMA and XGBoost"""
    df = df.copy()
    for lag in lags:
        df[f'lag_{lag}'] = df[target_col].shift(lag)
    df['rolling_mean_12'] = df[target_col].rolling(window=3).mean()
    df['rolling_std_36'] = df[target_col].rolling(window=3).std()

    return df.dropna()


def create_sequences(df_to_np, lookback):
    """Create lookback sequences for LSTM"""
    #convert the df to numpy
    #df_to_np = df.to_numpy()
    X, y = [],[]

    for i in range(len(df_to_np) - lookback):
        X.append(df_to_np[i:i+lookback])
        y.append(df_to_np[i + lookback])
    #print("in create_sequences X:",X)
    return np.array(X), np.array(y)

def visualize_results(train, test, models):
        """Create comparison visualization"""
        print("\nGenerating visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        test_years = test.index.year

        print('Test years = ', test_years)

        sarima_forecast = models[0]['predictions']
        lstm_forecast = models[1]['predictions']

        ax1 = axes[0,0]
    
        ax1.plot(train, label='Train')
        ax1.plot(test, label='Test', color='green')
        ax1.plot(sarima_forecast, label='SARIMA predictions', color='red')
        ax1.set_title('SARIMA: Caribbean Precipitation Predictions')
        ax1.legend()

        ax2 = axes[0,1]
        ax2.plot(train, label='Train')
        ax2.plot(test, label='Test', color='green')
        ax2.plot(, lstm_forecast, label='LSTM predictions', color='red') #index should be 12 month less than the forecast,
        #remove the beginning 12 indecex from the lstm forecast years
        ax2.set_title('LSTM: Caribbean Precipitation Predictions')
        ax2.legend()

        plt.show()
        
        '''
        # Plot 1: All models comparison
        ax1 = axes[0, 0]
        ax1.plot(data.index, data['precip'], 'o-', 
                label='Actual', linewidth=2, markersize=6)
        
        # Align years for each model
        for name, model in [('SARIMA', models[0])]:
            pred_years = test_years[-12:]
            ax1.plot(pred_years, model['predictions'], 
                    's--', label=name, linewidth=2, markersize=5, alpha=0.7)

        plt.show()
        '''

def train_sarima(train_data, test_data):
    """Train SARIMA model"""
    print("\n[1/4] Training SARIMA...")

    # Parameters
    p, d, q = 0, 1, 0  # Non-seasonal ARIMA parameters
    P, D, Q, m = 0, 1, 0, 12  # Seasonal parameters with yearly seasonality (m=12)
    
    
    # Train SARIMA(0,1,0) with AIC=533 confirmed with auto_arima
    model = SARIMAX(train_data, order=(p, d, q), seasonal_order=(P, D, Q, m), freq='MS')
    fitted = model.fit(disp=False, maxiter=200)
    
    # Forecast
    forecast = fitted.forecast(steps=len(test_data))
    forecast.index = test_data.index
    
    mae = mean_absolute_error(test_data, forecast)
    rmse = np.sqrt(mean_squared_error(test_data, forecast))
    
    print(f"  MAE: {mae:.3f} mm | RMSE: {rmse:.3f} mm")
    
    return {
        'model': fitted,
        'predictions': forecast,
        'actuals': test_data,
        'mae': mae,
        'rmse': rmse
    }

def train_lstm(train, test, lookback=12):
    """Train LSTM model"""
    print("\n[2/4] Training LSTM...")
    
    # Scale data
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train[['precip_log']])
    test_scaled = scaler.transform(test[['precip_log']])

    print(f'test_scaled length = {len(test_scaled)}')

    # Create training and testing sets from scaled data and create sequences of 12 months
    X_train, y_train = create_sequences(train_scaled, lookback)
    X_test, y_test = create_sequences(test_scaled, lookback)

    print("Before reshape")
    print(f"X_train.shape = {X_train.shape}, X_test.shape = {X_test.shape}")

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    print("After reshape")
    print(f"X_train.shape = {X_train.shape}, X_test.shape = {X_test.shape}")

    ## Build the lstm model, create callback
    model = Sequential([
    layers.LSTM(50, activation='relu', input_shape=(lookback, 1)),  # 12 timesteps, 1 feature
    layers.Dense(1)  # Predict the 13th month value
    ])

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse', metrics=[RootMeanSquaredError()])

    # This code saves the best performing model to .keras file
    filepath = 'model3/my_best_model.keras'
    cp = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=1, mode='min', save_best_only = True)

    # Train the model
    print("\nFitting the model...")
    history = model.fit(X_train, y_train, validation_split=0.2, batch_size = lookback, epochs=10, callbacks=[cp])

    # Load the best model
    model = load_model(filepath=filepath)

    # Predict the test set
    y_pred = model.predict(X_test, verbose=0)
    y_pred_unscaled = scaler.inverse_transform(y_pred)
    y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    mae = mean_absolute_error(y_test_unscaled, y_pred_unscaled)
    rmse = np.sqrt(mean_squared_error(y_test_unscaled, y_pred_unscaled))
    
    print(f'y_pred.shape = {y_pred.shape}')
    print(f"  MAE: {mae:.3f} mm | RMSE: {rmse:.3f} mm")
    
    return {
        'model': model,
        'scaler': scaler,
        'predictions': y_pred.flatten(),
        'actuals': y_test.flatten(),
        'mae': mae,
        'rmse': rmse,
        'lookback': lookback
    }

def main():

    """Main execution"""
    print("=" * 80)
    print("TIME SERIES FORECASTING - PRODUCTION RUN")
    print("=" * 80)

    # Load data
    data = load_and_prepare_data()

    print(data.tail(3))

    # Split train/test
    train = data[:-TEST_SIZE] # everything until the last 24 months
    test = data[-TEST_SIZE:] # the last 24 months

    print(f"\nTrain: {len(train)} months | Test: {len(test)} months")

    # Train models
    lstm_results = train_lstm(train, test)
    #xgb_results = train_xgboost(train, test)
    sarima_results = train_sarima(train, test)

    models = [sarima_results, lstm_results]

    # Visualize
    visualize_results(train, test, models)

if __name__ == '__main__':
    results = main()