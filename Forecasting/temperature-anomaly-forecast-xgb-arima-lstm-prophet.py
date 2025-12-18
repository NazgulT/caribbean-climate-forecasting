"""
Time Series Forecasting for Caribbean Region Temperature Anomaly 
Trains LSTM, XGBoost, SARIMA, and Facebook Prophet models on historical temperature anomaly data dated Jan 1980 - {Current Month} 2025
"""

import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from pathlib import Path
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from statsmodels.tsa.statespace.sarimax import SARIMAX


from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics


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
    df = load_caribbean_weather()[['temp_anomaly']]

    return df

def create_lag_features(df, target_col, lags=[1, 2]):
    """Create lagged features for XGBoost training"""
    df = df.copy()
    for lag in lags:
        df[f'lag_{lag}'] = df[target_col].shift(lag)

    # based on acf and pacf results, seasonality seems to be repeating every 3 years,
    # therefore, rolling stats is set to be with window = 36 months

    #Warning: had to change back to window = 3 because the test set was smaller than 48 months
    df['rolling_mean_3'] = df[target_col].rolling(window=3).mean()
    df['rolling_std_3'] = df[target_col].rolling(window=3).std()

    # Trend features for stationarity
    df['temp_diff_1'] = df[target_col].diff(1)
    df['temp_diff_2'] = df[target_col].diff(2)

    return df.dropna()

def prophet_df(df):
    df = df.copy()
    df = df.reset_index()

    ds = df['date']
    y = df['temp_anomaly']

    df = pd.DataFrame({'ds': ds, 'y': y})

    return df
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

def visualize_results(train, test, models, future):
        """Create comparison visualization"""
        print("\nGenerating visualizations...")
        
        fig, axes = plt.subplots(2, 3, figsize=(14, 7))
        
        test_years = test.index.year

        sarima_forecast = models[0]['predictions']
        lstm_forecast = models[1]['predictions']
        xgb_forecast = models[2]['predictions']
        prophet_forecast = models[3]['predictions']
        future_forecast = future['future_forecast']
    
        test_years_lstm = test.index[-len(lstm_forecast):]
        test_years_xgb = test.index[-len(xgb_forecast):]

        ax1 = axes[0,0]
    
        ax1.plot(train, label='Train')
        ax1.plot(test, label='Test', color='green')
        ax1.plot(sarima_forecast, label='SARIMAX pred', color='red')
        ax1.set_title('SARIMAX')
        ax1.legend(fontsize='x-small')
        ax1.tick_params(rotation=60)

        ax2 = axes[0,1]
        ax2.plot(train, label='Train')
        ax2.plot(test, label='Test', color='green')
        ax2.plot(test_years_xgb, xgb_forecast, label='XGBoost pred', color='red') #index should be 12 month less than the forecast,
        ax2.set_title('XGBoost')
        ax2.legend(fontsize='x-small')
        ax2.tick_params(rotation=60)


        ax3 = axes[0,2]
        ax3.plot(train, label='Train')
        ax3.plot(test, label='Test', color='green')
        ax3.plot(test_years_lstm, lstm_forecast, label='LSTM pred', color='red') #index should be 12 month less than the forecast,
        ax3.set_title('LSTM')
        ax3.legend(fontsize='x-small')
        ax3.tick_params(rotation=60)

        ax5 = axes[1,0]
        # Optional: Plot the results
        models[3]['model'].plot(prophet_forecast, ax = ax5)
        
        # Get the date where forecast begins
        forecast_start_date = test.index.min()
        #ax = ax5.gca()  # Get current axes
        # Add red dashed vertical line at the start of the forecast period
        ax5.axvline(x=forecast_start_date, color='red', linestyle='--', linewidth=1.5, label='Prophet Forecast Start')
        ax5.set_title('Prophet')
        ax5.legend(fontsize='x-small')
        ax5.tick_params(rotation=60)

        # Plot 2: Model performance
        ax4 = axes[1, 1]
        model_names = ['SARIMA', 'LSTM', 'XGBoost', 'Prophet']
        maes = [m['mae'] for m in models] 
        
        bars = ax4.bar(model_names, maes, color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'], 
                    alpha=0.8, edgecolor='black', linewidth=1.5)
        ax4.set_ylabel('MAE (mm)', fontsize=12, fontweight='bold')
        ax4.set_title('Model Performance', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add values on bars
        for bar, mae in zip(bars, maes):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{mae:.3f}mm',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

        # Plot Future Forecasts with Prophet
        ax6 = axes[1,2]
        # Optional: Plot the results
        future['model'].plot(future_forecast, ax = ax6)

        # Get the date where forecast begins (first date not in historical data)
        last_historical_date = test.index.max()
        print(f'last_historical_date = {last_historical_date}')
        forecast_start_date = future_forecast[future_forecast['ds'] > last_historical_date]['ds'].min()
        #ax = fig1.gca()  # Get current axes
        # Add red dashed vertical line at the start of the forecast period
        ax6.axvline(x=forecast_start_date, color='red', linestyle='--', linewidth=1.5, label='Forecast Start')
        ax6.set_title('Prophet: 2026 - 2027 Temperature Anomaly Forecast')
        plt.legend(fontsize='x-small')
        
        plt.suptitle('Caribbean Region Temperature Anomaly Forecast')
        plt.xticks(rotation=60)
        plt.show()

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
    train_scaled = scaler.fit_transform(train[['temp_anomaly']])
    test_scaled = scaler.transform(test[['temp_anomaly']])

    #print(f'test_scaled length = {len(test_scaled)}')

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
        'predictions': y_pred_unscaled.flatten(),
        'actuals': y_test_unscaled.flatten(),
        'mae': mae,
        'rmse': rmse,
        'lookback': lookback
    }

def train_xgboost(train, test):
    """Train XGBoost model"""
    print("\n[3/4] Training XGBoost...")

    # Create features
    train_lags = create_lag_features(train.copy(), 'temp_anomaly')
    test_lags = create_lag_features(test.copy(), 'temp_anomaly')

    feature_cols = [c for c in train_lags.columns if c not in ['temp_anomaly']]

    X_train = train_lags[feature_cols]
    y_train = train_lags['temp_anomaly']

    X_test = test_lags[feature_cols]
    y_test = test_lags['temp_anomaly']

    #Train the XGBoost 
    xgb_model = xgb.XGBRegressor(
        n_estimators = 100,
        max_depth = 3,
        learning_rate = 0.0001,
        subsample = 0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    xgb_model.fit(X_train, y_train)

    # Predict
    y_pred = xgb_model.predict(X_test)

    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"  MAE: {mae:.3f}mm | RMSE: {rmse:.3f}mm")
    
    return {
        'model': xgb_model,
        'predictions': y_pred,
        'actuals': y_test.values,
        'mae': mae,
        'rmse': rmse,
        'features': feature_cols
    }

def train_prophet(train, test):
    # Engineer a dataframe suitable for prophet

    X_train = prophet_df(train)
    X_test = prophet_df(test)

    # Instantiate and fit the Prophet model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode='additive',  # Anomalies are deviations; additive works well
        changepoint_prior_scale=0.05  # Moderate flexibility in trend
    )

    model.fit(X_train)
    # Cross-validation for out-of-sample performance
    # horizon: 12 months, initial training: ~10 years (120 months), period: 6 months
    df_cv = cross_validation(model, initial='3650 days', period='180 days', horizon='365 days', parallel="processes")
    df_p = performance_metrics(df_cv)

    mae = df_p['mae'].mean()
    rmse = df_p['rmse'].mean()

    print(f"  MAE: {mae:.3f} mm| RMSE: {rmse:.3f} mm")

    #Predict on test set
    #Make future dataframe for the next 12 months (Jan 2026 to Dec 2026)
    #future = model.make_future_dataframe(periods=TEST_SIZE, freq='MS')
    # Generate forecast
    forecast = model.predict(X_test)

    return {
        'model': model,
        'predictions': forecast,
        'actuals':test.values,
        'mae': mae,
        'rmse': rmse
    }

def forecast_future(data, model, duration=12):
    """Predicting future precipitation"""
    print(f"\nPredicting future {duration} months with Prophet")

    data = prophet_df(data)
    # Instantiate a new model since Prophet can only be fit once
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode='additive',  # Anomalies are deviations; additive works well
        changepoint_prior_scale=0.05  # Moderate flexibility in trend
    )
    #Train on the whole dataset
    model.fit(data)

    #Make future dataframe for the next 12 months (Jan 2026 to Dec 2026)
    future = model.make_future_dataframe(periods=12, freq='MS')

    # Generate forecast
    forecast = model.predict(future)

    # The forecast for the next 12 months
    forecast_next_12 = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(12)
    forecast_next_12 = forecast_next_12.round(2)
    #forecast_next_12['ds'] = forecast_next_12['ds'].dt.strftime('%Y-%m')

    print("Forecasted Caribbean Precipitation for 2026 (mm):")
    print(forecast_next_12[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

    return {
        'model' : model,
        'future_forecast' : forecast_next_12
    }

def main():

    """Main execution"""
    print("=" * 80)
    print("TIME SERIES FORECASTING - PRODUCTION RUN")
    print("=" * 80)

    # Load data
    data = load_and_prepare_data()


    # Split train/test
    train = data[:-TEST_SIZE] # everything until the last 24 months
    test = data[-TEST_SIZE:] # the last 24 months

    print(f"\nTrain: {len(train)} months | Test: {len(test)} months")

    # Train models
    sarima_results = train_sarima(train, test)
    lstm_results = train_lstm(train, test)
    xgb_results = train_xgboost(train, test)
    prophet_results = train_prophet(train, test)

    future = forecast_future(data, prophet_results['model'])

    models = [sarima_results, lstm_results, xgb_results, prophet_results]

    # Visualize
    visualize_results(train, test, models, future)

if __name__ == '__main__':
    results = main()