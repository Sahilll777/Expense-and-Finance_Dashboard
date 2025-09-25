# src/expense_forecast.py
import pandas as pd
import os
import joblib
from prophet import Prophet

def generate_forecast(df_clean):
    """
    Generates forecast for all categories using Prophet.
    Saves forecast CSV and returns the forecast DataFrame.
    
    Parameters:
        df_clean (pd.DataFrame): Preprocessed transactions with 'category_pred' and 'amount_abs'
    
    Returns:
        forecast_df (pd.DataFrame): Forecast DataFrame with columns ['category', 'ds', 'yhat']
    """

    forecast_dir = "data/processed/forecast"
    forecast_file = os.path.join(forecast_dir, "forecast_all_categories.csv")

    # Ensure category_pred exists
    if "category_pred" not in df_clean.columns:
        raise ValueError("df_clean must contain 'category_pred' column")

    df_clean['date'] = pd.to_datetime(df_clean['date'])
    categories = df_clean['category_pred'].unique()
    forecast_list = []

    for cat in categories:
        df_cat = df_clean[df_clean['category_pred'] == cat].copy()
        monthly_sum = df_cat.groupby('date')['amount_abs'].sum().reset_index()
        monthly_sum = monthly_sum.rename(columns={'date': 'ds', 'amount_abs': 'y'})

        if len(monthly_sum) < 2:
            # Not enough data to forecast
            next_day = monthly_sum['ds'].max() + pd.Timedelta(days=30)
            forecast_list.append({
                "category": cat,
                "ds": next_day,
                "yhat": monthly_sum['y'].iloc[-1]
            })
            continue

        m = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
        m.add_seasonality(name='monthly', period=30.5, fourier_order=5)  # monthly pattern
        m.fit(monthly_sum)

        future = m.make_future_dataframe(periods=30)  # forecast next 30 days
        forecast = m.predict(future)
        next_day_forecast = forecast.iloc[-1]

        forecast_list.append({
            "category": cat,
            "ds": next_day_forecast['ds'],
            "yhat": max(0, next_day_forecast['yhat'])  # ensure no negative
        })

    forecast_df = pd.DataFrame(forecast_list)

    # Save forecast CSV
    if not os.path.exists(forecast_dir):
        os.makedirs(forecast_dir)
    forecast_df.to_csv(forecast_file, index=False)

    return forecast_df
