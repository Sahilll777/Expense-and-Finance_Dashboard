# app.py
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os
import glob

from src.data_preprocess import preprocess_data
from src.category_model import train_model
from src.expense_forecast import generate_forecast  # new helper function

st.set_page_config(page_title="AI Finance Dashboard", layout="wide")
st.title("Personal Finance & Expense Predictor")

# ------------------ Train category model at startup ------------------
st.info("🔄 Training category model... please wait.")
model = train_model()  # trains on transactions_mapped.csv by default
st.success("✅ Model trained successfully!")

# ------------------ Dual Input: Realtime or Manual Upload ------------------
df = None
mode = None

# Check for realtime CSVs
realtime_files = glob.glob('data/realtime/*.csv')
if realtime_files:
    latest_file = max(realtime_files, key=os.path.getctime)
    df = pd.read_csv(latest_file)
    mode = f"Realtime CSV: {os.path.basename(latest_file)}"

# Manual upload option
uploaded_file = st.file_uploader("Or upload your transactions CSV manually:", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    mode = f"Manual Upload: {uploaded_file.name}"

if df is None:
    st.info("📂 No transactions yet. Place a CSV in 'data/realtime/' folder or upload manually to get started.")
else:
    st.markdown(f"**Current Input Mode:** {mode}")

    st.subheader("📊 Raw Transactions")
    st.dataframe(df.head())

    # Preprocess
    df_clean = preprocess_data(df)
    st.subheader("🧹 Cleaned Transactions")
    st.dataframe(df_clean.head())

    # Predict categories
    preds = model.predict(df_clean['desc_clean'])
    df_clean['category_pred'] = preds
    st.subheader("🤖 Transactions with Predicted Categories")
    st.dataframe(df_clean[['date','amount','description','category_pred']].head())

    # ------------------ Spending Charts ------------------
    st.subheader("📈 Total Spending by Category")
    monthly_summary = df_clean.groupby('category_pred')['amount_abs'].sum().reset_index()
    fig = px.bar(
        monthly_summary, x='category_pred', y='amount_abs',
        labels={'category_pred':'Category','amount_abs':'Total Amount'},
        color='category_pred', text='amount_abs',
        hover_data={'amount_abs':':,.2f'}
    )
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(yaxis_title='Total Spending', xaxis_title='Category', uniformtext_minsize=8, uniformtext_mode='hide')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Monthly Spending per Category Over Time")
    df_clean['month'] = pd.to_datetime(df_clean['date']).dt.to_period('M')
    monthly_time = df_clean.groupby(['month','category_pred'])['amount_abs'].sum().reset_index()
    monthly_time['month'] = monthly_time['month'].astype(str)
    fig2 = px.line(
        monthly_time, x='month', y='amount_abs', color='category_pred',
        markers=True, labels={'month':'Month','amount_abs':'Total Amount','category_pred':'Category'},
        hover_data={'amount_abs':':,.2f'}
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ------------------ Forecast Generation ------------------
    st.info("🔮 Generating forecast for next month...")
    forecast_df = generate_forecast(df_clean)  # automatically creates forecast_all_categories.csv
    st.success("✅ Forecast generated successfully!")

    # ------------------ Budget Alerts ------------------
    st.subheader("⚠️ Category-wise Budget Alerts")
    budget = {
        'Food': 5000,
        'Shopping': 7000,
        'Transport': 3000,
        'Utilities': 4000,
        'Entertainment': 2000
    }

    alerts = []
    next_month = forecast_df['ds'].max()
    for category, limit in budget.items():
        pred = forecast_df[(forecast_df['category']==category) & (forecast_df['ds']==next_month)]['yhat'].values
        if len(pred) > 0 and pred[0] > limit:
            alerts.append(f"{category}: Predicted {pred[0]:,.2f} exceeds budget {limit}")

    if alerts:
        for a in alerts:
            st.error(a)
    else:
        st.success("All categories are within budget!")

    # ------------------ Personal Monthly Forecast ------------------
    st.subheader("📊 Total Monthly Expense Forecast")
    forecast_df['month'] = forecast_df['ds'].dt.to_period('M')
    monthly_total = forecast_df.groupby('month')['yhat'].sum().reset_index()
    monthly_total['month'] = monthly_total['month'].astype(str)

    personal_budget = st.number_input("Set your personal monthly budget:", min_value=1000, value=20000, step=1000)
    monthly_total['alert'] = monthly_total['yhat'] > personal_budget

    fig_total = px.bar(
        monthly_total, x='month', y='yhat', text='yhat',
        labels={'month':'Month','yhat':'Predicted Total Expenses'},
        color='alert', color_discrete_map={True:'red', False:'green'},
        hover_data={'yhat':':,.2f'}
    )
    fig_total.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    st.plotly_chart(fig_total, use_container_width=True)

    # Alerts for months exceeding personal budget
    overshoot = monthly_total[monthly_total['alert']]
    if not overshoot.empty:
        for idx, row in overshoot.iterrows():
            st.error(f"⚠️ Predicted total spending {row['yhat']:,.2f} in {row['month']} exceeds your personal budget {personal_budget}")
    else:
        st.success("✅ Predicted total expenses are within your personal budget!")
