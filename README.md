# Expense-and-Finance_Dashboard

End-to-end Streamlit app to clean bank CSVs, auto-categorize transactions, forecast spending per category, and show budget alerts.

# Run locally

Create venv and install:

python -m venv venv
source venv/bin/activate    
pip install -r requirements.txt

python src/data_preprocess.py data/sample_transactions.csv data/processed/transactions_processed.csv  # Preprocess Sample Data

python src/expense_forecast.py  # To Generate Forecast

streamlit run app.py  # To Run the app

