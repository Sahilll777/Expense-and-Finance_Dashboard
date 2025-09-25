# notebooks/step2c_labeling.py
import pandas as pd

# 1️⃣ Load rule-based mapped CSV
df = pd.read_csv('data/processed/transactions_mapped.csv')

# 2️⃣ Filter uncategorized transactions
df_other = df[df['category_rule'] == 'Other']
print(f"Uncategorized transactions: {len(df_other)}")

# 3️⃣ Export a small subset for manual labeling
df_other_sample = df_other.head(20)
df_other_sample.to_csv('data/processed/other_sample.csv', index=False)
print("Sample saved to data/processed/other_sample.csv")

# ----
# 4️⃣ After manual labeling (add 'category_manual' in CSV), merge back
# Load manually labeled CSV after you edit it
# df_manual = pd.read_csv('data/processed/other_sample.csv')
# df.loc[df_manual.index, 'category_rule'] = df_manual['category_manual']
# df.to_csv('data/processed/transactions_labeled.csv', index=False)
# print("✅ transactions_labeled.csv is ready")

print("Next: Open other_sample.csv in Excel/Sheets and fill 'category_manual'. Then uncomment merge code to create transactions_labeled.csv")

