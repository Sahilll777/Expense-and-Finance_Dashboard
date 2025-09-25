# src/rule_based_mapper.py
import pandas as pd

CATEGORY_KEYWORDS = {
    'Food': ['swiggy', 'zomato', 'pizza', 'coffee', 'starbucks', 'restaurant'],
    'Shopping': ['amazon', 'flipkart', 'shopping', 'myntra', 'store', 'electronics'],
    'Transport': ['uber', 'ola', 'cab', 'taxi', 'ride'],
    'Subscription': ['netflix', 'spotify', 'prime', 'subscription', 'hotstar'],
    'Groceries': ['bigbasket', 'grocery', 'supermarket', 'dmart'],
    'Entertainment': ['movie', 'ticket', 'entertainment', 'play', 'event'],
    'Utilities': ['electricity', 'water', 'recharge', 'bill', 'utility'],
    'Income': ['salary', 'bonus', 'freelance', 'payment']
}

def map_category_rule_based(df: pd.DataFrame) -> pd.DataFrame:
    df['category_rule'] = 'Other'
    
    for category, keywords in CATEGORY_KEYWORDS.items():
        mask = df['desc_clean'].str.contains('|'.join(keywords), case=False, na=False)
        df.loc[mask, 'category_rule'] = category
    return df

if __name__ == '__main__':
    df = pd.read_csv('data/processed/transactions_processed.csv')
    df_mapped = map_category_rule_based(df)
    df_mapped.to_csv('data/processed/transactions_mapped.csv', index=False)
    print("✅ Rule-based categories added and saved to transactions_mapped.csv")
