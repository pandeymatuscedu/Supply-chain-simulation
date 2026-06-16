import pandas as pd
import matplotlib.pyplot as plt

# ============================================
# STEP 1: Load the raw data
# ============================================
df = pd.read_csv("Amazon Sale Report.csv", encoding="latin-1")

print("Raw shape:", df.shape)

# ============================================
# STEP 2: Clean the data
# ============================================

# Drop columns we don't need
df = df.drop(columns=['fulfilled-by', 'Unnamed: 22', 'index'])

# Remove orders that don't represent real demand
remove_statuses = [
    'Cancelled',
    'Shipped - Returned to Seller',
    'Shipped - Rejected by Buyer',
    'Shipped - Damaged'
]
df = df[~df['Status'].isin(remove_statuses)]

# Convert Date column from text to actual date format
df['Date'] = pd.to_datetime(df['Date'])

print("Clean shape:", df.shape)
print("Date range:", df['Date'].min(), "to", df['Date'].max())

# ============================================
# STEP 3: Focus on one category ("Set")
# ============================================
set_df = df[df['Category'] == 'Set']

# Build daily demand: one row per day, total Qty sold that day
daily_demand = set_df.groupby('Date')['Qty'].sum().reset_index()
daily_demand.columns = ['Date', 'Qty']

print("\nDaily demand shape:", daily_demand.shape)
print(daily_demand.head())

# ============================================
# STEP 4: Visualize daily demand
# ============================================
plt.figure(figsize=(12, 5))
plt.plot(daily_demand['Date'], daily_demand['Qty'])
plt.title('Daily Demand - Set Category')
plt.xlabel('Date')
plt.ylabel('Quantity Sold')
plt.grid(True)
plt.savefig('daily_demand_plot.png')
plt.show()

# ============================================
# INVESTIGATION: Why the level shift?
# ============================================

# 1. Total demand across ALL categories, by day
all_categories_daily = df.groupby('Date')['Qty'].sum().reset_index()
all_categories_daily.columns = ['Date', 'Qty']

plt.figure(figsize=(12, 5))
plt.plot(all_categories_daily['Date'], all_categories_daily['Qty'])
plt.title('Daily Demand - ALL Categories')
plt.xlabel('Date')
plt.ylabel('Quantity Sold')
plt.grid(True)
plt.savefig('all_categories_demand_plot.png')
plt.show()

# 2. Promotion usage for "Set" category, before vs after May 15
set_df_copy = set_df.copy()
set_df_copy['has_promo'] = set_df_copy['promotion-ids'].notna()

before = set_df_copy[set_df_copy['Date'] < '2022-05-15']
after = set_df_copy[set_df_copy['Date'] >= '2022-05-15']

print("\nPromo % before May 15:", before['has_promo'].mean() * 100)
print("Promo % after May 15:", after['has_promo'].mean() * 100)

# 3. Order count vs quantity per order, before vs after
print("\nBefore May 15 - Orders:", len(before), "| Avg Qty per order:", before['Qty'].mean())
print("After May 15 - Orders:", len(after), "| Avg Qty per order:", after['Qty'].mean())