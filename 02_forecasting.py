import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

# ============================================
# STEP 1: Load and clean data
# ============================================
df = pd.read_csv("Amazon Sale Report.csv", encoding="latin-1")

df = df.drop(columns=['fulfilled-by', 'Unnamed: 22', 'index'])

remove_statuses = [
    'Cancelled',
    'Shipped - Returned to Seller',
    'Shipped - Rejected by Buyer',
    'Shipped - Damaged'
]
df = df[~df['Status'].isin(remove_statuses)]
df['Date'] = pd.to_datetime(df['Date'])

# Filter to Set category and build daily demand
set_df = df[df['Category'] == 'Set']
daily_demand = set_df.groupby('Date')['Qty'].sum().reset_index()
daily_demand.columns = ['Date', 'Qty']

# Drop first and last days (partial days)
daily_demand = daily_demand.iloc[1:-1].reset_index(drop=True)

print("Forecasting dataset shape:", daily_demand.shape)
print("Date range:", daily_demand['Date'].min(), "to", daily_demand['Date'].max())

# ============================================
# STEP 2: Train/Test Split (80/20)
# ============================================
split_index = int(len(daily_demand) * 0.8)
train = daily_demand[:split_index].copy()
test = daily_demand[split_index:].copy()

print("\nTraining days:", len(train))
print("Testing days:", len(test))
print("Test starts on:", test['Date'].min())

# ============================================
# STEP 3: Baseline Model (7-day Rolling Average)
# ============================================
train['rolling_avg'] = train['Qty'].rolling(window=7).mean()
baseline_forecast = train['rolling_avg'].iloc[-1]

print("\nBaseline forecast (7-day rolling avg):", round(baseline_forecast, 1))
print("Actual avg demand in test period:", round(test['Qty'].mean(), 1))

test['baseline_pred'] = baseline_forecast

# ============================================
# STEP 4: MAPE Function
# ============================================
def mape(actual, predicted):
    return round((abs(actual - predicted) / actual).mean() * 100, 2)

baseline_mape = mape(test['Qty'], test['baseline_pred'])
print("\nBaseline MAPE:", baseline_mape, "%")

# ============================================
# STEP 5: Visualize Baseline vs Actual
# ============================================
plt.figure(figsize=(12, 5))
plt.plot(train['Date'], train['Qty'], label='Training Actual', color='blue')
plt.plot(test['Date'], test['Qty'], label='Test Actual', color='green')
plt.axhline(y=baseline_forecast, color='red', linestyle='--',
            label=f'Baseline Forecast ({round(baseline_forecast, 1)})')
plt.title('Baseline Forecast vs Actual Demand')
plt.xlabel('Date')
plt.ylabel('Quantity')
plt.legend()
plt.grid(True)
plt.savefig('baseline_forecast.png')
plt.show()

# ============================================
# STEP 6: Prophet Model
# ============================================

# Prophet requires columns named 'ds' and 'y'
prophet_train = train[['Date', 'Qty']].rename(columns={'Date': 'ds', 'Qty': 'y'})

# Build and train the model
model = Prophet(
    changepoint_prior_scale=0.5,
    yearly_seasonality=False,
    weekly_seasonality=True,
    daily_seasonality=False,
    changepoints=['2022-05-10']
)

model.fit(prophet_train)  # <- this must come BEFORE make_future_dataframe

future = model.make_future_dataframe(periods=len(test))
forecast = model.predict(future)



# Create future dataframe for test period
future = model.make_future_dataframe(periods=len(test))
forecast = model.predict(future)

# Extract test period predictions only
prophet_test_forecast = forecast[forecast['ds'] >= test['Date'].min()][['ds', 'yhat']]
prophet_test_forecast = prophet_test_forecast.rename(
    columns={'ds': 'Date', 'yhat': 'prophet_pred'}
)

# Merge predictions with test actuals
test = test.merge(prophet_test_forecast, on='Date', how='left')

# ============================================
# STEP 7: Compare Models
# ============================================
prophet_mape = mape(test['Qty'], test['prophet_pred'])

print("\n--- Model Comparison ---")
print("Baseline MAPE:", baseline_mape, "%")
print("Prophet MAPE:", prophet_mape, "%")
print("Improvement:", round(baseline_mape - prophet_mape, 2), "percentage points")

# ============================================
# STEP 8: Visualize Prophet vs Baseline vs Actual
# ============================================
plt.figure(figsize=(12, 5))
plt.plot(train['Date'], train['Qty'], label='Training Actual', color='blue')
plt.plot(test['Date'], test['Qty'], label='Test Actual', color='green')
plt.plot(test['Date'], test['prophet_pred'], label='Prophet Forecast',
         color='orange', linestyle='--')
plt.axhline(y=baseline_forecast, color='red', linestyle='--',
            label=f'Baseline ({round(baseline_forecast, 1)})')
plt.title('Prophet vs Baseline Forecast')
plt.xlabel('Date')
plt.ylabel('Quantity')
plt.legend()
plt.grid(True)
plt.savefig('prophet_vs_baseline.png')
plt.show()