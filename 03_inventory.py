import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

set_df = df[df['Category'] == 'Set']
daily_demand = set_df.groupby('Date')['Qty'].sum().reset_index()
daily_demand.columns = ['Date', 'Qty']
daily_demand = daily_demand.iloc[1:-1].reset_index(drop=True)

split_index = int(len(daily_demand) * 0.8)
train = daily_demand[:split_index].copy()
test = daily_demand[split_index:].copy()

# ============================================
# STEP 2: Business Parameters
# ============================================
LEAD_TIME = 7        # days
Z = 1.65             # 95% service level
ORDER_COST = 500     # cost per order (INR)
HOLDING_COST = 2     # cost per unit per day (INR)
UNIT_COST = 300      # cost per unit (INR)

# ============================================
# STEP 3: Calculate Inventory Parameters
# ============================================
avg_daily_demand = train['Qty'].mean()
std_daily_demand = train['Qty'].std()

safety_stock = Z * std_daily_demand * np.sqrt(LEAD_TIME)
rop = (avg_daily_demand * LEAD_TIME) + safety_stock
annual_demand = avg_daily_demand * 365
eoq = np.sqrt((2 * annual_demand * ORDER_COST) / (HOLDING_COST * 365))

print("=== Inventory Parameters (Actual Demand) ===")
print(f"Avg Daily Demand: {round(avg_daily_demand, 1)} units")
print(f"Std Dev of Demand: {round(std_daily_demand, 1)} units")
print(f"Safety Stock: {round(safety_stock, 1)} units")
print(f"Reorder Point: {round(rop, 1)} units")
print(f"EOQ: {round(eoq, 1)} units")

# ============================================
# STEP 4: Recalculate using Prophet Forecast
# ============================================
from prophet import Prophet

# Prepare Prophet training data
prophet_train = train[['Date', 'Qty']].rename(columns={'Date': 'ds', 'Qty': 'y'})

# Build and fit model
model = Prophet(
    changepoint_prior_scale=0.5,
    yearly_seasonality=False,
    weekly_seasonality=True,
    daily_seasonality=False,
    changepoints=['2022-05-10']
)
model.fit(prophet_train)

# Forecast over training period to get fitted values
train_future = model.make_future_dataframe(periods=0)
train_forecast = model.predict(train_future)

# Merge forecast with actuals
train_eval = train.merge(
    train_forecast[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'prophet_pred'}),
    on='Date',
    how='left'
)

# Forecast error = actual - predicted
train_eval['forecast_error'] = train_eval['Qty'] - train_eval['prophet_pred']

# Prophet-based inventory parameters
# We use std dev of FORECAST ERROR instead of std dev of demand
# This is more accurate — safety stock only needs to cover what the forecast misses
prophet_avg = train_eval['prophet_pred'].mean()
prophet_error_std = train_eval['forecast_error'].std()

prophet_safety_stock = Z * prophet_error_std * np.sqrt(LEAD_TIME)
prophet_rop = (prophet_avg * LEAD_TIME) + prophet_safety_stock
prophet_annual_demand = prophet_avg * 365
prophet_eoq = np.sqrt((2 * prophet_annual_demand * ORDER_COST) / (HOLDING_COST * 365))

print("\n=== Inventory Parameters (Prophet Forecast) ===")
print(f"Avg Daily Demand (Prophet): {round(prophet_avg, 1)} units")
print(f"Std Dev of Forecast Error: {round(prophet_error_std, 1)} units")
print(f"Safety Stock: {round(prophet_safety_stock, 1)} units")
print(f"Reorder Point: {round(prophet_rop, 1)} units")
print(f"EOQ: {round(prophet_eoq, 1)} units")

# ============================================
# STEP 5: Compare the two approaches
# ============================================
print("\n=== Comparison: Naive vs Prophet ===")
print(f"Safety Stock reduction: {round(safety_stock - prophet_safety_stock, 1)} units")
print(f"Safety Stock reduction %: {round((safety_stock - prophet_safety_stock) / safety_stock * 100, 1)}%")
print(f"Capital freed up: INR {round((safety_stock - prophet_safety_stock) * UNIT_COST, 0):,.0f}")

# ============================================
# STEP 8: Day-by-Day Inventory Simulation
# ============================================

def simulate_inventory(demand_series, rop, eoq, safety_stock,
                       lead_time, holding_cost, order_cost, unit_cost):
    
    inventory = rop  # start with inventory at ROP level
    pending_orders = {}  # dictionary: {delivery_day: quantity}
    
    total_holding_cost = 0
    total_stockout_cost = 0
    total_order_cost = 0
    stockout_days = 0
    units_lost = 0
    
    inventory_levels = []
    stockout_flags = []
    
    for day_idx, row in demand_series.iterrows():
        day = day_idx
        demand = row['Qty']
        
        # 1. Receive pending order if due today
        if day in pending_orders:
            inventory += pending_orders[day]
            del pending_orders[day]
        
        # 2. Fulfill demand
        if inventory >= demand:
            inventory -= demand
        else:
            # Stockout — can only fulfill what's available
            units_lost += (demand - inventory)
            total_stockout_cost += (demand - inventory) * unit_cost
            stockout_days += 1
            inventory = 0
        
        # 3. Calculate today's holding cost
        total_holding_cost += inventory * holding_cost
        
        # 4. Check if we need to reorder
        if inventory <= rop and day not in pending_orders.values():
            delivery_day = day + lead_time
            pending_orders[delivery_day] = eoq
            total_order_cost += order_cost
        
        inventory_levels.append(inventory)
        stockout_flags.append(1 if inventory == 0 else 0)
    
    total_cost = total_holding_cost + total_stockout_cost + total_order_cost
    
    return {
        'inventory_levels': inventory_levels,
        'stockout_flags': stockout_flags,
        'total_holding_cost': round(total_holding_cost, 0),
        'total_stockout_cost': round(total_stockout_cost, 0),
        'total_order_cost': round(total_order_cost, 0),
        'total_cost': round(total_cost, 0),
        'stockout_days': stockout_days,
        'units_lost': round(units_lost, 0),
        'service_level': round((1 - stockout_days / len(demand_series)) * 100, 1)
    }

# Reset test index for simulation
test_sim = test.reset_index(drop=True)

# Run naive simulation
naive_results = simulate_inventory(
    demand_series=test_sim,
    rop=rop,
    eoq=eoq,
    safety_stock=safety_stock,
    lead_time=LEAD_TIME,
    holding_cost=HOLDING_COST,
    order_cost=ORDER_COST,
    unit_cost=UNIT_COST
)

# Run Prophet simulation
prophet_results = simulate_inventory(
    demand_series=test_sim,
    rop=prophet_rop,
    eoq=prophet_eoq,
    safety_stock=prophet_safety_stock,
    lead_time=LEAD_TIME,
    holding_cost=HOLDING_COST,
    order_cost=ORDER_COST,
    unit_cost=UNIT_COST
)

# ============================================
# STEP 9: Print Simulation Results
# ============================================
print("\n=== Simulation Results (18-day test period) ===")
print(f"{'Metric':<30} {'Naive':>12} {'Prophet':>12}")
print("-" * 55)
print(f"{'Stockout Days':<30} {naive_results['stockout_days']:>12} {prophet_results['stockout_days']:>12}")
print(f"{'Units Lost':<30} {naive_results['units_lost']:>12} {prophet_results['units_lost']:>12}")
print(f"{'Service Level %':<30} {naive_results['service_level']:>12} {prophet_results['service_level']:>12}")
print(f"{'Holding Cost (INR)':<30} {naive_results['total_holding_cost']:>12,.0f} {prophet_results['total_holding_cost']:>12,.0f}")
print(f"{'Stockout Cost (INR)':<30} {naive_results['total_stockout_cost']:>12,.0f} {prophet_results['total_stockout_cost']:>12,.0f}")
print(f"{'Order Cost (INR)':<30} {naive_results['total_order_cost']:>12,.0f} {prophet_results['total_order_cost']:>12,.0f}")
print(f"{'Total Cost (INR)':<30} {naive_results['total_cost']:>12,.0f} {prophet_results['total_cost']:>12,.0f}")
print(f"{'Cost Saved (INR)':<30} {naive_results['total_cost'] - prophet_results['total_cost']:>12,.0f}")

# ============================================
# STEP 10: Visualize Inventory Levels
# ============================================
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

axes[0].plot(test_sim.index, naive_results['inventory_levels'],
             label='Inventory Level', color='blue')
axes[0].axhline(y=rop, color='red', linestyle='--', label=f'ROP ({round(rop,0)})')
axes[0].axhline(y=safety_stock, color='orange', linestyle='--',
                label=f'Safety Stock ({round(safety_stock,0)})')
axes[0].set_title('Naive Approach - Inventory Levels')
axes[0].set_ylabel('Units')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(test_sim.index, prophet_results['inventory_levels'],
             label='Inventory Level', color='green')
axes[1].axhline(y=prophet_rop, color='red', linestyle='--',
                label=f'ROP ({round(prophet_rop,0)})')
axes[1].axhline(y=prophet_safety_stock, color='orange', linestyle='--',
                label=f'Safety Stock ({round(prophet_safety_stock,0)})')
axes[1].set_title('Prophet Approach - Inventory Levels')
axes[1].set_ylabel('Units')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('inventory_simulation.png')
plt.show()