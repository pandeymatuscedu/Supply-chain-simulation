import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="Supply Chain Simulation",
    page_icon="📦",
    layout="wide"
)

st.title("📦 Amazon Supply Chain Simulation")
st.markdown("**End-to-end demand forecasting and inventory optimization for Amazon India fashion category**")
st.markdown("---")

# ============================================
# LOAD AND CLEAN DATA
# ============================================
@st.cache_data
def load_data():
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
    return daily_demand

daily_demand = load_data()

split_index = int(len(daily_demand) * 0.8)
train = daily_demand[:split_index].copy()
test = daily_demand[split_index:].copy()

# ============================================
# SIDEBAR - USER INPUTS
# ============================================
st.sidebar.header("⚙️ Simulation Parameters")

LEAD_TIME = st.sidebar.slider("Lead Time (days)", min_value=1, max_value=30, value=7)
Z = st.sidebar.selectbox("Service Level", options=[1.28, 1.65, 2.05, 2.33],
                          format_func=lambda x: {1.28: "90%", 1.65: "95%",
                                                  2.05: "98%", 2.33: "99%"}[x])
ORDER_COST = st.sidebar.number_input("Order Cost (INR)", value=500)
HOLDING_COST = st.sidebar.number_input("Holding Cost per unit/day (INR)", value=2)
UNIT_COST = st.sidebar.number_input("Unit Cost (INR)", value=300)

# ============================================
# SECTION 1: DEMAND OVERVIEW
# ============================================
st.header("1️⃣ Demand Analysis")

col1, col2, col3 = st.columns(3)
col1.metric("Total Days", len(daily_demand))
col2.metric("Avg Daily Demand", f"{round(daily_demand['Qty'].mean(), 1)} units")
col3.metric("Demand Std Dev", f"{round(daily_demand['Qty'].std(), 1)} units")

fig1, ax1 = plt.subplots(figsize=(12, 4))
ax1.plot(daily_demand['Date'], daily_demand['Qty'], color='steelblue')
ax1.axvline(pd.Timestamp('2022-05-10'), color='red', linestyle='--',
            label='Eid Changepoint (May 10)')
ax1.set_title('Daily Demand - Set Category')
ax1.set_xlabel('Date')
ax1.set_ylabel('Quantity')
ax1.legend()
ax1.grid(True)
st.pyplot(fig1)

st.markdown("""
**Key Finding:** Demand peaked around Eid al-Fitr (May 2-3, 2022) at ~700 units/day,
then dropped ~35% to a stable plateau of ~380 units/day post-festival.
This structural shift (regime change) is a critical input to our forecasting approach.
""")

# ============================================
# SECTION 2: FORECASTING
# ============================================
st.header("2️⃣ Demand Forecasting")

with st.spinner("Training Prophet model..."):
    prophet_train = train[['Date', 'Qty']].rename(columns={'Date': 'ds', 'Qty': 'y'})
    model = Prophet(
        changepoint_prior_scale=0.5,
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoints=['2022-05-10']
    )
    model.fit(prophet_train)
    future = model.make_future_dataframe(periods=len(test))
    forecast = model.predict(future)

prophet_test = forecast[forecast['ds'] >= test['Date'].min()][['ds', 'yhat']]
prophet_test = prophet_test.rename(columns={'ds': 'Date', 'yhat': 'prophet_pred'})
test = test.merge(prophet_test, on='Date', how='left')
test['baseline_pred'] = train['Qty'].rolling(7).mean().iloc[-1]

def mape(actual, predicted):
    return round((abs(actual - predicted) / actual).mean() * 100, 2)

baseline_mape = mape(test['Qty'], test['baseline_pred'])
prophet_mape = mape(test['Qty'], test['prophet_pred'])

col1, col2, col3 = st.columns(3)
col1.metric("Baseline MAPE", f"{baseline_mape}%")
col2.metric("Prophet MAPE", f"{prophet_mape}%")
col3.metric("Improvement", f"{round(baseline_mape - prophet_mape, 2)} pp",
            delta=f"{round((baseline_mape - prophet_mape)/baseline_mape*100, 1)}% better")

fig2, ax2 = plt.subplots(figsize=(12, 4))
ax2.plot(train['Date'], train['Qty'], label='Training Actual', color='blue')
ax2.plot(test['Date'], test['Qty'], label='Test Actual', color='green')
ax2.plot(test['Date'], test['prophet_pred'], label='Prophet Forecast',
         color='orange', linestyle='--')
ax2.axhline(y=test['baseline_pred'].iloc[0], color='red', linestyle='--',
            label=f'Baseline ({round(test["baseline_pred"].iloc[0], 0)})')
ax2.set_title('Prophet vs Baseline Forecast')
ax2.set_xlabel('Date')
ax2.set_ylabel('Quantity')
ax2.legend()
ax2.grid(True)
st.pyplot(fig2)

# ============================================
# SECTION 3: INVENTORY PARAMETERS
# ============================================
st.header("3️⃣ Inventory Optimization")

avg_demand = train['Qty'].mean()
std_demand = train['Qty'].std()

train_eval = train.copy()
train_future = model.make_future_dataframe(periods=0)
train_forecast = model.predict(train_future)
train_eval = train.merge(
    train_forecast[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'prophet_pred'}),
    on='Date', how='left'
)
train_eval['forecast_error'] = train_eval['Qty'] - train_eval['prophet_pred']
prophet_error_std = train_eval['forecast_error'].std()
prophet_avg = train_eval['prophet_pred'].mean()

naive_ss = Z * std_demand * np.sqrt(LEAD_TIME)
naive_rop = (avg_demand * LEAD_TIME) + naive_ss
naive_eoq = np.sqrt((2 * avg_demand * 365 * ORDER_COST) / (HOLDING_COST * 365))

prophet_ss = Z * prophet_error_std * np.sqrt(LEAD_TIME)
prophet_rop = (prophet_avg * LEAD_TIME) + prophet_ss
prophet_eoq = np.sqrt((2 * prophet_avg * 365 * ORDER_COST) / (HOLDING_COST * 365))

col1, col2, col3 = st.columns(3)
col1.metric("Safety Stock Reduction",
            f"{round(naive_ss - prophet_ss, 0)} units",
            delta=f"{round((naive_ss - prophet_ss)/naive_ss*100, 1)}% less")
col2.metric("Capital Freed",
            f"INR {round((naive_ss - prophet_ss) * UNIT_COST, 0):,.0f}")
col3.metric("ROP Reduction",
            f"{round(naive_rop - prophet_rop, 0)} units")

fig3, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(['Naive', 'Prophet'], [naive_ss, prophet_ss], color=['red', 'green'])
axes[0].set_title('Safety Stock Comparison')
axes[0].set_ylabel('Units')
for i, v in enumerate([naive_ss, prophet_ss]):
    axes[0].text(i, v + 5, str(round(v, 0)), ha='center')

axes[1].bar(['Naive', 'Prophet'], [naive_rop, prophet_rop], color=['red', 'green'])
axes[1].set_title('Reorder Point Comparison')
axes[1].set_ylabel('Units')
for i, v in enumerate([naive_rop, prophet_rop]):
    axes[1].text(i, v + 5, str(round(v, 0)), ha='center')

plt.tight_layout()
st.pyplot(fig3)

# ============================================
# SECTION 4: SIMULATION RESULTS
# ============================================
st.header("4️⃣ Cost Simulation")

def simulate_inventory(demand_series, rop, eoq, lead_time,
                       holding_cost, order_cost, unit_cost):
    inventory = rop
    pending_orders = {}
    total_holding_cost = 0
    total_stockout_cost = 0
    total_order_cost = 0
    stockout_days = 0
    units_lost = 0
    inventory_levels = []

    for day_idx, row in demand_series.iterrows():
        demand = row['Qty']
        if day_idx in pending_orders:
            inventory += pending_orders[day_idx]
            del pending_orders[day_idx]
        if inventory >= demand:
            inventory -= demand
        else:
            units_lost += (demand - inventory)
            total_stockout_cost += (demand - inventory) * unit_cost
            stockout_days += 1
            inventory = 0
        total_holding_cost += inventory * holding_cost
        if inventory <= rop:
            delivery_day = day_idx + lead_time
            if delivery_day not in pending_orders:
                pending_orders[delivery_day] = eoq
                total_order_cost += order_cost
        inventory_levels.append(inventory)

    total_cost = total_holding_cost + total_stockout_cost + total_order_cost
    return {
        'inventory_levels': inventory_levels,
        'total_holding_cost': round(total_holding_cost, 0),
        'total_stockout_cost': round(total_stockout_cost, 0),
        'total_order_cost': round(total_order_cost, 0),
        'total_cost': round(total_cost, 0),
        'stockout_days': stockout_days,
        'units_lost': round(units_lost, 0),
        'service_level': round((1 - stockout_days / len(demand_series)) * 100, 1)
    }

test_sim = test.reset_index(drop=True)
naive_results = simulate_inventory(test_sim, naive_rop, naive_eoq,
                                    LEAD_TIME, HOLDING_COST, ORDER_COST, UNIT_COST)
prophet_results = simulate_inventory(test_sim, prophet_rop, prophet_eoq,
                                      LEAD_TIME, HOLDING_COST, ORDER_COST, UNIT_COST)

results_df = pd.DataFrame({
    'Metric': ['Stockout Days', 'Units Lost', 'Service Level %',
               'Holding Cost (INR)', 'Stockout Cost (INR)',
               'Order Cost (INR)', 'Total Cost (INR)'],
    'Naive': [naive_results['stockout_days'], naive_results['units_lost'],
              naive_results['service_level'], naive_results['total_holding_cost'],
              naive_results['total_stockout_cost'], naive_results['total_order_cost'],
              naive_results['total_cost']],
    'Prophet': [prophet_results['stockout_days'], prophet_results['units_lost'],
                prophet_results['service_level'], prophet_results['total_holding_cost'],
                prophet_results['total_stockout_cost'], prophet_results['total_order_cost'],
                prophet_results['total_cost']]
})

st.dataframe(results_df, use_container_width=True)

cost_saved = naive_results['total_cost'] - prophet_results['total_cost']
st.success(f"💰 Total Cost Saved with Prophet Forecasting: INR {cost_saved:,.0f} over 18-day test period")
st.info(f"📈 Annualized savings estimate: INR {round(cost_saved * 365/18, 0):,.0f} per SKU")

fig4, axes = plt.subplots(2, 1, figsize=(12, 7))
axes[0].plot(test_sim.index, naive_results['inventory_levels'], color='red', label='Inventory')
axes[0].axhline(y=naive_rop, color='gray', linestyle='--', label=f'ROP ({round(naive_rop,0)})')
axes[0].axhline(y=naive_ss, color='orange', linestyle='--',
                label=f'Safety Stock ({round(naive_ss,0)})')
axes[0].set_title('Naive Approach - Inventory Levels')
axes[0].set_ylabel('Units')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(test_sim.index, prophet_results['inventory_levels'], color='green', label='Inventory')
axes[1].axhline(y=prophet_rop, color='gray', linestyle='--', label=f'ROP ({round(prophet_rop,0)})')
axes[1].axhline(y=prophet_ss, color='orange', linestyle='--',
                label=f'Safety Stock ({round(prophet_ss,0)})')
axes[1].set_title('Prophet Approach - Inventory Levels')
axes[1].set_ylabel('Units')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
st.pyplot(fig4)

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("Built by **Mayank Pandey** | USC Viterbi Engineering Management | [LinkedIn](https://linkedin.com/in/mayankmpandey)")