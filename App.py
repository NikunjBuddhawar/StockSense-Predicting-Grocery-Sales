import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model and scaler using pickle
with open("final_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("expected_cols.pkl", "rb") as f:
    expected_cols = pickle.load(f)

numerical = ['Stock_Quantity', 'Reorder_Level', 'Unit_Price',
             'Inventory_Turnover_Rate', 'Shelf_Life_Days',
             'Reorder_Margin', 'Sales_per_Stock', 'Days_Since_Received']

st.title("🛒 Grocery Sales Volume Predictor")
st.markdown("Predict the expected sales volume based on inventory and product info.")

# --- Input form ---
with st.form("prediction_form"):
    st.subheader("📦 Product Info")
    stock_qty = st.number_input("Stock Quantity", min_value=0, value=40)
    reorder_lvl = st.number_input("Reorder Level", min_value=0, value=20)
    unit_price = st.number_input("Unit Price (₹)", min_value=0.0, value=25.0)
    turnover_rate = st.number_input("Inventory Turnover Rate", min_value=0, value=5)
    shelf_life = st.number_input("Shelf Life (Days)", min_value=0, value=60)
    days_since = st.number_input("Days Since Received", min_value=0, value=15)

    cat = st.selectbox("Category", ["Beverages", "Grains & Pulses", "Oils & Fats", "Seafood"])
    wh = st.selectbox("Warehouse", ["A1", "B1", "C1"])
    supp = st.selectbox("Supplier", ["ABC Suppliers", "XYZ Traders", "Fresh & Co"])

    submitted = st.form_submit_button("Predict")

# --- Prediction logic ---
if submitted:
    reorder_margin = stock_qty - reorder_lvl
    sales_per_stock = 1.2  # Placeholder or an estimated value

    input_data = {
        'Stock_Quantity': stock_qty,
        'Reorder_Level': reorder_lvl,
        'Unit_Price': unit_price,
        'Inventory_Turnover_Rate': turnover_rate,
        'Shelf_Life_Days': shelf_life,
        'Reorder_Margin': reorder_margin,
        'Sales_per_Stock': sales_per_stock,
        'Days_Since_Received': days_since,
        f'Catagory_{cat}': 1,
        f'Warehouse_Location_{wh}': 1,
        f'Supplier_Name_{supp}': 1
    }

    input_df = pd.DataFrame([input_data])

    for col in expected_cols:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected_cols]

    input_df[numerical] = scaler.transform(input_df[numerical])

    prediction = model.predict(input_df)[0]

    st.success(f"📈 Predicted Sales Volume: **{prediction:.2f} units**")
