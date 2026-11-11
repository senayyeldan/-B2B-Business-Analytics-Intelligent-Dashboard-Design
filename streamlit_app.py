# -*- coding: utf-8 -*-
"""
B2B Business Analytics Intelligent Dashboard Design
Author: Şenay Yeldan

"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="B2B Business Analytics",
    page_icon="📊",
    layout="wide"
)

# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_data():

    file_path = Path(__file__).parent / "df_analytical.csv.gz"

    data = pd.read_csv(
        file_path,
        parse_dates=["InvoiceDate"],
        dtype={
            "InvoiceNo": "string",
            "CustomerID": "string",
            "StockCode": "string"
        }
    )

    # Identify transaction-level refund records.
    refund_mask = (
        (data["Quantity"] == 0) &
        (data["NetPrice"] < 0)
    )

    data["Refund_Amount"] = 0.0
    data.loc[refund_mask, "Refund_Amount"] = (
        -data.loc[refund_mask, "NetPrice"]
    )

    # Calculate transaction-line revenue after refunds.
    data["Revenue"] = data["Quantity"] * data["NetPrice"]
    data["Net_Revenue"] = data["Revenue"] - data["Refund_Amount"]

    return data


df_all = load_data()

# --------------------------------------------------
# 3. SIDEBAR: FILTERS AND NAVIGATION
# --------------------------------------------------

st.sidebar.title("📊 B2B Analytics")

st.sidebar.subheader("Dashboard Filters")


min_date = df_all["InvoiceDate"].min().date()
max_date = df_all["InvoiceDate"].max().date()

st.write("Minimum date:", min_date)
st.write("Maximum date:", max_date)

selected_dates = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Apply the date filter only after both dates have been selected.
if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:
    start_date, end_date = selected_dates

    df = df_all[
        df_all["InvoiceDate"].dt.date.between(start_date, end_date)
    ].copy()
else:
    df = df_all.copy()

st.sidebar.divider()
st.sidebar.subheader("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Executive Overview",
        "Sales Analytics",
        "Marketing Analytics",
        "Supply Chain Analytics",
        "Demand Prediction",
        "Dataset Overview"
    ],
    label_visibility="collapsed"
)

st.sidebar.divider()
st.sidebar.caption("B2B Business Intelligence | 2020")

# --------------------------------------------------
# 4. VALIDATE FILTERED DATA
# --------------------------------------------------

st.title("B2B Business Performance Dashboard")
st.caption("Business Intelligence and Predictive Analytics | 2020")

if df.empty:
    st.warning(
        "There are no transactions in the selected date range. "
        "Please choose another date range."
    )
    st.stop()

# --------------------------------------------------
# 5. EXECUTIVE OVERVIEW
# --------------------------------------------------

if page == "Executive Overview":

    st.header("Executive Overview")
    st.caption("A high-level view of business performance.")

    # Calculate the main business KPIs.
    total_revenue = df["Net_Revenue"].sum()
    total_quantity = df["Quantity"].sum()
    total_invoices = df["InvoiceNo"].nunique()
    total_customers = df["CustomerID"].nunique()
    total_products = df["StockCode"].nunique()

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Net Revenue", f"₺{total_revenue:,.2f}")
    col2.metric("Sales Volume", f"{total_quantity:,.0f}")
    col3.metric("Total Invoices", f"{total_invoices:,}")
    col4.metric("Active Customers", f"{total_customers:,}")
    col5.metric("Active Products", f"{total_products:,}")

    st.divider()

    # Aggregate revenue and sales volume by calendar month.
    monthly_performance = (
        df.groupby(df["InvoiceDate"].dt.month)
        .agg(
            Net_Revenue=("Net_Revenue", "sum"),
            Sales_Volume=("Quantity", "sum")
        )
        .reindex(range(1, 13), fill_value=0)
    )

    monthly_performance.index = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ]

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.subheader("Monthly Net Revenue")
        st.bar_chart(monthly_performance["Net_Revenue"])

    with col2:
        st.subheader("Monthly Sales Volume")
        st.bar_chart(monthly_performance["Sales_Volume"])

    st.divider()

    # Rank products by net revenue.
    top_products = (
        df.groupby(["StockCode", "Description"])["Net_Revenue"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    # Rank categories by net revenue.
    top_categories = (
        df.dropna(subset=["Category"])
        .groupby("Category")["Net_Revenue"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.subheader("Top 10 Products by Net Revenue")
        st.bar_chart(
            top_products.set_index("Description")["Net_Revenue"],
            horizontal=True
        )

    with col2:
        st.subheader("Top 10 Categories by Net Revenue")
        st.bar_chart(top_categories, horizontal=True)
        st.caption("Categories with missing values are excluded.")

# --------------------------------------------------
# 6. SALES ANALYTICS
# --------------------------------------------------

elif page == "Sales Analytics":

    st.header("Sales Analytics")
    st.caption("Revenue performance, customer activity and product sales.")

    # Calculate average revenue per invoice.
    invoice_count = df["InvoiceNo"].nunique()
    average_invoice_revenue = (
        df["Net_Revenue"].sum() / invoice_count
        if invoice_count else 0
    )

    # Calculate the weighted discount rate.
    total_discount_value = (
        (df["UnitPrice"] - df["NetPrice"]) * df["Quantity"]
    ).sum()

    total_list_price_value = (
        df["UnitPrice"] * df["Quantity"]
    ).sum()

    weighted_discount_rate = (
        total_discount_value / total_list_price_value
        if total_list_price_value else 0
    )

    col1, col2 = st.columns(2, gap="large")

    col1.metric(
        "Average Revenue per Invoice",
        f"₺{average_invoice_revenue:,.2f}"
    )

    col2.metric(
        "Weighted Discount Rate",
        f"{weighted_discount_rate:.2%}"
    )

    st.divider()

    # Top customers by net revenue.
    top_customers = (
        df.groupby("CustomerID")["Net_Revenue"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    # Top customers by purchasing frequency.
    top_customers_frequency = (
        df.groupby("CustomerID")["InvoiceNo"]
        .nunique()
        .sort_values(ascending=False)
        .head(10)
    )

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.subheader("Top 10 Customers by Net Revenue")
        st.bar_chart(top_customers, horizontal=True)

    with col2:
        st.subheader("Top 10 Customers by Purchasing Frequency")
        st.bar_chart(top_customers_frequency, horizontal=True)

    st.divider()

    # Top products by sales volume.
    top_products_volume = (
        df.groupby("Description")["Quantity"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    # Top categories by sales volume.
    top_categories_volume = (
        df.dropna(subset=["Category"])
        .groupby("Category")["Quantity"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.subheader("Top 10 Products by Sales Volume")
        st.bar_chart(top_products_volume, horizontal=True)

    with col2:
        st.subheader("Top 10 Categories by Sales Volume")
        st.bar_chart(top_categories_volume, horizontal=True)
        st.caption("Categories with missing values are excluded.")

# --------------------------------------------------
# 7. MARKETING ANALYTICS
# --------------------------------------------------

elif page == "Marketing Analytics":

    st.header("Marketing Analytics")
    st.caption("Customer purchasing behaviour, value and category preferences.")

    # Calculate customer purchasing activity.
    customer_invoices = (
        df.groupby("CustomerID")["InvoiceNo"]
        .nunique()
    )

    average_invoices_per_customer = (
        customer_invoices.mean()
        if not customer_invoices.empty else 0
    )

    # Calculate average net revenue per customer.
    customer_revenue = (
        df.groupby("CustomerID")["Net_Revenue"]
        .sum()
    )

    average_customer_revenue = (
        customer_revenue.mean()
        if not customer_revenue.empty else 0
    )

    col1, col2 = st.columns(2, gap="large")

    col1.metric(
        "Average Invoices per Customer",
        f"{average_invoices_per_customer:.2f}"
    )

    col2.metric(
        "Average Customer Revenue",
        f"₺{average_customer_revenue:,.2f}"
    )

    st.divider()

    # Divide customers into four equal-sized value segments.
    customer_segments = pd.qcut(
        customer_revenue.rank(method="first"),
        q=4,
        labels=[
            "Low Value",
            "Medium Value",
            "High Value",
            "Very High Value"
        ]
    )

    # Combine customer revenue and segment information.
    customer_analysis = customer_revenue.to_frame(
        name="Net Revenue"
    )
    customer_analysis["Segment"] = customer_segments

    # Calculate total revenue generated by each segment.
    customer_segment_revenue = (
        customer_analysis
        .groupby("Segment", observed=False)["Net Revenue"]
        .sum()
        .reindex([
            "Low Value",
            "Medium Value",
            "High Value",
            "Very High Value"
        ])
    )

    st.subheader("Net Revenue by Customer Segment")
    st.bar_chart(customer_segment_revenue, horizontal=True)

    st.divider()

    # Count unique customers purchasing each category.
    category_customer_reach = (
        df.dropna(subset=["Category"])
        .groupby("Category")["CustomerID"]
        .nunique()
        .sort_values(ascending=False)
        .head(10)
    )

    st.subheader("Top 10 Categories by Customer Reach")
    st.bar_chart(category_customer_reach, horizontal=True)

    st.caption(
        "Customer reach represents the number of unique customers "
        "who purchased from each category."
    )

# --------------------------------------------------
# 8. SUPPLY CHAIN ANALYTICS
# --------------------------------------------------

elif page == "Supply Chain Analytics":

    st.header("Supply Chain Analytics")
    st.caption("Product demand patterns and demand distribution.")

    # Calculate daily demand, including days without transactions.
    daily_demand = (
        df.groupby(df["InvoiceDate"].dt.normalize())["Quantity"]
        .sum()
    )

    date_range = pd.date_range(
        start=df["InvoiceDate"].min().normalize(),
        end=df["InvoiceDate"].max().normalize(),
        freq="D"
    )

    daily_demand = daily_demand.reindex(
        date_range,
        fill_value=0
    )

    # Calculate supply chain KPIs.
    total_demand = df["Quantity"].sum()
    average_daily_demand = daily_demand.mean()

    col1, col2 = st.columns(2, gap="large")

    col1.metric("Total Sales Volume", f"{total_demand:,.0f}")
    col2.metric("Average Daily Demand", f"{average_daily_demand:,.2f}")

    st.divider()

    # Calculate monthly demand.
    monthly_demand = (
        df.groupby(df["InvoiceDate"].dt.month)["Quantity"]
        .sum()
        .reindex(range(1, 13), fill_value=0)
    )

    monthly_demand.index = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ]

    # Identify the products with the highest demand.
    top_demand_products = (
        df.groupby(["StockCode", "Description"])["Quantity"]
        .sum()
        .sort_values(ascending=False)
        .head(20)
        .reset_index()
    )

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.subheader("Monthly Demand Trend")
        st.bar_chart(monthly_demand)

    with col2:
        number_of_products = st.slider(
            "Number of products to display",
            min_value=5,
            max_value=20,
            value=10,
            step=5
        )

        st.subheader("Top Products by Demand")
        st.bar_chart(
            top_demand_products
            .head(number_of_products)
            .set_index("Description")["Quantity"],
            horizontal=True
        )

# --------------------------------------------------
# 9. DEMAND PREDICTION
# --------------------------------------------------

elif page == "Demand Prediction":

    st.header("Demand Prediction & Business Insights")
    st.caption("Comparison of regression models using chronological test data.")

    # Prepare daily demand data, including days without transactions.
    daily_demand = (
        df.groupby(df["InvoiceDate"].dt.normalize())["Quantity"]
        .sum()
    )

    date_range = pd.date_range(
        start=df["InvoiceDate"].min().normalize(),
        end=df["InvoiceDate"].max().normalize(),
        freq="D"
    )

    daily_demand = daily_demand.reindex(
        date_range,
        fill_value=0
    )

    if len(daily_demand) < 10:
        st.warning(
            "Please select a wider date range to train and evaluate "
            "the demand prediction models."
        )
        st.stop()

    # Prepare time-based predictor variables.
    prediction_data = daily_demand.to_frame(name="Demand")
    prediction_data["Day_Number"] = np.arange(len(prediction_data))
    prediction_data["Month"] = prediction_data.index.month
    prediction_data["Day_of_Week"] = prediction_data.index.dayofweek

    X = prediction_data[
        ["Day_Number", "Month", "Day_of_Week"]
    ]
    y = prediction_data["Demand"]

    # Use chronological training and testing sets.
    split_point = int(len(prediction_data) * 0.8)

    X_train = X.iloc[:split_point]
    X_test = X.iloc[split_point:]
    y_train = y.iloc[:split_point]
    y_test = y.iloc[split_point:]

    # Train and compare four regression models.
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42)
    }

    model_results = []
    predictions = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        predictions[name] = y_pred

        model_results.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "R²": r2_score(y_test, y_pred) if len(y_test) > 1 else np.nan
        })

    model_comparison = pd.DataFrame(model_results)

    st.subheader("Model Performance Comparison")

    st.dataframe(
        model_comparison.style.format({
            "MAE": "{:,.2f}",
            "RMSE": "{:,.2f}",
            "R²": "{:.4f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    # Identify the best model based on RMSE.
    best_model_name = model_comparison.loc[
        model_comparison["RMSE"].idxmin(), "Model"
    ]

    st.success(f"Best model based on RMSE: {best_model_name}")

    # Compare actual demand with Gradient Boosting predictions.
    prediction_comparison = pd.DataFrame({
        "Actual Demand": y_test,
        "Predicted Demand": predictions["Gradient Boosting"]
    }, index=y_test.index)

    st.divider()
    st.subheader("Actual vs Predicted Daily Demand")
    st.line_chart(prediction_comparison)

    # Display a concise data-driven business insight.
    highest_demand_day = daily_demand.idxmax()
    highest_demand_value = daily_demand.max()

    st.subheader("Business Insights")

    st.write(
        f"The highest observed daily demand in the selected period was "
        f"{highest_demand_value:,.0f} units on "
        f"{highest_demand_day.strftime('%d %B %Y')}."
    )

    st.write(
        f"{best_model_name} achieved the lowest RMSE among the four "
        f"models on the chronological test period. Model performance "
        f"should be interpreted in the context of the limited 2020 dataset."
    )

# --------------------------------------------------
# 10. DATASET OVERVIEW
# --------------------------------------------------

elif page == "Dataset Overview":

    st.header("Dataset Overview")
    st.caption("A preview of the filtered transaction data.")

    col1, col2, col3 = st.columns(3)

    col1.metric("Transaction Lines", f"{len(df):,}")
    col2.metric("Unique Invoices", f"{df['InvoiceNo'].nunique():,}")
    col3.metric("Unique Customers", f"{df['CustomerID'].nunique():,}")

    st.subheader("Filtered Transaction Data")
    st.dataframe(
        df.head(10),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "The table displays the first 10 rows of the data after applying "
        "the selected date range."
    )
    
    
    
    
    
    
