# streamlit_app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import boto3
from functools import lru_cache
import io


st.set_page_config(layout="wide")
st.title("🛒 Amazon Product Price Change Analysis")

# ---------------------------------------
# 1. LOAD DATA (from CSV)
# ---------------------------------------

# @st.cache_data
# def load_data(category):
#     if category == "Mattress":
#         mattress_df = pd.read_csv("C:/Users/bande/Downloads/Mattress_Category_Mar-Apr.csv", parse_dates=["date"])
#         mattress_df.rename(columns={'ASIN': 'asin', 'price': 'price', 'date': 'date'}, inplace=True)
#         return mattress_df
#     else:
#         pet_df = pd.read_csv("C:/Users/bande/Downloads/Pet_Category_Mar-Apr.csv", parse_dates=["date"])
#         pet_df.rename(columns={'ASIN': 'asin', 'price': 'price', 'date': 'date'}, inplace=True)
#         return pet_df

@st.cache_data
def load_data_from_s3(category):
    bucket_name = "anarix-cpi"
    if category == "Mattress":
        key = "ANALYSIS/Mattress_Category_Mar-Apr.csv"
    else:
        key = "ANALYSIS/Pet_Category_Mar-Apr.csv"

    # Create S3 client
    s3 = boto3.client('s3')

    # Fetch the object
    response = s3.get_object(Bucket=bucket_name, Key=key)
    data = response["Body"].read()
    df = pd.read_csv(io.BytesIO(data), parse_dates=["date"])
    return df

# ---------------------------------------
# 2. ANALYSIS FUNCTIONS
# ---------------------------------------

def calculate_price_change(df, start, end, label, avg=False):
    df_period = df[(df["date"] >= start) & (df["date"] <= end)]

    if avg:
        start_prices = df_period[df_period["date"].between(start, pd.Timestamp("2025-03-31"))].groupby("asin")["price"].mean()
        end_prices = df_period[df_period["date"].between(pd.Timestamp("2025-04-01"), end)].groupby("asin")["price"].mean()
    else:
        start_prices = df_period.sort_values("date").groupby("asin").first()["price"]
        end_prices = df_period.sort_values("date").groupby("asin").last()["price"]

    df_change = pd.DataFrame({
        "start_price": start_prices,
        "end_price": end_prices
    }).dropna()

    df_change["abs_change"] = df_change["end_price"] - df_change["start_price"]
    df_change["pct_change"] = ((df_change["end_price"] - df_change["start_price"]) / df_change["start_price"]) * 100
    df_change["Period"] = label

    def bucket(pct):
        if pct >= 30: return ">=30% Increase"
        elif pct >= 20: return "20–30% Increase"
        elif pct >= 10: return "10–20% Increase"
        elif pct > 0: return "0–10% Increase"
        elif pct == 0: return "No Change"
        elif pct <= -30: return ">=30% Decrease"
        elif pct <= -20: return "20–30% Decrease"
        elif pct <= -10: return "10–20% Decrease"
        else: return "0–10% Decrease"

    df_change["change_bucket"] = df_change["pct_change"].apply(bucket)
    df_change.reset_index(inplace=True)

    return df_change

# ---------------------------------------
# 3. APP UI
# ---------------------------------------

category = st.selectbox("📂 Select Category", ["Mattress", "Pet Supplies"])

df = load_data_from_s3(category)

# Dates
march_start = pd.to_datetime("2025-03-10")
march_end = pd.to_datetime("2025-03-31")
april_start = pd.to_datetime("2025-04-01")
april_end = pd.to_datetime("2025-04-08")

# Calculate
march = calculate_price_change(df, march_start, march_end, "March")
april = calculate_price_change(df, april_start, april_end, "April")
average = calculate_price_change(df, march_start, april_end, "Average", avg=True)

combined = pd.concat([march, april, average])

# ---------------------------------------
# 4. CHARTS
# ---------------------------------------

# col1, col2 = st.columns([2, 3])

# with col1:
#     st.subheader("📊 Price Change Distribution")
st.subheader("📊 Price Change Distribution")
fig_hist = px.histogram(combined,
                        x="change_bucket",
                        color="Period",
                        # color_discrete_map={
                        #     "March": "blue",
                        #     "April": "orange",
                        #     "Average": "green"
                        # },
                        barmode="group",
                        category_orders={"change_bucket": [
                            ">=30% Decrease", "20–30% Decrease", "10–20% Decrease", "0–10% Decrease",
                            "No Change",
                            "0–10% Increase", "10–20% Increase", "20–30% Increase", ">=30% Increase"
                        ]})
st.plotly_chart(fig_hist, use_container_width=True)

# with col2:
st.subheader("📈 Daily Average Price Trend")
trend_df = df[(df["date"] >= march_start) & (df["date"] <= april_end)]
daily_avg = trend_df.groupby("date")["price"].mean().reset_index()
fig_line = px.line(daily_avg, x="date", y="price", title="Average Price Per Day")
st.plotly_chart(fig_line, use_container_width=True)


# st.plotly_chart(fig_hist, use_container_width=True)

# st.subheader("📈 Daily Average Price Trend")
# st.plotly_chart(fig_line, use_container_width=True)


# ---------------------------------------
# 5. TABLE VIEW
# ---------------------------------------

st.subheader("📋 ASIN-wise Price Change Summary")

# Download button
csv = combined.to_csv(index=False)
st.download_button("📥 Download as CSV", csv, file_name="price_changes.csv")

# Table
st.dataframe(combined, use_container_width=True, height=500)
# ({
#    "start_price": "{:.2f}",
    # "end_price": "{:.2f}",
    # "abs_change": "{:.2f}",
    # "pct_change": "{:.2f}%"
#}),