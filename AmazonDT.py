import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopy.distance
import mlflow
import mlflow.sklearn
import streamlit as st
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score

# ✅ Streamlit Page Config
st.set_page_config(layout="wide")

# --- Load dataset ---
df = pd.read_csv("AmazonDT_Dataset.csv").drop_duplicates()

# Handle missing values
df.fillna(df.median(numeric_only=True), inplace=True)
df.fillna(df.mode().iloc[0], inplace=True)

# Convert Order_Date to datetime and extract features
df["Order_Date"] = pd.to_datetime(df["Order_Date"], dayfirst=True)
df["Order_Year"] = df["Order_Date"].dt.year
df["Order_Month"] = df["Order_Date"].dt.month
df["Order_Day"] = df["Order_Date"].dt.day

# Convert Order_Time and Pickup_Time to datetime
df["Order_Time"] = pd.to_datetime(df["Order_Time"], format="%H:%M:%S").dt.hour
df["Pickup_Time"] = pd.to_datetime(df["Pickup_Time"], format="%H:%M:%S").dt.hour

# ✅ Ensure Distance is correctly calculated
def calculate_distance(row):
    coords_1 = (row["Store_Latitude"], row["Store_Longitude"])
    coords_2 = (row["Drop_Latitude"], row["Drop_Longitude"])
    return geopy.distance.geodesic(coords_1, coords_2).km

df["Distance_KM"] = df.apply(calculate_distance, axis=1)

# Encode categorical variables
label_encoders = {}
categorical_features = ["Weather", "Traffic", "Vehicle", "Area", "Category"]

for col in categorical_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Select features and target variable
X = df.drop(columns=["Order_ID", "Order_Date", "Delivery_Time"])
y = df["Delivery_Time"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Use MinMaxScaler to ensure distance impact
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# ✅ Load or train the model
try:
    with open("best_model.pkl", "rb") as f:
        best_model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    best_model = RandomForestRegressor(n_estimators=100, random_state=42)
    best_model.fit(X_train, y_train)
    with open("best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

# --- Streamlit UI ---

st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg", use_container_width=True)
st.sidebar.markdown("<h2 style='text-align: center;'>User Input Features</h2>", unsafe_allow_html=True)

agent_age = st.sidebar.number_input("Agent Age", min_value=18, max_value=70, value=30)
agent_rating = st.sidebar.slider("Agent Rating", 1.0, 5.0, 4.5)
order_year = st.sidebar.selectbox("Order Year", sorted(df["Order_Year"].unique()))
order_month = st.sidebar.selectbox("Order Month", sorted(df["Order_Month"].unique()))
order_day = st.sidebar.selectbox("Order Day", sorted(df["Order_Day"].unique()))
order_hour = st.sidebar.slider("Order Time (Hour)", 0, 23, 12)
pickup_hour = st.sidebar.slider("Pickup Time (Hour)", 0, 23, 14)
distance = st.sidebar.number_input("Distance (KM)", min_value=0.1, max_value=50.0, value=5.0)

weather = st.sidebar.selectbox("Weather", label_encoders["Weather"].classes_)
traffic = st.sidebar.selectbox("Traffic", label_encoders["Traffic"].classes_)
vehicle = st.sidebar.selectbox("Vehicle", label_encoders["Vehicle"].classes_)
area = st.sidebar.selectbox("Area", label_encoders["Area"].classes_)
category = st.sidebar.selectbox("Category", label_encoders["Category"].classes_)

# Encode inputs
encoded_weather = label_encoders["Weather"].transform([weather])[0]
encoded_traffic = label_encoders["Traffic"].transform([traffic])[0]
encoded_vehicle = label_encoders["Vehicle"].transform([vehicle])[0]
encoded_area = label_encoders["Area"].transform([area])[0]
encoded_category = label_encoders["Category"].transform([category])[0]

features = [agent_age, agent_rating, order_year, order_month, order_day, order_hour, pickup_hour, distance, 
            encoded_weather, encoded_traffic, encoded_vehicle, encoded_area, encoded_category, 0, 0, 0, 0]

# --- Instant Prediction ---
def predict(features):
    features_scaled = scaler.transform([features])
    return best_model.predict(features_scaled)[0]

# ✅ Top-right predict button
col1, col2 = st.columns([3, 1])
with col2:
    if st.button("🚀 Predict Delivery Time"):
        prediction = predict(features)
        st.success(f"📌 Estimated Delivery Time: {round(prediction, 2)} minutes")

# --- EDA Section ---
st.header("📊 Exploratory Data Analysis")

eda_tabs = st.tabs(["Missing Values", "Feature Distributions", "Outliers", "Correlation Heatmap", "Delivery Time vs Distance"])

with eda_tabs[0]:  
    st.write("### 🔍 Missing Values")
    st.write(df.isnull().sum())

with eda_tabs[1]:  
    st.write("### 📊 Feature Distributions")
    fig, ax = plt.subplots(figsize=(8,6))
    sns.histplot(df["Delivery_Time"], bins=30, kde=True, ax=ax)
    st.pyplot(fig)

with eda_tabs[2]:  
    st.write("### ⚠️ Outliers in Delivery Time")
    fig, ax = plt.subplots(figsize=(8,6))
    sns.boxplot(x=df["Delivery_Time"], ax=ax)
    st.pyplot(fig)

with eda_tabs[3]:  
    st.write("### 🔥 Correlation Heatmap")
    df_numeric = df.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(df_numeric.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

with eda_tabs[4]:  
    st.write("### 🚗 Delivery Time vs Distance")
    fig, ax = plt.subplots(figsize=(8,6))
    sns.scatterplot(x=df["Distance_KM"], y=df["Delivery_Time"], alpha=0.5, ax=ax)
    st.pyplot(fig)
