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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("AmazonDT_Dataset.csv")

# Data Preprocessing
df.drop_duplicates(inplace=True)

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

# Calculate geospatial distance between store and drop location
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

# Ensure all required features are included
X = X[["Agent_Age", "Agent_Rating", "Order_Year", "Order_Month", "Order_Day", "Order_Time", "Pickup_Time", "Distance_KM",
       "Weather", "Traffic", "Vehicle", "Area", "Category", "Store_Latitude", "Store_Longitude", "Drop_Latitude", "Drop_Longitude"]]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Model Training
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
}

mlflow.set_experiment("Amazon_Delivery_Prediction")

best_model = None
best_r2 = -1

for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        # Log metrics to MLflow
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)
        mlflow.sklearn.log_model(model, model_name, input_example=X_test[:1])

        # Select best model
        if r2 > best_r2:
            best_model = model
            best_r2 = r2

# Save the best model
with open("best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

# Streamlit App
def predict_delivery_time(features):
    """Load trained model and scaler, apply transformations, and predict"""
    with open("best_model.pkl", "rb") as f:
        best_model = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # Convert features to numpy array and reshape
    features = np.array(features).reshape(1, -1)

    # Apply feature scaling
    features_scaled = scaler.transform(features)

    # Predict delivery time
    return best_model.predict(features_scaled)[0]

st.set_page_config(layout="wide")

# Add Amazon Logo to Sidebar
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg", use_column_width=True)
st.sidebar.markdown("<h2 style='text-align: center;'>User Input Features</h2>", unsafe_allow_html=True)

# --- EDA Section ---
st.header("📊 Exploratory Data Analysis")

# Arrange EDA into sections
eda_tabs = st.tabs([
    "Missing Values", "Feature Distributions", "Outliers", "Correlation Heatmap", "Delivery Time vs Distance",
    "Agent Age Distribution", "Order Hour Distribution", "Pickup vs Delivery Time", "Weather vs Delivery Time", "Traffic vs Delivery Time"
])

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

with eda_tabs[5]:  
    st.write("### 🧓 Agent Age Distribution")
    fig, ax = plt.subplots(figsize=(8,6))
    sns.histplot(df["Agent_Age"], bins=20, kde=True, ax=ax)
    st.pyplot(fig)

with eda_tabs[6]:  
    st.write("### ⏰ Order Hour Distribution")
    fig, ax = plt.subplots(figsize=(8,6))
    sns.histplot(df["Order_Time"], bins=24, kde=True, ax=ax)
    st.pyplot(fig)

with eda_tabs[7]:  
    st.write("### 🚚 Pickup vs Delivery Time")
    fig, ax = plt.subplots(figsize=(8,6))
    sns.scatterplot(x=df["Pickup_Time"], y=df["Delivery_Time"], ax=ax)
    st.pyplot(fig)

with eda_tabs[8]:  
    st.write("### ☁️ Weather vs Delivery Time")
    fig, ax = plt.subplots(figsize=(8,6))
    sns.boxplot(x=df["Weather"], y=df["Delivery_Time"], ax=ax)
    st.pyplot(fig)

with eda_tabs[9]:  
    st.write("### 🚦 Traffic vs Delivery Time")
    fig, ax = plt.subplots(figsize=(8,6))
    sns.boxplot(x=df["Traffic"], y=df["Delivery_Time"], ax=ax)
    st.pyplot(fig)