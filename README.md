🚀 Amazon Delivery Time Prediction

This project predicts the delivery time of Amazon orders based on multiple factors such as order time, traffic, weather, vehicle type, and delivery distance. The model uses Random Forest Regressor and is optimized for structured output and fast inference.


---

📌 Features

Predict Delivery Time based on real-world parameters

Streamlit UI for interactive user input and analysis

Exploratory Data Analysis (EDA) to visualize trends

Optimized Machine Learning Model with caching for fast results

Geospatial Distance Calculation for accurate distance estimation

MLflow Integration for experiment tracking



---

🛠️ Tech Stack

Python 🐍

Pandas, NumPy, Scikit-learn (Data Processing & ML)

Geopy (Geospatial calculations)

Streamlit (Web UI)

Seaborn, Matplotlib (Data Visualization)

MLflow (Model tracking & logging)

📊 Data Preprocessing

Missing values are handled using median & mode imputation

Categorical variables (e.g., weather, traffic) are label-encoded

Geospatial distance is calculated using Haversine formula

Feature scaling is applied using StandardScaler



---

🤖 Model Training

Algorithm: Random Forest Regressor

Optimized Hyperparameters:

n_estimators=200, max_depth=15, min_samples_split=5


Performance Metrics:

✅ MAE (Mean Absolute Error)

✅ RMSE (Root Mean Square Error)

✅ R² Score (Coefficient of Determination)




---

📌 Features in Streamlit UI

✔ Sidebar Inputs – Users can enter order details & predict delivery time
✔ "Predict Delivery Time" Button – Moved left near sidebar for easy access
✔ EDA Tabs – Visualize missing values, feature distributions, correlation heatmaps
✔ Faster Predictions – No unnecessary reloading, model caching enabled


---

🛠️ Future Enhancements

🔄 Model API Integration for real-time delivery predictions

🚀 Deep Learning Model for higher accuracy

🌎 Live Traffic & Weather Data integration

📊 More Advanced Visualizations
