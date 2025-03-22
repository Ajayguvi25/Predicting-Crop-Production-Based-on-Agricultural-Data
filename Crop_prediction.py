import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("C:/Users/AJITH HARISH/Downloads/cleaned_crop_data.csv")
    df['Productivity'] = df['Yield'] / df['Area_Harvested']
    return df

df = load_data()

# Handle Missing Values
imputer = SimpleImputer(strategy="mean")  # Fill NaN with Mean values
df[['Area_Harvested', 'Yield']] = imputer.fit_transform(df[['Area_Harvested', 'Yield']])
df.dropna(inplace=True)  # Drop rows with NaN in other columns

# Train-Test Split
X = df[['Area_Harvested', 'Year']]
y = df['Yield']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Models
lin_reg = LinearRegression()
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)

lin_reg.fit(X_train, y_train)
rf_reg.fit(X_train, y_train)

# Predictions
y_pred_lin = lin_reg.predict(X_test)
y_pred_rf = rf_reg.predict(X_test)

# Model Evaluation
lin_mse = mean_squared_error(y_test, y_pred_lin)
rf_mse = mean_squared_error(y_test, y_pred_rf)
lin_r2 = r2_score(y_test, y_pred_lin)
rf_r2 = r2_score(y_test, y_pred_rf)

# Streamlit UI
st.title("🌾 Crop Production Prediction & Analysis")
st.sidebar.header("🔎 Filter Options")

# Sidebar Filters
year_selected = st.sidebar.selectbox("📅 Select Year", df["Year"].unique())
region_selected = st.sidebar.selectbox("🌍 Select Region", df["Area"].unique())
crops_selected = st.sidebar.multiselect("🌿 Select Crop Types", df["Item"].unique(), default=df["Item"].unique()[:3])
yield_range = st.sidebar.slider("📏 Select Yield Range (kg/ha)", int(df["Yield"].min()), int(df["Yield"].max()), (int(df["Yield"].min()), int(df["Yield"].max())))

# Filter Data
df_filtered = df[
    (df["Year"] == year_selected) & 
    (df["Area"] == region_selected) & 
    (df["Item"].isin(crops_selected)) & 
    (df["Yield"].between(yield_range[0], yield_range[1]))
]

# Summary Statistics
st.write("## 📊 Summary Statistics")
col1, col2, col3 = st.columns(3)
col1.metric("🌾 Total Area Harvested", f"{df_filtered['Area_Harvested'].sum():,.0f} Ha")
col2.metric("📈 Average Yield", f"{df_filtered['Yield'].mean():,.2f} kg/ha")
col3.metric("🏆 Max Yield", f"{df_filtered['Yield'].max():,.2f} kg/ha")

# Data Visualization
st.write("## 📊 Crop Yield Distribution")
fig1 = px.histogram(df_filtered, x="Yield", color="Item", title="Yield Distribution Across Crops", nbins=20)
st.plotly_chart(fig1, use_container_width=True)

st.write("## 🌍 Region-Wise Crop Production")
fig2 = px.bar(df_filtered, x="Item", y="Production", color="Item", title="Crop Production by Region")
st.plotly_chart(fig2, use_container_width=True)

st.write("## 📈 Yearly Trends in Crop Yield")
fig3 = px.line(df, x="Year", y="Yield", color="Item", title="Crop Yield Over the Years")
st.plotly_chart(fig3, use_container_width=True)

# Model Performance
st.write("## 📊 Model Performance")
st.write(f"📌 **Linear Regression:** R² = {lin_r2:.2f}, MSE = {lin_mse:.2f}")
st.write(f"📌 **Random Forest:** R² = {rf_r2:.2f}, MSE = {rf_mse:.2f}")

# Prediction Input
st.write("## 🔮 Predict Crop Yield")
input_area = st.number_input("Enter Area Harvested (ha)", min_value=0.0, step=0.1)
input_year = st.number_input("Enter Year", min_value=int(df['Year'].min()), max_value=int(df['Year'].max()), step=1)
if st.button("Predict Yield"):
    pred_lin = lin_reg.predict([[input_area, input_year]])[0]
    pred_rf = rf_reg.predict([[input_area, input_year]])[0]
    st.success(f"📌 **Linear Regression Prediction:** {pred_lin:.2f} kg/ha")
    st.success(f"📌 **Random Forest Prediction:** {pred_rf:.2f} kg/ha")

# Download Filtered Data
st.write("## 📥 Download Filtered Data")
csv = df_filtered.to_csv(index=False).encode('utf-8')
st.download_button(label="Download CSV", data=csv, file_name="filtered_crop_data.csv", mime="text/csv")
