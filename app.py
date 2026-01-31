from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)

# =====================================================
# LOAD DATA
# =====================================================
print("Loading crop production data...")

crop_df = pd.read_csv("crop_production.csv")

crop_df = crop_df.dropna()
crop_df = crop_df[["State_Name", "Crop_Year", "Crop", "Area", "Production"]]
crop_df["Year"] = crop_df["Crop_Year"].astype(str).str[:4].astype(int)

# =====================================================
# STATE ENVIRONMENT DATA
# =====================================================
indian_state_rainfall = {
    'Andaman and Nicobar Islands': 3100, 'Andhra Pradesh': 940, 'Arunachal Pradesh': 2800,
    'Assam': 2400, 'Bihar': 1200, 'Chandigarh': 1100, 'Chhattisgarh': 1400,
    'Delhi': 800, 'Goa': 3000, 'Gujarat': 800, 'Haryana': 600,
    'Himachal Pradesh': 1500, 'Jammu and Kashmir': 1000, 'Jharkhand': 1400,
    'Karnataka': 1200, 'Kerala': 3000, 'Madhya Pradesh': 1200,
    'Maharashtra': 1200, 'Odisha': 1500, 'Punjab': 600, 'Rajasthan': 600,
    'Tamil Nadu': 1000, 'Telangana': 900, 'Uttar Pradesh': 1000,
    'Uttarakhand': 1500, 'West Bengal': 1800
}

indian_state_temperature = {
    'Andaman and Nicobar Islands': 27, 'Andhra Pradesh': 29, 'Arunachal Pradesh': 18,
    'Assam': 24, 'Bihar': 26, 'Chandigarh': 24, 'Chhattisgarh': 27,
    'Delhi': 25, 'Goa': 27, 'Gujarat': 27, 'Haryana': 24,
    'Himachal Pradesh': 16, 'Jammu and Kashmir': 14, 'Jharkhand': 25,
    'Karnataka': 25, 'Kerala': 27, 'Madhya Pradesh': 26,
    'Maharashtra': 26, 'Odisha': 27, 'Punjab': 23, 'Rajasthan': 27,
    'Tamil Nadu': 28, 'Telangana': 28, 'Uttar Pradesh': 25,
    'Uttarakhand': 20, 'West Bengal': 26
}

crop_df["ANNUAL_RAINFALL"] = crop_df["State_Name"].map(indian_state_rainfall).fillna(1170)
crop_df["AVG_TEMPERATURE"] = crop_df["State_Name"].map(indian_state_temperature).fillna(25)

# =====================================================
# CROP OPTIMAL CONDITIONS (FULL SET)
# =====================================================
crop_optimal_conditions = {
    'Rice': {'temp_min': 20, 'temp_max': 35, 'rain_min': 1000, 'rain_max': 2500},
    'Wheat': {'temp_min': 15, 'temp_max': 25, 'rain_min': 400, 'rain_max': 800},
    'Sugarcane': {'temp_min': 20, 'temp_max': 35, 'rain_min': 1500, 'rain_max': 2500},
    'Maize': {'temp_min': 21, 'temp_max': 30, 'rain_min': 500, 'rain_max': 1000},
    'Cotton': {'temp_min': 21, 'temp_max': 30, 'rain_min': 500, 'rain_max': 1000},
    'Potato': {'temp_min': 15, 'temp_max': 25, 'rain_min': 500, 'rain_max': 750},
    'Tomato': {'temp_min': 20, 'temp_max': 30, 'rain_min': 600, 'rain_max': 1250},
    'Banana': {'temp_min': 15, 'temp_max': 35, 'rain_min': 1000, 'rain_max': 2500},
    'Mango': {'temp_min': 18, 'temp_max': 40, 'rain_min': 750, 'rain_max': 2500},
    'DEFAULT': {'temp_min': 20, 'temp_max': 30, 'rain_min': 600, 'rain_max': 1500}
}

# =====================================================
# HELPER FUNCTIONS
# =====================================================
def get_crop_factor(crop, temp, rain):
    c = crop_optimal_conditions.get(crop, crop_optimal_conditions["DEFAULT"])

    temp_factor = 1 if c["temp_min"] <= temp <= c["temp_max"] else 0.8
    rain_factor = 1 if c["rain_min"] <= rain <= c["rain_max"] else 0.8

    return temp_factor * rain_factor


def yield_category(y):
    if y < 1.5:
        return "Low"
    elif y < 3.0:
        return "Medium"
    return "High"


# =====================================================
# FEATURE ENGINEERING
# =====================================================
crop_df["Base_Yield"] = crop_df["Production"] / crop_df["Area"]
crop_df["Crop_Factor"] = crop_df.apply(
    lambda r: get_crop_factor(r["Crop"], r["AVG_TEMPERATURE"], r["ANNUAL_RAINFALL"]),
    axis=1
)
crop_df["Adjusted_Yield"] = crop_df["Base_Yield"] * crop_df["Crop_Factor"]
crop_df["Yield_Category"] = crop_df["Adjusted_Yield"].apply(yield_category)

# =====================================================
# ENCODING & SCALING
# =====================================================
crop_encoder = LabelEncoder()
crop_df["Crop_Encoded"] = crop_encoder.fit_transform(crop_df["Crop"])

yield_encoder = LabelEncoder()
y = yield_encoder.fit_transform(crop_df["Yield_Category"])

scaler = StandardScaler()
X_cont = scaler.fit_transform(
    crop_df[["Area", "Production", "ANNUAL_RAINFALL", "AVG_TEMPERATURE"]]
)

X = np.hstack([crop_df[["Crop_Encoded"]].values, X_cont])

# =====================================================
# ANN MODEL
# =====================================================
model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    max_iter=1500,
    random_state=42
)

model.fit(X, y)
print("Model trained successfully")

# =====================================================
# FLASK ROUTE
# =====================================================
@app.route("/", methods=["GET", "POST"])
def index():

    result = None
    estimated_yield = None
    environmental_impact = None
    crop_ranges = None

    selected_crop = area = production = rainfall = temperature = None

    if request.method == "POST":
        selected_crop = request.form["crop"]
        area = float(request.form["area"])
        production = float(request.form["production"])
        rainfall = float(request.form["rainfall"])
        temperature = float(request.form["temperature"])

        base_yield = production / area
        factor = get_crop_factor(selected_crop, temperature, rainfall)
        estimated_yield = base_yield * factor

        # Rule-based category
        result = yield_category(estimated_yield)

        # Environmental feedback
        crop_ranges = crop_optimal_conditions.get(
            selected_crop, crop_optimal_conditions["DEFAULT"]
        )

        impacts = []
        impacts.append("✓ Optimal temperature" if crop_ranges["temp_min"] <= temperature <= crop_ranges["temp_max"]
                       else f"⚠ Temperature should be {crop_ranges['temp_min']}–{crop_ranges['temp_max']}°C")

        impacts.append("✓ Optimal rainfall" if crop_ranges["rain_min"] <= rainfall <= crop_ranges["rain_max"]
                       else f"⚠ Rainfall should be {crop_ranges['rain_min']}–{crop_ranges['rain_max']} mm")

        environmental_impact = " | ".join(impacts)

    return render_template(
        "index.html",
        crops=sorted(crop_encoder.classes_),
        result=result,
        estimated_yield=estimated_yield,
        environmental_impact=environmental_impact,
        crop_ranges=crop_ranges,
        selected_crop=selected_crop,
        area=area,
        production=production,
        rainfall=rainfall,
        temperature=temperature
    )
