from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# ============================================================
# LOAD DATA
# ============================================================
print("🌾 Loading crop production data...")

try:
    crop_df = pd.read_csv("crop_production.csv")
    print(f"✅ Loaded {len(crop_df)} rows")
except Exception as e:
    print(f"❌ Error loading CSV: {e}")
    raise

# ============================================================
# BASIC CLEANING (CRITICAL FOR DEPLOYMENT)
# ============================================================
required_cols = ["State_Name", "Crop_Year", "Crop", "Area", "Production"]
crop_df = crop_df[required_cols]

# Convert numeric columns safely
crop_df["Area"] = pd.to_numeric(crop_df["Area"], errors="coerce")
crop_df["Production"] = pd.to_numeric(crop_df["Production"], errors="coerce")

# Drop invalid rows
crop_df.replace([np.inf, -np.inf], np.nan, inplace=True)
crop_df.dropna(inplace=True)

crop_df["Year"] = crop_df["Crop_Year"].astype(str).str[:4].astype(int)

print(f"✅ Clean dataset rows: {len(crop_df)}")

# ============================================================
# SYNTHETIC RAINFALL & TEMPERATURE
# ============================================================
indian_state_rainfall = {
    'Andaman and Nicobar Islands': 3100, 'Andhra Pradesh': 940, 'Arunachal Pradesh': 2800,
    'Assam': 2400, 'Bihar': 1200, 'Chandigarh': 1100, 'Chhattisgarh': 1400,
    'Dadra and Nagar Haveli': 2000, 'Daman and Diu': 2000, 'Delhi': 800,
    'Goa': 3000, 'Gujarat': 800, 'Haryana': 600, 'Himachal Pradesh': 1500,
    'Jammu and Kashmir': 1000, 'Jharkhand': 1400, 'Karnataka': 1200,
    'Kerala': 3000, 'Madhya Pradesh': 1200, 'Maharashtra': 1200,
    'Manipur': 1800, 'Meghalaya': 2500, 'Mizoram': 2500, 'Nagaland': 2000,
    'Odisha': 1500, 'Puducherry': 1200, 'Punjab': 600, 'Rajasthan': 600,
    'Sikkim': 2500, 'Tamil Nadu': 1000, 'Telangana': 900, 'Tripura': 2400,
    'Uttar Pradesh': 1000, 'Uttarakhand': 1500, 'West Bengal': 1800
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

# Add controlled noise
np.random.seed(42)
crop_df["ANNUAL_RAINFALL"] *= np.random.uniform(0.9, 1.1, len(crop_df))
crop_df["AVG_TEMPERATURE"] += np.random.uniform(-2, 2, len(crop_df))

# ============================================================
# CROP OPTIMAL CONDITIONS
# ============================================================
crop_optimal_conditions = {
    'Rice': {'temp_min': 20, 'temp_max': 35, 'rain_min': 1000, 'rain_max': 2500},
    'Wheat': {'temp_min': 15, 'temp_max': 25, 'rain_min': 400, 'rain_max': 800},
    'Sugarcane': {'temp_min': 20, 'temp_max': 35, 'rain_min': 1500, 'rain_max': 2500},
    'Maize': {'temp_min': 21, 'temp_max': 30, 'rain_min': 500, 'rain_max': 1000},
    'DEFAULT': {'temp_min': 20, 'temp_max': 30, 'rain_min': 600, 'rain_max': 1500}
}

def get_crop_factor(crop, temp, rain):
    c = crop_optimal_conditions.get(crop, crop_optimal_conditions["DEFAULT"])

    temp_factor = 1.0 if c["temp_min"] <= temp <= c["temp_max"] else 0.85
    rain_factor = 1.0 if c["rain_min"] <= rain <= c["rain_max"] else 0.85

    return temp_factor * rain_factor

# ============================================================
# FEATURE ENGINEERING
# ============================================================
crop_df["Base_Yield"] = crop_df["Production"] / crop_df["Area"]
crop_df["Crop_Factor"] = crop_df.apply(
    lambda r: get_crop_factor(r["Crop"], r["AVG_TEMPERATURE"], r["ANNUAL_RAINFALL"]), axis=1
)
crop_df["Adjusted_Yield"] = crop_df["Base_Yield"] * crop_df["Crop_Factor"]

def yield_category(y):
    if y < 1.5:
        return "Low"
    elif y < 3.0:
        return "Medium"
    return "High"

crop_df["Yield_Category"] = crop_df["Adjusted_Yield"].apply(yield_category)

# ============================================================
# ENCODING & SCALING
# ============================================================
crop_encoder = LabelEncoder()
crop_df["Crop_Encoded"] = crop_encoder.fit_transform(crop_df["Crop"])

yield_encoder = LabelEncoder()
y = yield_encoder.fit_transform(crop_df["Yield_Category"])

scaler = StandardScaler()
X_cont = scaler.fit_transform(
    crop_df[["Area", "Production", "ANNUAL_RAINFALL", "AVG_TEMPERATURE"]]
)

X = np.hstack([crop_df[["Crop_Encoded"]].values, X_cont])

# ============================================================
# MODEL (SAFE SETTINGS)
# ============================================================
model = MLPClassifier(
    hidden_layer_sizes=(32, 16),
    max_iter=800,
    random_state=42
)

model.fit(X, y)
print(f"✅ Model trained successfully ({model.score(X, y):.3f})")

# ============================================================
# FLASK ROUTE
# ============================================================
@app.route("/", methods=["GET", "POST"])
def index():
    result = estimated_yield = yield_efficiency = None
    environmental_impact = None

    if request.method == "POST":
        crop = request.form["crop"]
        area = float(request.form["area"])
        production = float(request.form["production"])
        rainfall = float(request.form["rainfall"])
        temperature = float(request.form["temperature"])

        base_yield = production / area
        factor = get_crop_factor(crop, temperature, rainfall)
        adjusted = base_yield * factor

        yield_efficiency = factor * 100
        estimated_yield = round(adjusted, 2)

        crop_encoded = crop_encoder.transform([crop])[0]
        scaled = scaler.transform([[area, production, rainfall, temperature]])
        pred = model.predict([[crop_encoded, *scaled[0]]])
        result = yield_encoder.inverse_transform(pred)[0]

        environmental_impact = (
            "✓ Conditions acceptable" if factor > 0.9 else "⚠ Sub-optimal conditions"
        )

    return render_template(
        "index.html",
        crops=sorted(crop_encoder.classes_),
        result=result,
        estimated_yield=estimated_yield,
        yield_efficiency=yield_efficiency,
        environmental_impact=environmental_impact
    )
