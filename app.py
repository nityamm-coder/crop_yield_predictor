from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# =========================
# LOAD DATA
# =========================
crop_df = pd.read_csv("crop_production.csv")
crop_df = crop_df.dropna()
crop_df = crop_df[["State_Name", "Crop_Year", "Crop", "Area", "Production"]]
crop_df["Year"] = crop_df["Crop_Year"].astype(str).str[:4].astype(int)

# =========================
# CROP CONDITIONS
# =========================
crop_optimal_conditions = {
    'Rice': {'temp_min': 20, 'temp_max': 35, 'rain_min': 1000, 'rain_max': 2500},
    'Wheat': {'temp_min': 15, 'temp_max': 25, 'rain_min': 400, 'rain_max': 800},
    'Sugarcane': {'temp_min': 20, 'temp_max': 35, 'rain_min': 1500, 'rain_max': 2500},
    'Maize': {'temp_min': 21, 'temp_max': 30, 'rain_min': 500, 'rain_max': 1000},
    'Cotton': {'temp_min': 21, 'temp_max': 30, 'rain_min': 500, 'rain_max': 1000},
    'DEFAULT': {'temp_min': 20, 'temp_max': 30, 'rain_min': 600, 'rain_max': 1500}
}

def get_crop_factor(crop, temperature, rainfall):
    c = crop_optimal_conditions.get(crop, crop_optimal_conditions['DEFAULT'])
    temp_factor = 1.0 if c['temp_min'] <= temperature <= c['temp_max'] else 0.8
    rain_factor = 1.0 if c['rain_min'] <= rainfall <= c['rain_max'] else 0.85
    return temp_factor * rain_factor

# =========================
# MODEL PREP
# =========================
crop_encoder = LabelEncoder()
crop_df["Crop_Encoded"] = crop_encoder.fit_transform(crop_df["Crop"])

crop_df["Base_Yield"] = crop_df["Production"] / crop_df["Area"]
crop_df["Yield_Category"] = pd.cut(
    crop_df["Base_Yield"], bins=[0, 1.5, 3.0, 100], labels=["Low", "Medium", "High"]
)

yield_encoder = LabelEncoder()
y = yield_encoder.fit_transform(crop_df["Yield_Category"])

scaler = StandardScaler()
X_cont = scaler.fit_transform(crop_df[["Area", "Production"]])
X = np.hstack([crop_df[["Crop_Encoded"]].values, X_cont])

model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
model.fit(X, y)

# =========================
# ROUTE
# =========================
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    estimated_yield = None
    environmental_impact = None
    crop_info = None

    # Preserve inputs
    selected_crop = area = production = rainfall = temperature = None

    if request.method == "POST":
        selected_crop = request.form.get("crop")
        area = float(request.form.get("area"))
        production = float(request.form.get("production"))
        rainfall = float(request.form.get("rainfall"))
        temperature = float(request.form.get("temperature"))

        base_yield = production / area
        factor = get_crop_factor(selected_crop, temperature, rainfall)
        estimated_yield = base_yield * factor

        result = (
            "Low" if estimated_yield < 1.5 else
            "Medium" if estimated_yield < 3 else
            "High"
        )

        cond = crop_optimal_conditions.get(selected_crop, crop_optimal_conditions['DEFAULT'])
        crop_info = cond

        impacts = []
        impacts.append(
            f"Temperature range: {cond['temp_min']}°C – {cond['temp_max']}°C"
        )
        impacts.append(
            f"Rainfall range: {cond['rain_min']} – {cond['rain_max']} mm"
        )

        environmental_impact = " | ".join(impacts)

    return render_template(
        "index.html",
        crops=sorted(crop_encoder.classes_),
        result=result,
        estimated_yield=estimated_yield,
        environmental_impact=environmental_impact,
        crop_info=crop_info,
        selected_crop=selected_crop,
        area=area,
        production=production,
        rainfall=rainfall,
        temperature=temperature
    )
