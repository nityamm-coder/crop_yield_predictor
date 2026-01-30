from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ==============================
# LOAD CROP PRODUCTION DATA
# ==============================
print("Loading crop production data...")
crop_df = pd.read_csv("crop_production.csv")
crop_df = crop_df.dropna()
crop_df = crop_df[["State_Name", "Crop_Year", "Crop", "Area", "Production"]]
crop_df["Year"] = crop_df["Crop_Year"].astype(str).str[:4].astype(int)

# ==============================
# CREATE SYNTHETIC RAINFALL DATA FOR INDIAN STATES
# ==============================
indian_state_rainfall = {
    'Andaman and Nicobar Islands': 3100,
    'Andhra Pradesh': 940,
    'Arunachal Pradesh': 2800,
    'Assam': 2400,
    'Bihar': 1200,
    'Chandigarh': 1100,
    'Chhattisgarh': 1400,
    'Dadra and Nagar Haveli': 2000,
    'Daman and Diu': 2000,
    'Delhi': 800,
    'Goa': 3000,
    'Gujarat': 800,
    'Haryana': 600,
    'Himachal Pradesh': 1500,
    'Jammu and Kashmir': 1000,
    'Jharkhand': 1400,
    'Karnataka': 1200,
    'Kerala': 3000,
    'Madhya Pradesh': 1200,
    'Maharashtra': 1200,
    'Manipur': 1800,
    'Meghalaya': 2500,
    'Mizoram': 2500,
    'Nagaland': 2000,
    'Odisha': 1500,
    'Puducherry': 1200,
    'Punjab': 600,
    'Rajasthan': 600,
    'Sikkim': 2500,
    'Tamil Nadu': 1000,
    'Telangana': 900,
    'Tripura': 2400,
    'Uttar Pradesh': 1000,
    'Uttarakhand': 1500,
    'West Bengal': 1800
}

# ==============================
# CREATE SYNTHETIC TEMPERATURE DATA FOR INDIAN STATES
# ==============================
indian_state_temperature = {
    'Andaman and Nicobar Islands': 27,
    'Andhra Pradesh': 29,
    'Arunachal Pradesh': 18,
    'Assam': 24,
    'Bihar': 26,
    'Chandigarh': 24,
    'Chhattisgarh': 27,
    'Dadra and Nagar Haveli': 28,
    'Daman and Diu': 28,
    'Delhi': 25,
    'Goa': 27,
    'Gujarat': 27,
    'Haryana': 24,
    'Himachal Pradesh': 16,
    'Jammu and Kashmir': 14,
    'Jharkhand': 25,
    'Karnataka': 25,
    'Kerala': 27,
    'Madhya Pradesh': 26,
    'Maharashtra': 26,
    'Manipur': 21,
    'Meghalaya': 20,
    'Mizoram': 21,
    'Nagaland': 21,
    'Odisha': 27,
    'Puducherry': 28,
    'Punjab': 23,
    'Rajasthan': 27,
    'Sikkim': 15,
    'Tamil Nadu': 28,
    'Telangana': 28,
    'Tripura': 24,
    'Uttar Pradesh': 25,
    'Uttarakhand': 20,
    'West Bengal': 26
}

# Add rainfall and temperature to crop data
crop_df['ANNUAL_RAINFALL'] = crop_df['State_Name'].map(indian_state_rainfall)
crop_df['AVG_TEMPERATURE'] = crop_df['State_Name'].map(indian_state_temperature)

# For any states not in our mapping, use national average
crop_df['ANNUAL_RAINFALL'].fillna(1170, inplace=True)
crop_df['AVG_TEMPERATURE'].fillna(25, inplace=True)

# Add some variation (±10% for rainfall, ±2°C for temperature)
np.random.seed(42)
crop_df['ANNUAL_RAINFALL'] = crop_df['ANNUAL_RAINFALL'] * np.random.uniform(0.9, 1.1, size=len(crop_df))
crop_df['AVG_TEMPERATURE'] = crop_df['AVG_TEMPERATURE'] + np.random.uniform(-2, 2, size=len(crop_df))

print(f"Crop data with rainfall and temperature: {crop_df.shape}")

# ==============================
# CROP-SPECIFIC OPTIMAL CONDITIONS
# ==============================
crop_optimal_conditions = {
    'Rice': {'temp_min': 20, 'temp_max': 35, 'rain_min': 1000, 'rain_max': 2500},
    'Wheat': {'temp_min': 15, 'temp_max': 25, 'rain_min': 400, 'rain_max': 800},
    'Sugarcane': {'temp_min': 20, 'temp_max': 35, 'rain_min': 1500, 'rain_max': 2500},
    'Cotton': {'temp_min': 21, 'temp_max': 30, 'rain_min': 500, 'rain_max': 1000},
    'Maize': {'temp_min': 21, 'temp_max': 30, 'rain_min': 500, 'rain_max': 1000},
    'Jowar': {'temp_min': 26, 'temp_max': 33, 'rain_min': 400, 'rain_max': 600},
    'Bajra': {'temp_min': 25, 'temp_max': 35, 'rain_min': 400, 'rain_max': 600},
    'DEFAULT': {'temp_min': 20, 'temp_max': 30, 'rain_min': 600, 'rain_max': 1500}
}

def get_crop_factor(crop, temperature, rainfall):
    """Calculate crop-specific adjustment factor based on conditions"""
    conditions = crop_optimal_conditions.get(crop, crop_optimal_conditions['DEFAULT'])
    
    # Temperature factor
    if conditions['temp_min'] <= temperature <= conditions['temp_max']:
        temp_factor = 1.0
    else:
        if temperature < conditions['temp_min']:
            temp_diff = conditions['temp_min'] - temperature
            temp_factor = max(0.7, 1.0 - (temp_diff * 0.05))
        else:
            temp_diff = temperature - conditions['temp_max']
            temp_factor = max(0.7, 1.0 - (temp_diff * 0.05))
    
    # Rainfall factor
    if conditions['rain_min'] <= rainfall <= conditions['rain_max']:
        rain_factor = 1.0
    else:
        if rainfall < conditions['rain_min']:
            rain_diff = conditions['rain_min'] - rainfall
            rain_factor = max(0.7, 1.0 - (rain_diff / conditions['rain_min'] * 0.3))
        else:
            rain_diff = rainfall - conditions['rain_max']
            rain_factor = max(0.7, 1.0 - (rain_diff / conditions['rain_max'] * 0.3))
    
    return temp_factor * rain_factor

# ==============================
# CALCULATE YIELD WITH RAINFALL, TEMPERATURE & CROP INFLUENCE
# ==============================
crop_df["Base_Yield"] = crop_df["Production"] / crop_df["Area"]

# Apply crop-specific factors
crop_df['Crop_Factor'] = crop_df.apply(
    lambda row: get_crop_factor(row['Crop'], row['AVG_TEMPERATURE'], row['ANNUAL_RAINFALL']),
    axis=1
)

crop_df['Adjusted_Yield'] = crop_df['Base_Yield'] * crop_df['Crop_Factor']

# Use adjusted yield for categorization
def yield_category(y):
    if y < 1.5:
        return "Low"
    elif y < 3.0:
        return "Medium"
    else:
        return "High"

crop_df["Yield_Category"] = crop_df["Adjusted_Yield"].apply(yield_category)

print(f"Yield categories distribution:")
print(crop_df["Yield_Category"].value_counts())

# ==============================
# ENCODING
# ==============================
crop_encoder = LabelEncoder()
crop_df["Crop_Encoded"] = crop_encoder.fit_transform(crop_df["Crop"])

yield_encoder = LabelEncoder()
y_encoded = yield_encoder.fit_transform(crop_df["Yield_Category"])

# ==============================
# FEATURE SCALING (Only for continuous features, NOT crop encoding)
# ==============================
# Scale only Area, Production, Rainfall, Temperature
# Keep Crop_Encoded separate to maintain crop distinctions
scaler = StandardScaler()

X_continuous = crop_df[["Area", "Production", "ANNUAL_RAINFALL", "AVG_TEMPERATURE"]]
X_continuous_scaled = scaler.fit_transform(X_continuous)

# Combine scaled continuous features with crop encoding
X_crop = crop_df[["Crop_Encoded"]].values
X_combined = np.hstack([X_crop, X_continuous_scaled])

y = y_encoded

print(f"\nTraining model with {len(X_combined)} samples...")
print(f"Features: Crop (encoded) + Area + Production + Rainfall + Temperature")

# ==============================
# ANN MODEL
# ==============================
model = MLPClassifier(
    hidden_layer_sizes=(64, 32, 16),  # Deeper network for more complex patterns
    activation="relu",
    max_iter=2000,
    random_state=42,
    learning_rate='adaptive',
    early_stopping=True,
    validation_fraction=0.1
)

model.fit(X_combined, y)
print("Model trained successfully!")
print(f"Model score: {model.score(X_combined, y):.4f}")
print(f"Available crops: {len(crop_encoder.classes_)}")

# ==============================
# FLASK ROUTE
# ==============================
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    estimated_yield = None
    adjusted_yield = None
    selected_crop = None
    area = production = rainfall = temperature = None
    environmental_impact = None

    if request.method == "POST":
        try:
            selected_crop = request.form["crop"]
            area = float(request.form["area"])
            production = float(request.form["production"])
            rainfall = float(request.form["rainfall"])
            temperature = float(request.form["temperature"])

            if area <= 0 or production < 0 or rainfall < 0:
                result = "Error: Invalid input values"
            else:
                # Calculate base yield
                base_yield = production / area
                
                # Calculate crop-specific factor
                crop_factor = get_crop_factor(selected_crop, temperature, rainfall)
                adjusted_yield = base_yield * crop_factor
                
                # Determine environmental impact message
                conditions = crop_optimal_conditions.get(selected_crop, crop_optimal_conditions['DEFAULT'])
                
                impacts = []
                if conditions['temp_min'] <= temperature <= conditions['temp_max']:
                    impacts.append("✓ Optimal temperature")
                elif temperature < conditions['temp_min']:
                    impacts.append(f"⚠ Temperature too low (needs {conditions['temp_min']}°C+)")
                else:
                    impacts.append(f"⚠ Temperature too high (max {conditions['temp_max']}°C)")
                
                if conditions['rain_min'] <= rainfall <= conditions['rain_max']:
                    impacts.append("✓ Optimal rainfall")
                elif rainfall < conditions['rain_min']:
                    impacts.append(f"⚠ Insufficient rainfall (needs {conditions['rain_min']}mm+)")
                else:
                    impacts.append(f"⚠ Excess rainfall (max {conditions['rain_max']}mm)")
                
                environmental_impact = " | ".join(impacts)
                estimated_yield = adjusted_yield

                # Encode crop
                crop_encoded = crop_encoder.transform([selected_crop])[0]

                # Scale continuous features
                continuous_data = [[area, production, rainfall, temperature]]
                continuous_scaled = scaler.transform(continuous_data)
                
                # Combine with crop encoding
                input_combined = np.hstack([[crop_encoded], continuous_scaled[0]])

                prediction = model.predict([input_combined])
                result = yield_encoder.inverse_transform(prediction)[0]

        except Exception as e:
            print(f"Error during prediction: {e}")
            import traceback
            traceback.print_exc()
            result = "Error in prediction"

    return render_template(
        "index.html",
        crops=sorted(crop_encoder.classes_),
        result=result,
        estimated_yield=estimated_yield,
        selected_crop=selected_crop,
        area=area,
        production=production,
        rainfall=rainfall,
        temperature=temperature,
        environmental_impact=environmental_impact
    )

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Starting Flask app on http://127.0.0.1:5000")
    print("Now with TEMPERATURE parameter!")
    print("Different crops now give different results!")
    print("="*60 + "\n")
    app.run(debug=True, port=5000)