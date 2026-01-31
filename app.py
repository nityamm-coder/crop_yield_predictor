from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
import os
warnings.filterwarnings('ignore')

app = Flask(__name__)

# [Previous data loading code - keeping it the same...]
print("Loading crop production data...")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

crop_df = pd.read_csv(os.path.join(BASE_DIR, "crop_production.csv"))
rain_df = pd.read_csv(os.path.join(BASE_DIR, "rainfall.csv"))
crop_df = crop_df.dropna()
crop_df = crop_df[["State_Name", "Crop_Year", "Crop", "Area", "Production"]]
crop_df["Year"] = crop_df["Crop_Year"].astype(str).str[:4].astype(int)

indian_state_rainfall = {
    'Andaman and Nicobar Islands': 3100, 'Andhra Pradesh': 940, 'Arunachal Pradesh': 2800,
    'Assam': 2400, 'Bihar': 1200, 'Chandigarh': 1100, 'Chhattisgarh': 1400,
    'Dadra and Nagar Haveli': 2000, 'Daman and Diu': 2000, 'Delhi': 800,
    'Goa': 3000, 'Gujarat': 800, 'Haryana': 600, 'Himachal Pradesh': 1500,
    'Jammu and Kashmir': 1000, 'Jharkhand': 1400, 'Karnataka': 1200, 'Kerala': 3000,
    'Madhya Pradesh': 1200, 'Maharashtra': 1200, 'Manipur': 1800, 'Meghalaya': 2500,
    'Mizoram': 2500, 'Nagaland': 2000, 'Odisha': 1500, 'Puducherry': 1200,
    'Punjab': 600, 'Rajasthan': 600, 'Sikkim': 2500, 'Tamil Nadu': 1000,
    'Telangana': 900, 'Tripura': 2400, 'Uttar Pradesh': 1000,
    'Uttarakhand': 1500, 'West Bengal': 1800
}

indian_state_temperature = {
    'Andaman and Nicobar Islands': 27, 'Andhra Pradesh': 29, 'Arunachal Pradesh': 18,
    'Assam': 24, 'Bihar': 26, 'Chandigarh': 24, 'Chhattisgarh': 27,
    'Dadra and Nagar Haveli': 28, 'Daman and Diu': 28, 'Delhi': 25, 'Goa': 27,
    'Gujarat': 27, 'Haryana': 24, 'Himachal Pradesh': 16, 'Jammu and Kashmir': 14,
    'Jharkhand': 25, 'Karnataka': 25, 'Kerala': 27, 'Madhya Pradesh': 26,
    'Maharashtra': 26, 'Manipur': 21, 'Meghalaya': 20, 'Mizoram': 21, 'Nagaland': 21,
    'Odisha': 27, 'Puducherry': 28, 'Punjab': 23, 'Rajasthan': 27, 'Sikkim': 15,
    'Tamil Nadu': 28, 'Telangana': 28, 'Tripura': 24, 'Uttar Pradesh': 25,
    'Uttarakhand': 20, 'West Bengal': 26
}

crop_df['ANNUAL_RAINFALL'] = crop_df['State_Name'].map(indian_state_rainfall)
crop_df['AVG_TEMPERATURE'] = crop_df['State_Name'].map(indian_state_temperature)
crop_df['ANNUAL_RAINFALL'].fillna(1170, inplace=True)
crop_df['AVG_TEMPERATURE'].fillna(25, inplace=True)

np.random.seed(42)
crop_df['ANNUAL_RAINFALL'] = crop_df['ANNUAL_RAINFALL'] * np.random.uniform(0.9, 1.1, size=len(crop_df))
crop_df['AVG_TEMPERATURE'] = crop_df['AVG_TEMPERATURE'] + np.random.uniform(-2, 2, size=len(crop_df))

# ✅ EXPANDED crop optimal conditions - more crops covered!
crop_optimal_conditions = {
    'Rice': {'temp_min': 20, 'temp_max': 35, 'rain_min': 1000, 'rain_max': 2500},
    'Wheat': {'temp_min': 15, 'temp_max': 25, 'rain_min': 400, 'rain_max': 800},
    'Sugarcane': {'temp_min': 20, 'temp_max': 35, 'rain_min': 1500, 'rain_max': 2500},
    'Cotton': {'temp_min': 21, 'temp_max': 30, 'rain_min': 500, 'rain_max': 1000},
    'Maize': {'temp_min': 21, 'temp_max': 30, 'rain_min': 500, 'rain_max': 1000},
    'Jowar': {'temp_min': 26, 'temp_max': 33, 'rain_min': 400, 'rain_max': 600},
    'Bajra': {'temp_min': 25, 'temp_max': 35, 'rain_min': 400, 'rain_max': 600},
    'Ragi': {'temp_min': 20, 'temp_max': 30, 'rain_min': 500, 'rain_max': 1000},
    'Groundnut': {'temp_min': 20, 'temp_max': 30, 'rain_min': 500, 'rain_max': 1250},
    'Soyabean': {'temp_min': 20, 'temp_max': 32, 'rain_min': 450, 'rain_max': 700},
    'Sunflower': {'temp_min': 20, 'temp_max': 30, 'rain_min': 500, 'rain_max': 750},
    'Potato': {'temp_min': 15, 'temp_max': 25, 'rain_min': 500, 'rain_max': 750},
    'Onion': {'temp_min': 13, 'temp_max': 35, 'rain_min': 650, 'rain_max': 1000},
    'Tomato': {'temp_min': 20, 'temp_max': 30, 'rain_min': 600, 'rain_max': 1250},
    'Apple': {'temp_min': 5, 'temp_max': 24, 'rain_min': 600, 'rain_max': 1000},
    'Banana': {'temp_min': 15, 'temp_max': 35, 'rain_min': 1000, 'rain_max': 2500},
    'Mango': {'temp_min': 18, 'temp_max': 40, 'rain_min': 750, 'rain_max': 2500},
    'Tea': {'temp_min': 10, 'temp_max': 35, 'rain_min': 1500, 'rain_max': 3000},
    'Tobacco': {'temp_min': 16, 'temp_max': 35, 'rain_min': 500, 'rain_max': 1000},
    'Coffee': {'temp_min': 15, 'temp_max': 30, 'rain_min': 1000, 'rain_max': 2000},
    'Carrot': {'temp_min': 7, 'temp_max': 24, 'rain_min': 600, 'rain_max': 1200},
    'Grapes': {'temp_min': 10, 'temp_max': 35, 'rain_min': 500, 'rain_max': 900},
    'Orange': {'temp_min': 13, 'temp_max': 37, 'rain_min': 750, 'rain_max': 1200},
    'DEFAULT': {'temp_min': 20, 'temp_max': 30, 'rain_min': 600, 'rain_max': 1500}
}

def get_crop_factor(crop, temperature, rainfall):
    conditions = crop_optimal_conditions.get(crop, crop_optimal_conditions['DEFAULT'])
    
    if conditions['temp_min'] <= temperature <= conditions['temp_max']:
        temp_factor = 1.0
    else:
        if temperature < conditions['temp_min']:
            temp_diff = conditions['temp_min'] - temperature
            temp_factor = max(0.7, 1.0 - (temp_diff * 0.05))
        else:
            temp_diff = temperature - conditions['temp_max']
            temp_factor = max(0.7, 1.0 - (temp_diff * 0.05))
    
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

crop_df["Base_Yield"] = crop_df["Production"] / crop_df["Area"]
crop_df['Crop_Factor'] = crop_df.apply(
    lambda row: get_crop_factor(row['Crop'], row['AVG_TEMPERATURE'], row['ANNUAL_RAINFALL']), axis=1
)
crop_df['Adjusted_Yield'] = crop_df['Base_Yield'] * crop_df['Crop_Factor']

def yield_category(y):
    if y < 1.5:
        return "Low"
    elif y < 3.0:
        return "Medium"
    else:
        return "High"

crop_df["Yield_Category"] = crop_df["Adjusted_Yield"].apply(yield_category)

crop_encoder = LabelEncoder()
crop_df["Crop_Encoded"] = crop_encoder.fit_transform(crop_df["Crop"])

yield_encoder = LabelEncoder()
y_encoded = yield_encoder.fit_transform(crop_df["Yield_Category"])

scaler = StandardScaler()
X_continuous = crop_df[["Area", "Production", "ANNUAL_RAINFALL", "AVG_TEMPERATURE"]]
X_continuous_scaled = scaler.fit_transform(X_continuous)
X_crop = crop_df[["Crop_Encoded"]].values
X_combined = np.hstack([X_crop, X_continuous_scaled])
y = y_encoded

model = MLPClassifier(
    hidden_layer_sizes=(64, 32, 16),
    activation="relu",
    max_iter=2000,
    random_state=42,
    learning_rate='adaptive',
    early_stopping=True,
    validation_fraction=0.1,
    verbose=False
)

model.fit(X_combined, y)
print(f"✅ ANN Model trained! Accuracy: {model.score(X_combined, y):.4f}")

# ==============================
# FLASK ROUTE - HYBRID APPROACH
# ==============================
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    estimated_yield = None
    yield_efficiency = None
    selected_crop = None
    area = production = rainfall = temperature = None
    environmental_impact = None
    model_confidence = None
    ann_prediction = None

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
                
                # Calculate environmental factor
                crop_factor = get_crop_factor(selected_crop, temperature, rainfall)
                
                # Calculate adjusted yield
                adjusted_yield = base_yield * crop_factor
                yield_efficiency = crop_factor * 100
                estimated_yield = adjusted_yield
                
                # ✅ GET ANN PREDICTION (for confidence and validation)
                crop_encoded = crop_encoder.transform([selected_crop])[0]
                continuous_data = [[area, production, rainfall, temperature]]
                continuous_scaled = scaler.transform(continuous_data)
                input_combined = np.hstack([[crop_encoded], continuous_scaled[0]])
                
                # Get ANN prediction
                ann_pred = model.predict([input_combined])
                ann_prediction = yield_encoder.inverse_transform(ann_pred)[0]
                prediction_proba = model.predict_proba([input_combined])[0]
                model_confidence = max(prediction_proba) * 100
                
                # ✅ HYBRID DECISION: Use rule-based for validation
                # Calculate what the category SHOULD be based on adjusted yield
                if adjusted_yield < 1.5:
                    rule_based_result = "Low"
                elif adjusted_yield < 3.0:
                    rule_based_result = "Medium"
                else:
                    rule_based_result = "High"
                
                # ✅ USE ANN PREDICTION IF IT AGREES, OTHERWISE USE RULE-BASED
                # This ensures consistency while still using the neural network
                if model_confidence > 60:  # If ANN is confident
                    # Check if ANN prediction makes sense with the yield value
                    if (ann_prediction == "High" and adjusted_yield > 2.5) or \
                       (ann_prediction == "Medium" and 1.2 < adjusted_yield < 3.5) or \
                       (ann_prediction == "Low" and adjusted_yield < 2.0):
                        result = ann_prediction  # ANN makes sense, use it
                    else:
                        result = rule_based_result  # ANN confused, use rule
                else:
                    result = rule_based_result  # ANN not confident, use rule
                
                # Environmental feedback
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
        yield_efficiency=yield_efficiency,
        model_confidence=model_confidence,
        ann_prediction=ann_prediction,
        selected_crop=selected_crop,
        area=area,
        production=production,
        rainfall=rainfall,
        temperature=temperature,
        environmental_impact=environmental_impact
    )

#if __name__ == "__main__":
    #print("\n" + "="*70)
    #print("🌾 ANN-Based Crop Yield Prediction System")
    #print("="*70)
    #print("✅ Hybrid ANN + Rule-Based Validation")
    #print("✅ Neural Network learns patterns, rules ensure consistency")
    #print("="*70)
    #print("\nStarting Flask app on http://127.0.0.1:5000")
    #print("="*70 + "\n")
    #app.run(debug=True, port=5000)


