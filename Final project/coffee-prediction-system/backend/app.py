from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import os
import json

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Global variables to store loaded models
coffee_models = None
price_model = None
coffee_encoder = None
price_encoder = None
label_encoders = None
feature_names = None
best_model_name = None

# Load ML models
def load_models():
    global coffee_models, price_model, coffee_encoder, price_encoder
    global label_encoders, feature_names, best_model_name
    
    try:
        print("🔮 Loading trained ML models...")
        
        # Load the saved model artifacts
        model_artifacts = joblib.load('models/coffee_prediction_system.pkl')
        
        # Extract all components
        coffee_models = model_artifacts['coffee_models']
        price_model = model_artifacts['price_model']
        coffee_encoder = model_artifacts['coffee_encoder']
        price_encoder = model_artifacts['price_encoder']
        label_encoders = model_artifacts['label_encoders']
        feature_names = model_artifacts['feature_names']
        best_model_name = model_artifacts['best_model_name']
        
        print(f"✅ Models loaded successfully!")
        print(f"   Best model: {best_model_name}")
        print(f"   Coffee classes: {list(coffee_encoder.classes_)}")
        print(f"   Price tiers: {list(price_encoder.classes_)}")
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        raise e

# Feature engineering function (same as training)
def get_day_part(hour):
    """Convert hour to morning/afternoon/evening categories"""
    if 5 <= hour <= 11: return 'Morning'
    elif 12 <= hour <= 16: return 'Afternoon'
    else: return 'Evening'

def prepare_features(hour, weekday, month_name):
    """Convert user input to model features"""
    
    # Calculate derived features
    is_peak_morning = 1 if 7 <= hour <= 10 else 0
    is_peak_afternoon = 1 if 14 <= hour <= 17 else 0
    is_weekend = 1 if weekday in ['Sat', 'Sun'] else 0
    day_part = get_day_part(hour)
    is_cold_season = 1 if month_name in ['Jan', 'Feb', 'Dec', 'Mar'] else 0
    is_warm_season = 1 if month_name in ['Jun', 'Jul', 'Aug'] else 0
    
    # Initialize binary features (these will be determined by prediction)
    is_milk_based = 0
    is_espresso = 0
    is_chocolate = 0
    
    # Encode categorical features using saved encoders
    try:
        weekday_encoded = label_encoders['Weekday'].transform([weekday])[0]
        month_encoded = label_encoders['Month_name'].transform([month_name])[0]
        day_part_encoded = label_encoders['day_part'].transform([day_part])[0]
    except Exception as e:
        print(f"Encoding error: {e}")
        # Default fallback encodings
        weekday_encoded = 0
        month_encoded = 0
        day_part_encoded = 0
    
    # Create feature dictionary in exact same order as training
    features = {
        'hour_of_day': hour,
        'is_peak_morning': is_peak_morning,
        'is_peak_afternoon': is_peak_afternoon,
        'is_weekend': is_weekend,
        'is_milk_based': is_milk_based,
        'is_espresso': is_espresso,
        'is_chocolate': is_chocolate,
        'is_cold_season': is_cold_season,
        'is_warm_season': is_warm_season,
        'Weekday_encoded': weekday_encoded,
        'Month_name_encoded': month_encoded,
        'day_part_encoded': day_part_encoded
    }
    
    # Convert to DataFrame with correct column order
    features_df = pd.DataFrame([features])[feature_names]
    
    return features_df

# ==================== API ROUTES ====================

@app.route('/')
def home():
    """API Homepage"""
    return jsonify({
        "message": "☕ Coffee Prediction API is running!",
        "endpoints": {
            "/predict": "Get coffee and price predictions",
            "/models": "Get model information",
            "/health": "Check API health"
        },
        "status": "active"
    })

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "models_loaded": coffee_models is not None,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/models')
def get_model_info():
    """Get information about loaded models"""
    if coffee_models is None:
        return jsonify({"error": "Models not loaded"}), 500
    
    return jsonify({
        "best_coffee_model": best_model_name,
        "available_coffee_models": list(coffee_models.keys()),
        "coffee_categories": list(coffee_encoder.classes_),
        "price_tiers": list(price_encoder.classes_),
        "features_used": feature_names
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['hour', 'weekday', 'month']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        hour = data['hour']
        weekday = data['weekday']
        month = data['month']
        
        # Validate input ranges
        if not (0 <= hour <= 23):
            return jsonify({"error": "Hour must be between 0-23"}), 400
        
        valid_weekdays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        if weekday not in valid_weekdays:
            return jsonify({"error": f"Weekday must be one of {valid_weekdays}"}), 400
        
        valid_months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        if month not in valid_months:
            return jsonify({"error": f"Month must be one of {valid_months}"}), 400
        
        print(f"🎯 Prediction request: {hour}:00, {weekday}, {month}")
        
        # Prepare features for model
        features_df = prepare_features(hour, weekday, month)
        
        # Get best coffee model
        best_model = coffee_models[best_model_name]
        
        # Make predictions
        coffee_pred_encoded = best_model.predict(features_df)[0]
        coffee_pred = coffee_encoder.inverse_transform([coffee_pred_encoded])[0]
        coffee_confidence = float(np.max(best_model.predict_proba(features_df)[0]) * 100)
        
        price_pred_encoded = price_model.predict(features_df)[0]
        price_pred = price_encoder.inverse_transform([price_pred_encoded])[0]
        price_confidence = float(np.max(price_model.predict_proba(features_df)[0]) * 100)
        
        # Get business insights
        insights = generate_business_insights(coffee_pred, price_pred, hour, weekday)
        
        # Prepare response
        response = {
            "success": True,
            "predictions": {
                "coffee_group": coffee_pred,
                "coffee_confidence": round(coffee_confidence, 1),
                "price_tier": price_pred,
                "price_confidence": round(price_confidence, 1),
                "model_used": best_model_name
            },
            "insights": insights,
            "input_parameters": {
                "hour": hour,
                "weekday": weekday,
                "month": month
            }
        }
        
        print(f"✅ Prediction successful: {coffee_pred} ({coffee_confidence:.1f}%)")
        
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

def generate_business_insights(coffee_group, price_tier, hour, weekday):
    """Generate business insights based on predictions"""
    insights = []
    
    # Time-based insights
    if 7 <= hour <= 10:
        insights.append("🌅 Morning rush hour - expect high demand for quick coffees")
    elif 14 <= hour <= 17:
        insights.append("☀️ Afternoon leisure time - premium drinks popular")
    
    # Weekend insights
    if weekday in ['Sat', 'Sun']:
        insights.append("🎉 Weekend trend - customers prefer indulgent, leisurely drinks")
    
    # Coffee group insights
    if coffee_group == 'Milk_Based_Drinks':
        insights.append("🥛 Milk-based drinks trending - ensure milk inventory is stocked")
    elif coffee_group == 'Pure_Espresso':
        insights.append("⚡ Strong coffee preference - espresso machines should be ready")
    elif coffee_group == 'Chocolate_Drinks':
        insights.append("🍫 Chocolate drinks popular - great for cold weather")
    
    # Price tier insights
    if price_tier == 'Luxury':
        insights.append("💎 Premium pricing acceptable - customers willing to spend more")
    elif price_tier == 'Budget':
        insights.append("💰 Price-sensitive customers - consider promotions")
    
    return insights

@app.route('/predict/batch', methods=['POST'])
def batch_predict():
    """Batch prediction endpoint for multiple inputs"""
    try:
        data = request.get_json()
        inputs = data.get('inputs', [])
        
        results = []
        for input_data in inputs:
            # Reuse single prediction logic
            features_df = prepare_features(
                input_data['hour'], 
                input_data['weekday'], 
                input_data['month']
            )
            
            best_model = coffee_models[best_model_name]
            
            coffee_pred_encoded = best_model.predict(features_df)[0]
            coffee_pred = coffee_encoder.inverse_transform([coffee_pred_encoded])[0]
            coffee_confidence = float(np.max(best_model.predict_proba(features_df)[0]) * 100)
            
            price_pred_encoded = price_model.predict(features_df)[0]
            price_pred = price_encoder.inverse_transform([price_pred_encoded])[0]
            price_confidence = float(np.max(price_model.predict_proba(features_df)[0]) * 100)
            
            results.append({
                "input": input_data,
                "coffee_group": coffee_pred,
                "coffee_confidence": round(coffee_confidence, 1),
                "price_tier": price_pred,
                "price_confidence": round(price_confidence, 1)
            })
        
        return jsonify({"success": True, "results": results})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==================== MAIN ====================

if __name__ == '__main__':
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Load models when starting the app
    print("🚀 Starting Coffee Prediction API...")
    load_models()
    
    # Start Flask development server
    print("🌐 API Server starting on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)