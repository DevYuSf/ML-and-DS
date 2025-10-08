from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global variables to store loaded models
coffee_models = None
price_model = None
coffee_encoder = None
price_encoder = None
label_encoders = None
feature_names = None
best_model_name = None

def load_models():
    global coffee_models, price_model, coffee_encoder, price_encoder
    global label_encoders, feature_names, best_model_name
    
    try:
        print("🔮 Loading trained ML models...")
        
        model_artifacts = joblib.load('models/coffee_prediction_system_v2.pkl')
        
        # Extract all components
        coffee_models = model_artifacts['coffee_models']
        price_model = model_artifacts['price_model']
        coffee_encoder = model_artifacts['coffee_encoder']
        price_encoder = model_artifacts['price_encoder']
        label_encoders = model_artifacts['label_encoders']
        feature_names = model_artifacts['feature_names']
        
        # Handle missing keys
        if 'best_model_name' in model_artifacts:
            best_model_name = model_artifacts['best_model_name']
        else:
            best_model_name = 'LogisticRegression'
            print(f"⚠️  best_model_name not found, using default: {best_model_name}")
        
        # Auto-detect best model from performance
        if 'model_performance' in model_artifacts:
            model_perf = model_artifacts['model_performance']
            best_model_name = max(model_perf.items(), key=lambda x: x[1]['accuracy'])[0]
            print(f"🔍 Auto-detected best model: {best_model_name}")
        
        print(f"✅ Models loaded successfully!")
        print(f"   Best model: {best_model_name}")
        print(f"   Coffee classes: {list(coffee_encoder.classes_)}")
        print(f"   Price tiers: {list(price_encoder.classes_)}")
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        raise e

def get_day_part(hour):
    if 5 <= hour <= 11: return 'Morning'
    elif 12 <= hour <= 16: return 'Afternoon'
    else: return 'Evening'

def prepare_features(hour, weekday, month_name):
    is_peak_morning = 1 if 7 <= hour <= 10 else 0
    is_peak_afternoon = 1 if 14 <= hour <= 17 else 0
    is_weekend = 1 if weekday in ['Sat', 'Sun'] else 0
    day_part = get_day_part(hour)
    is_cold_season = 1 if month_name in ['Jan', 'Feb', 'Dec', 'Mar'] else 0
    is_warm_season = 1 if month_name in ['Jun', 'Jul', 'Aug'] else 0
    

    # Instead of setting to 0, use REASONABLE DEFAULTS based on time patterns
    if 7 <= hour <= 10:  # Morning peak
        is_milk_based, is_espresso, is_chocolate = 0, 1, 0  # More espresso in morning
    elif 14 <= hour <= 17:  # Afternoon peak  
        is_milk_based, is_espresso, is_chocolate = 1, 0, 0  # More milk drinks in afternoon
    else:  # Evening
        is_milk_based, is_espresso, is_chocolate = 0, 0, 1  # More chocolate in evening
    
    # Weekend adjustment
    if weekday in ['Sat', 'Sun']:
        is_milk_based, is_chocolate = 1, 1  # More indulgent drinks on weekends


  
    # Encode categorical features
    try:
        weekday_encoded = label_encoders['Weekday'].transform([weekday])[0]
        month_encoded = label_encoders['Month_name'].transform([month_name])[0]
        day_part_encoded = label_encoders['day_part'].transform([day_part])[0]
    except Exception as e:
        print(f"Encoding error: {e}")
        weekday_encoded, month_encoded, day_part_encoded = 0, 0, 0
    
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
    
    return pd.DataFrame([features])[feature_names]

def generate_business_insights(coffee_group, price_tier, hour, weekday):
    insights = []
    
    # Time-based insights with emojis
    if 5 <= hour <= 8:
        insights.append("🌅 Early bird special! Perfect time for quick, energizing coffees")
    elif 7 <= hour <= 10:
        insights.append("🚀 Morning rush! Customers need their caffeine boost")
    elif 11 <= hour <= 13:
        insights.append("🍽️ Lunch crowd incoming! Great time for balanced drinks")
    elif 14 <= hour <= 17:
        insights.append("☀️ Afternoon delight! Customers want leisurely, premium drinks")
    elif 18 <= hour <= 21:
        insights.append("🌙 Evening relaxation! Comfort drinks are trending")
    else:
        insights.append("🌜 Late night cravings! Sweet and comforting options popular")
    
    # Weekend insights
    if weekday in ['Sat', 'Sun']:
        insights.append("🎉 Weekend vibes! Customers are here to indulge and relax")
    else:
        insights.append("💼 Workday mode! Quick and efficient orders expected")
    
    # Coffee group specific insights
    if coffee_group == 'Milk_Based_Drinks':
        insights.append("🥛 Creamy delights! Perfect for leisurely sipping")
        insights.append("💡 Pro tip: Steam extra milk - these are crowd pleasers!")
    elif coffee_group == 'Strong_Coffee':
        insights.append("⚡ Power boost! Ideal for morning energy or focus time")
        insights.append("💡 Pro tip: Have espresso shots ready to go!")
    elif coffee_group == 'Chocolate_Drinks':
        insights.append("🍫 Sweet treats! Great for comfort and indulgence")
        insights.append("💡 Pro tip: Display chocolate shavings and marshmallows!")
    elif coffee_group == 'Americano_with_Milk':
        insights.append("☕ Balanced choice! The perfect middle-ground coffee")
        insights.append("💡 Pro tip: Keep Americano shots and milk pitchers handy!")
    
    # Price tier insights
    if price_tier == 'Luxury':
        insights.append("💎 Premium time! Customers willing to splurge on quality")
    elif price_tier == 'Premium':
        insights.append("⭐ Quality hour! Perfect for showcasing your best offerings")
    elif price_tier == 'Standard':
        insights.append("📊 Steady demand! Reliable favorites are trending")
    elif price_tier == 'Budget':
        insights.append("💰 Value time! Focus on quick, efficient service")
    
    return insights


def get_coffee_recommendations(coffee_group):
    """Get specific coffee recommendations based on the predicted group"""
    recommendations = {
        'Milk_Based_Drinks': {
            'recommendations': ['Latte', 'Cappuccino', 'Cortado', 'Flat White'],
            'description': 'Creamy & comforting milk-based delights 🥛',
            'prep_tip': 'Steam milk to 65°C for perfect texture',
            'serving_suggestion': 'Serve with latte art for extra appeal'
        },
        'Strong_Coffee': {
            'recommendations': ['Espresso', 'Americano', 'Double Espresso', 'Ristretto'],
            'description': 'Bold & energizing strong coffees ⚡',
            'prep_tip': 'Use freshly ground beans for maximum flavor',
            'serving_suggestion': 'Pair with a small glass of water'
        },
        'Chocolate_Drinks': {
            'recommendations': ['Hot Chocolate', 'Cocoa', 'Mocha', 'White Hot Chocolate'],
            'description': 'Sweet & indulgent chocolate treats 🍫',
            'prep_tip': 'Use high-quality cocoa powder for rich flavor',
            'serving_suggestion': 'Top with whipped cream and chocolate shavings'
        },
        'Americano_with_Milk': {
            'recommendations': ['Americano with Milk', 'Long Black with Milk', 'Café Au Lait'],
            'description': 'Perfectly balanced coffee with milk ☕',
            'prep_tip': 'Brew strong coffee and add steamed milk to taste',
            'serving_suggestion': 'Offer sugar and cinnamon on the side'
        }
    }
    
    return recommendations.get(coffee_group, {
        'recommendations': ['Latte', 'Americano', 'Cappuccino', 'Hot Chocolate'],
        'description': 'Popular crowd favorites 🌟',
        'prep_tip': 'Keep all stations ready for varied orders',
        'serving_suggestion': 'Smile and provide excellent service!'
    })

# ==================== API ROUTES ====================

@app.route('/')
def home():
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
    return jsonify({
        "status": "healthy",
        "models_loaded": coffee_models is not None,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/models')
def get_model_info():
    if coffee_models is None:
        return jsonify({"error": "Models not loaded"}), 500
    
    # Default performance metrics
    model_performance = {
        'LogisticRegression': {'accuracy': 0.963, 'speed': 'fast'},
        'RandomForest': {'accuracy': 0.952, 'speed': 'medium'},
        'NaiveBayes': {'accuracy': 0.963, 'speed': 'very fast'},
        'XGBoost': {'accuracy': 0.948, 'speed': 'medium'}
    }
    
    return jsonify({
        "best_coffee_model": best_model_name,
        "available_coffee_models": list(coffee_models.keys()),
        "model_performance": model_performance,
        "coffee_categories": list(coffee_encoder.classes_),
        "price_tiers": list(price_encoder.classes_),
        "features_used": feature_names
    })

@app.route('/models/select', methods=['POST'])
def select_model():
    try:
        data = request.get_json()
        selected_model = data.get('model_name')
        
        if selected_model not in coffee_models:
            return jsonify({
                "error": f"Model '{selected_model}' not available. Choose from: {list(coffee_models.keys())}"
            }), 400
        
        global best_model_name
        best_model_name = selected_model
        
        return jsonify({
            "success": True,
            "message": f"Model changed to {selected_model}",
            "selected_model": selected_model,
            "available_models": list(coffee_models.keys())
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Get model selection (optional)
        selected_model = data.get('model', best_model_name)
        if selected_model not in coffee_models:
            selected_model = best_model_name
        
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
        
        print(f"🎯 Prediction request - Model: {selected_model}, Time: {hour}:00, {weekday}, {month}")
        
        # Prepare features and make predictions
        features_df = prepare_features(hour, weekday, month)
        model = coffee_models[selected_model]
        
        coffee_pred_encoded = model.predict(features_df)[0]
        coffee_pred = coffee_encoder.inverse_transform([coffee_pred_encoded])[0]
        coffee_confidence = float(np.max(model.predict_proba(features_df)[0]) * 100)
        
        price_pred_encoded = price_model.predict(features_df)[0]
        price_pred = price_encoder.inverse_transform([price_pred_encoded])[0]
        price_confidence = float(np.max(price_model.predict_proba(features_df)[0]) * 100)
        
        insights = generate_business_insights(coffee_pred, price_pred, hour, weekday)
        recommendations = get_coffee_recommendations(coffee_pred)
        
        response = {
            "success": True,
            "predictions": {
                "coffee_group": coffee_pred,
                "coffee_confidence": round(coffee_confidence, 1),
                "price_tier": price_pred,
                "price_confidence": round(price_confidence, 1),
                "model_used": selected_model
            },
            "recommendations": {
                "coffee_type_description": recommendations['description'],
                "suggested_drinks": recommendations['recommendations'],
                "preparation_tip": recommendations['prep_tip'],
                "serving_suggestion": recommendations['serving_suggestion'],
                "success_message": f"🎯 Perfect! Based on current patterns, focus on {recommendations['description'].lower()}"
            },
            "insights": insights,
            "input_parameters": {
                "hour": hour,
                "weekday": weekday,
                "month": month
            },
            "model_info": {
                "selected": selected_model,
                "available": list(coffee_models.keys()),
                "default": best_model_name
            }
        }
        
        print(f"✅ Prediction successful: {coffee_pred} ({coffee_confidence:.1f}%) using {selected_model}")
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/predict/batch', methods=['POST'])
def batch_predict():
    try:
        data = request.get_json()
        inputs = data.get('inputs', [])
        
        results = []
        for input_data in inputs:
            features_df = prepare_features(
                input_data['hour'], 
                input_data['weekday'], 
                input_data['month']
            )
            
            model = coffee_models[best_model_name]
            
            coffee_pred_encoded = model.predict(features_df)[0]
            coffee_pred = coffee_encoder.inverse_transform([coffee_pred_encoded])[0]
            coffee_confidence = float(np.max(model.predict_proba(features_df)[0]) * 100)
            
            price_pred_encoded = price_model.predict(features_df)[0]
            price_pred = price_encoder.inverse_transform([price_pred_encoded])[0]
            price_confidence = float(np.max(price_model.predict_proba(features_df)[0]) * 100)
            
            recommendations = get_coffee_recommendations(coffee_pred)
            
            results.append({
                "input": input_data,
                "coffee_group": coffee_pred,
                "coffee_confidence": round(coffee_confidence, 1),
                "price_tier": price_pred,
                "price_confidence": round(price_confidence, 1),
                "recommendations": {
                    "suggested_drinks": recommendations['recommendations'][:2],  # Top 2 recommendations
                    "preparation_tip": recommendations['prep_tip']
                }
            })
        
        return jsonify({"success": True, "results": results})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==================== MAIN ====================

if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)
    
    print("🚀 Starting Coffee Prediction API...")
    load_models()
    
    print("🌐 API Server starting on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)