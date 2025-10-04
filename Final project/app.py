import pandas as pd

df = pd.read_csv('coffe_sales.csv')
# print("=== INITIAL DATA INSPECTION ===")
# print(df.info(10))
# Features to remove
features_to_drop = [
    'Date',           # Redundant with Month_name/Monthsort
    'Time',           # Redundant with hour_of_day
    'Weekdaysort',    # Redundant with Weekday (one-hot encode Weekday instead)
    'Monthsort',      # Redundant with Month_name
    'Time_of_Day'     # Can be derived from hour_of_day
]

df_clean = df.drop(columns=features_to_drop)
# print(f"Features after cleaning: {df_clean.columns.tolist()}")
# print(df_clean["Month_name"].value_counts() )
# print(df_clean["Weekday"].value_counts() )
# print(df_clean["hour_of_day"].value_counts() )
# print(df_clean["coffee_name"].value_counts() )
# print(df_clean["money"].value_counts() )
# print(df_clean["cash_type"].value_counts() )
#
#
#
# First, let's understand the relationship between coffee_name and money
# print("Price analysis by coffee type:")
price_analysis = df_clean.groupby('coffee_name')['money'].agg(['min', 'max', 'mean', 'count', 'nunique']).round(2)
# print(price_analysis)

# Check if price is perfectly determined by coffee name
price_uniqueness = df_clean.groupby('coffee_name')['money'].nunique()
# print(f"\nUnique prices per coffee type:")
# print(price_uniqueness)

# Drop the useless cash_type column (only one value)
df_clean = df_clean.drop(columns=['cash_type'])
# print(f"\nFeatures after removing cash_type: {df_clean.columns.tolist()}")

#
#
#

# === STEP 2A: Time-Based Features (ESSENTIAL) ===
df_clean['is_peak_morning'] = ((df_clean['hour_of_day'] >= 7) & (df_clean['hour_of_day'] <= 10)).astype(int)
df_clean['is_peak_afternoon'] = ((df_clean['hour_of_day'] >= 14) & (df_clean['hour_of_day'] <= 17)).astype(int)

def get_day_part(hour):
    if 5 <= hour <= 11:
        return 'Morning'
    elif 12 <= hour <= 16:
        return 'Afternoon'
    else:
        return 'Evening'

df_clean['day_part'] = df_clean['hour_of_day'].apply(get_day_part)

# === STEP 2B: Weekend Fix (ESSENTIAL) ===
df_clean['is_weekend'] = (df_clean['Weekday'].isin(['Sat', 'Sun'])).astype(int)

# === STEP 2C: Coffee Categories (ESSENTIAL) ===
df_clean['is_milk_based'] = df_clean['coffee_name'].str.contains('Latte|Cappuccino|Cortado', case=False, na=False).astype(int)
df_clean['is_espresso'] = df_clean['coffee_name'].str.contains('Espresso|Americano', case=False, na=False).astype(int)
df_clean['is_chocolate'] = df_clean['coffee_name'].str.contains('Chocolate|Cocoa', case=False, na=False).astype(int)

df_clean['is_cold_season'] = df_clean['Month_name'].isin(['Jan', 'Feb', 'Dec', 'Mar']).astype(int)
df_clean['is_warm_season'] = df_clean['Month_name'].isin(['Jun', 'Jul', 'Aug']).astype(int)

# === STEP 2D: Final Encoding (ESSENTIAL) ===
df_clean['price_tier'] = pd.cut(df_clean['money'], 
                               bins=[0, 25, 30, 35, float('inf')],
                               labels=['Budget', 'Standard', 'Premium', 'Luxury'])

from sklearn.preprocessing import LabelEncoder
label_encoders = {}
categorical_features = ['Weekday', 'Month_name', 'day_part']

for col in categorical_features:
    le = LabelEncoder()
    df_clean[col + '_encoded'] = le.fit_transform(df_clean[col])
    label_encoders[col] = le

numerical_features = ['hour_of_day', 'is_peak_morning', 'is_peak_afternoon', 
                     'is_weekend', 'is_milk_based', 'is_espresso', 'is_chocolate',
                     'is_cold_season', 'is_warm_season']

all_features = numerical_features + [f"{col}_encoded" for col in categorical_features]

# print("=== FEATURE ENGINEERING COMPLETE ===")
# print(f"Total features: {len(all_features)}")
# print(f"Dataset shape: {df_clean.shape}")
# print("=== CLEANED DATA INSPECTION ===")
# print(df_clean.info())


# print("\n=== CHOOSE PREDICTION TARGET ===")
# print("Coffee types distribution:")
# print(df_clean['coffee_name'].value_counts())

# print("\nPrice tiers distribution:") 
# print(df_clean['price_tier'].value_counts())

# print("\nWhich target to predict?")
# print("1. coffee_name - Predict specific drink (8 classes) - Better for menu planning")
# print("2. price_tier - Predict spending level (4 classes) - Better for revenue forecasting")

# print("=== Step 3: Building Dual Prediction Models ===")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib  # For saving models

# Prepare features and both targets
X = df_clean[all_features]
y_coffee = df_clean['coffee_name']
y_price = df_clean['price_tier']

# print(f"Features: {len(all_features)}")
# print(f"Coffee classes: {y_coffee.nunique()}")
# print(f"Price tiers: {y_price.nunique()}")

# Split data once for both models
X_train, X_test, y_coffee_train, y_coffee_test, y_price_train, y_price_test = train_test_split(
    X, y_coffee, y_price, test_size=0.2, random_state=42, stratify=y_coffee
)

# print(f"Training set: {X_train.shape[0]} samples")
# print(f"Test set: {X_test.shape[0]} samples")

# print("=== Step 4: Training Dual Prediction Models ===")

# Model for coffee name prediction
coffee_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=10,
    random_state=42,
    class_weight='balanced'
)

# Model for price tier prediction  
price_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=15,
    random_state=42
)

# print("Training coffee prediction model...")
coffee_model.fit(X_train, y_coffee_train)

# print("Training price tier prediction model...")
price_model.fit(X_train, y_price_train)

# Evaluate both models
coffee_pred = coffee_model.predict(X_test)
price_pred = price_model.predict(X_test)

# print("\n=== MODEL PERFORMANCE ===")
# print("COFFEE PREDICTION:")
# print(f"Accuracy: {accuracy_score(y_coffee_test, coffee_pred):.3f}")

# print("\nPRICE TIER PREDICTION:")
# print(f"Accuracy: {accuracy_score(y_price_test, price_pred):.3f}")

# Feature importance
coffee_importance = pd.DataFrame({
    'feature': all_features,
    'importance': coffee_model.feature_importances_
}).sort_values('importance', ascending=False)

# print("\nTop 5 Features for Coffee Prediction:")
# print(coffee_importance.head(5))



# print("=== Step 5: Improving Model Performance ===")

# Let's first understand why coffee prediction is difficult
# print("\n=== ANALYSIS ===")
# print("Coffee distribution in test set:")
# print(y_coffee_test.value_counts())
# print(f"\nBaseline (predicting most frequent): {y_coffee_test.value_counts().max() / len(y_coffee_test):.3f}")

# print("\nPrice tier distribution in test set:")
# print(y_price_test.value_counts())
# print(f"Baseline: {y_price_test.value_counts().max() / len(y_price_test):.3f}")

print("=== Step 6: Implementing Model Improvements ===")

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Strategy 1: Handle class imbalance with better class weights
coffee_classes = y_coffee_train.unique()
coffee_weights = compute_class_weight(
    class_weight='balanced',
    classes=coffee_classes,
    y=y_coffee_train
)
coffee_class_weight_dict = dict(zip(coffee_classes, coffee_weights))

# print("Class weights for coffee prediction:")
for coffee, weight in coffee_class_weight_dict.items():
    print(f"  {coffee}: {weight:.2f}")

# Strategy 2: Improved Random Forest with better parameters
improved_coffee_model = RandomForestClassifier(
    n_estimators=200,           # More trees
    max_depth=20,               # Deeper trees
    min_samples_split=5,        # More flexible
    min_samples_leaf=2,         # More flexible  
    max_features='sqrt',        # Better feature selection
    class_weight=coffee_class_weight_dict,  # Handle imbalance
    random_state=42,
    bootstrap=True
)

improved_price_model = RandomForestClassifier(
    n_estimators=150,
    max_depth=15,
    min_samples_split=10,
    random_state=42
)

# print("\nTraining improved models...")
improved_coffee_model.fit(X_train, y_coffee_train)
improved_price_model.fit(X_train, y_price_train)

# Evaluate improved models
coffee_pred_improved = improved_coffee_model.predict(X_test)
price_pred_improved = improved_price_model.predict(X_test)

# print("\n=== IMPROVED MODEL PERFORMANCE ===")
# print("COFFEE PREDICTION:")
# print(f"Accuracy: {accuracy_score(y_coffee_test, coffee_pred_improved):.3f} (Before: 0.524)")

# print("\nPRICE TIER PREDICTION:")
# print(f"Accuracy: {accuracy_score(y_price_test, price_pred_improved):.3f} (Before: 0.730)")

# Detailed analysis
from sklearn.metrics import classification_report
# print("\nDetailed Coffee Prediction Report:")
# print(classification_report(y_coffee_test, coffee_pred_improved))

print("=== Step 7 (Fixed): Advanced Model Improvements ===")

from sklearn.preprocessing import LabelEncoder

# Strategy 3: Simplify the problem - Group rare coffees
print("Current coffee distribution:")
coffee_counts = y_coffee_train.value_counts()
print(coffee_counts)

# Create a simplified target by grouping rare coffees
def simplify_coffee_type(coffee_name):
    if coffee_name in ['Espresso', 'Cortado']:  # Group rare types
        return 'Rare_Espresso'
    elif coffee_name in ['Cocoa', 'Hot Chocolate']:
        return 'Chocolate_Drinks'
    else:
        return coffee_name  # Keep common types as-is

# Apply simplification and encode for XGBoost
y_coffee_simple_train = y_coffee_train.apply(simplify_coffee_type)
y_coffee_simple_test = y_coffee_test.apply(simplify_coffee_type)

# Encode the simplified coffee types for XGBoost
coffee_encoder = LabelEncoder()
y_coffee_simple_encoded_train = coffee_encoder.fit_transform(y_coffee_simple_train)
y_coffee_simple_encoded_test = coffee_encoder.transform(y_coffee_simple_test)

print("\nSimplified coffee distribution:")
print(y_coffee_simple_train.value_counts())
print(f"Original classes: {y_coffee_train.nunique()} → Simplified: {y_coffee_simple_train.nunique()}")
print(f"Encoded classes: {coffee_encoder.classes_}")

# Strategy 4: Try XGBoost with proper encoding
simple_coffee_model_xgb = XGBClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    random_state=42,
    eval_metric='mlogloss'
)

# Also encode price tier for XGBoost
price_encoder = LabelEncoder()
y_price_encoded_train = price_encoder.fit_transform(y_price_train)
y_price_encoded_test = price_encoder.transform(y_price_test)

simple_price_model_xgb = XGBClassifier(
    n_estimators=150,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)

print("\nTraining XGBoost models on simplified targets...")
simple_coffee_model_xgb.fit(X_train, y_coffee_simple_encoded_train)
simple_price_model_xgb.fit(X_train, y_price_encoded_train)

# Evaluate simplified models
coffee_pred_simple_encoded = simple_coffee_model_xgb.predict(X_test)
coffee_pred_simple = coffee_encoder.inverse_transform(coffee_pred_simple_encoded)

price_pred_xgb_encoded = simple_price_model_xgb.predict(X_test)
price_pred_xgb = price_encoder.inverse_transform(price_pred_xgb_encoded)

print("\n=== SIMPLIFIED MODEL PERFORMANCE ===")
print("COFFEE PREDICTION (Simplified):")
print(f"Accuracy: {accuracy_score(y_coffee_simple_test, coffee_pred_simple):.3f} (Before: 0.530)")
print(f"Classes: {y_coffee_simple_train.nunique()} vs original {y_coffee_train.nunique()}")

print("\nPRICE TIER PREDICTION (XGBoost):")
print(f"Accuracy: {accuracy_score(y_price_test, price_pred_xgb):.3f} (Before: 0.745)")

print("\nDetailed Simplified Coffee Prediction:")
print(classification_report(y_coffee_simple_test, coffee_pred_simple))