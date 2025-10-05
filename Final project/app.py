import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Load and clean data
df = pd.read_csv('coffe_sales.csv')
features_to_drop = ['Date', 'Time', 'Weekdaysort', 'Monthsort', 'Time_of_Day', 'cash_type']
df_clean = df.drop(columns=features_to_drop)

# Feature Engineering
df_clean['is_peak_morning'] = ((df_clean['hour_of_day'] >= 7) & (df_clean['hour_of_day'] <= 10)).astype(int)
df_clean['is_peak_afternoon'] = ((df_clean['hour_of_day'] >= 14) & (df_clean['hour_of_day'] <= 17)).astype(int)

def get_day_part(hour):
    if 5 <= hour <= 11: return 'Morning'
    elif 12 <= hour <= 16: return 'Afternoon'
    else: return 'Evening'

df_clean['day_part'] = df_clean['hour_of_day'].apply(get_day_part)
df_clean['is_weekend'] = (df_clean['Weekday'].isin(['Sat', 'Sun'])).astype(int)
df_clean['is_milk_based'] = df_clean['coffee_name'].str.contains('Latte|Cappuccino|Cortado', case=False, na=False).astype(int)
df_clean['is_espresso'] = df_clean['coffee_name'].str.contains('Espresso|Americano', case=False, na=False).astype(int)
df_clean['is_chocolate'] = df_clean['coffee_name'].str.contains('Chocolate|Cocoa', case=False, na=False).astype(int)
df_clean['is_cold_season'] = df_clean['Month_name'].isin(['Jan', 'Feb', 'Dec', 'Mar']).astype(int)
df_clean['is_warm_season'] = df_clean['Month_name'].isin(['Jun', 'Jul', 'Aug']).astype(int)
df_clean['price_tier'] = pd.cut(df_clean['money'], bins=[0, 25, 30, 35, float('inf')], labels=['Budget', 'Standard', 'Premium', 'Luxury'])

# Encode categorical features
label_encoders = {}
categorical_features = ['Weekday', 'Month_name', 'day_part']
for col in categorical_features:
    le = LabelEncoder()
    df_clean[col + '_encoded'] = le.fit_transform(df_clean[col])
    label_encoders[col] = le

numerical_features = ['hour_of_day', 'is_peak_morning', 'is_peak_afternoon', 'is_weekend', 'is_milk_based', 'is_espresso', 'is_chocolate', 'is_cold_season', 'is_warm_season']
all_features = numerical_features + [f"{col}_encoded" for col in categorical_features]

# Prepare data for modeling
X = df_clean[all_features]
y_coffee = df_clean['coffee_name']
y_price = df_clean['price_tier']

X_train, X_test, y_coffee_train, y_coffee_test, y_price_train, y_price_test = train_test_split(
    X, y_coffee, y_price, test_size=0.2, random_state=42, stratify=y_coffee
)

# === IMPROVED COFFEE GROUPING - FIXING OUTLIERS ===
# print("Applying improved coffee grouping strategy...")

def improved_coffee_grouping(coffee_name):
    """
    Better grouping based on drink characteristics and business logic
    """
    if coffee_name == 'Espresso':
        return 'Pure_Espresso'  # Keep separate - unique category
    elif coffee_name in ['Cocoa', 'Hot Chocolate']:
        return 'Chocolate_Drinks'
    elif coffee_name in ['Cortado', 'Latte', 'Cappuccino']:
        return 'Milk_Based_Drinks'  # Cortado belongs with milk drinks
    elif coffee_name in ['Americano', 'Americano with Milk']:
        return 'Americano_Family'  # Americano variations together
    else:
        return coffee_name

# Apply improved grouping
y_coffee_improved_train = y_coffee_train.apply(improved_coffee_grouping)
y_coffee_improved_test = y_coffee_test.apply(improved_coffee_grouping)

# print("Improved coffee distribution:")
# print(y_coffee_improved_train.value_counts())
# print(f"Classes: {y_coffee_improved_train.nunique()}")

# Update the coffee encoder with improved groups
coffee_encoder = LabelEncoder()
y_coffee_improved_encoded_train = coffee_encoder.fit_transform(y_coffee_improved_train)
y_coffee_improved_encoded_test = coffee_encoder.transform(y_coffee_improved_test)

# print("Encoded classes:", coffee_encoder.classes_)

# Train XGBoost models (use the already encoded data)
price_encoder = LabelEncoder()
y_price_encoded_train = price_encoder.fit_transform(y_price_train)
y_price_encoded_test = price_encoder.transform(y_price_test)

simple_coffee_model_xgb = XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.1, random_state=42, eval_metric='mlogloss')
simple_price_model_xgb = XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1, random_state=42)

print("Training final XGBoost models...")
simple_coffee_model_xgb.fit(X_train, y_coffee_improved_encoded_train)  # Use existing encoding
simple_price_model_xgb.fit(X_train, y_price_encoded_train)

coffee_pred_simple_encoded = simple_coffee_model_xgb.predict(X_test)

print("Training final XGBoost models...")
simple_coffee_model_xgb.fit(X_train, y_coffee_improved_encoded_train)  # ✅ Use training labels
simple_price_model_xgb.fit(X_train, y_price_encoded_train)

# Evaluate final models
coffee_pred_simple = coffee_encoder.inverse_transform(coffee_pred_simple_encoded)
price_pred_xgb_encoded = simple_price_model_xgb.predict(X_test)
price_pred_xgb = price_encoder.inverse_transform(price_pred_xgb_encoded)

# print("\n=== FINAL MODEL PERFORMANCE ===")
# print("COFFEE PREDICTION:")
# print(f"Accuracy: {accuracy_score(y_coffee_improved_encoded_test, coffee_pred_simple):.3f}")
# # print(f"Classes: {y_coffee_improved_encoded_train.nunique()} vs original {y_coffee_train.nunique()}")
# print(f"Classes: {y_coffee_improved_train.nunique()} vs original {y_coffee_train.nunique()}")

# print("\nPRICE TIER PREDICTION:")
# print(f"Accuracy: {accuracy_score(y_price_test, price_pred_xgb):.3f}")

# print("\nDetailed Coffee Prediction Report:")
# print(classification_report(y_coffee_improved_encoded_test, coffee_pred_simple))

def compare_models(X_train_data, X_test_data, y_train_labels, y_test_labels, target_name):
    """
    Comprehensive model comparison for classification tasks
    Returns DataFrame with metrics for all models
    """   
    
    models = {
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
        'RandomForest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='mlogloss'),
        'NaiveBayes': GaussianNB(),
        'SVM': SVC(random_state=42),
        'KNN': KNeighborsClassifier()
    }
    
    results = []
    
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        
        # Training time
        start_train = time.time()
        model.fit(X_train_data, y_train_labels)  # Use the new parameter names
        train_time = time.time() - start_train
        
        # Prediction time
        start_pred = time.time()
        y_pred = model.predict(X_test_data)  # Use the new parameter names
        pred_time = (time.time() - start_pred) / len(X_test_data) * 1000  # ms per prediction
        
        # Metrics
        accuracy = accuracy_score(y_test_labels, y_pred)
        precision = precision_score(y_test_labels, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_test_labels, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test_labels, y_pred, average='macro', zero_division=0)
        
        results.append({
            'Model': model_name,
            'Target': target_name,
            'Accuracy': round(accuracy, 4),
            'Precision': round(precision, 4),
            'Recall': round(recall, 4),
            'F1-Score': round(f1, 4),
            'Train_Time_sec': round(train_time, 4),
            'Pred_Time_ms': round(pred_time, 4)
        })
    
    return pd.DataFrame(results)

# Test the function
print("=== COMPREHENSIVE MODEL COMPARISON ===")

# Compare for Coffee Groups
print("=== DEBUG SHAPES ===")
print(f"X_train shape: {X_train.shape}")
print(f"y_coffee_improved_encoded_train shape: {y_coffee_improved_encoded_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_coffee_improved_encoded_test shape: {y_coffee_improved_encoded_test.shape}")
# Compare for Coffee Groups
print("\n🔍 COMPARING MODELS FOR COFFEE GROUPS PREDICTION:")
coffee_comparison = compare_models(
    X_train, 
    X_test,
    y_coffee_improved_encoded_train, 
    y_coffee_improved_encoded_test,
    'Coffee_Groups'
)

# Compare for Price Tiers  
print("\n💰 COMPARING MODELS FOR PRICE TIER PREDICTION:")
price_comparison = compare_models(
    X_train,
    X_test,
    y_price_encoded_train,
    y_price_encoded_test,
    'Price_Tiers'
)

# Combine results and display
all_results = pd.concat([coffee_comparison, price_comparison], ignore_index=True)

print("\n🎯 FINAL MODEL COMPARISON RESULTS:")
print("=" * 80)

# Show best models for each target
best_coffee = coffee_comparison.loc[coffee_comparison['Accuracy'].idxmax()]
best_price = price_comparison.loc[price_comparison['Accuracy'].idxmax()]

print(f"\n🏆 BEST FOR COFFEE GROUPS: {best_coffee['Model']}")
print(f"   Accuracy: {best_coffee['Accuracy']}, F1-Score: {best_coffee['F1-Score']}")

print(f"\n🏆 BEST FOR PRICE TIERS: {best_price['Model']}")  
print(f"   Accuracy: {best_price['Accuracy']}, F1-Score: {best_price['F1-Score']}")

print("\n📊 DETAILED RESULTS:")
print(all_results.sort_values(['Target', 'Accuracy'], ascending=[True, False]))

# Save results for reference
all_results.to_csv('model_comparison_results.csv', index=False)
print("\n💾 Results saved to 'model_comparison_results.csv'")