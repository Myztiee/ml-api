"""
Flask API for Student Completion Prediction Model
Deployed on Railway
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import json
import os
import logging
import shap
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import base64
import io
import sys

# Force unbuffered stdout (prints appear immediately in Railway logs)
sys.stdout.reconfigure(line_buffering=True, write_through=True)

# Configure logging to always show in Railway logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

print("🚀 App is starting up...")
logger.info("🚀 App is starting up...")

app = Flask(__name__)

# Configure CORS - Update with your domain in production
CORS(app, resources={
    r"/api/*": {
        "origins": ["*"],  # Replace with your domain in production
        "methods": ["POST", "GET"],
        "allow_headers": ["Content-Type", "X-API-Key"]
    }
})

# Configuration
API_KEY = os.environ.get('API_KEY', 'your-secure-api-key-here')  # Set via Railway environment variables
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'trained_model.pkl')
FEATURES_PATH = os.path.join(BASE_DIR, 'feature_columns.pkl')
THRESHOLD = 0.50

print("🔍 Checking model path:", os.path.abspath(MODEL_PATH))
print("📁 Files in current directory:", os.listdir(os.path.dirname(os.path.abspath(__file__))))

# Global variables for model caching
model = None
feature_columns = None
model_metrics = None

# Load model immediately when module is imported (for Gunicorn)
logger.info("🔄 Attempting to load model at module import...")

# =====================================================
# MODEL LOADING
# =====================================================

def load_model():
    """Load the trained model and feature columns"""
    global model, feature_columns, model_metrics
    
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(FEATURES_PATH):
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            with open(FEATURES_PATH, 'rb') as f:
                feature_columns = pickle.load(f)
            
            # Load metrics if available
            if os.path.exists('model_metadata.json'):
                with open('model_metadata.json', 'r') as f:
                    metadata = json.load(f)
                    model_metrics = metadata.get('metrics', {})
            
            logger.info("✅ Model loaded successfully")
            print("✅ Model successfully loaded!")
            return True
        else:
            print(f"❌ Model files not found - MODEL_PATH: {os.path.exists(MODEL_PATH)}, FEATURES_PATH: {os.path.exists(FEATURES_PATH)}")
            logger.error("❌ Model files not found")
            return False
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        logger.error(f"Error loading model: {e}")
        return False

if not load_model():
    logger.error("⚠️ Model failed to load at startup - predictions will fail!")
else:
    logger.info("✅ Model loaded successfully at startup!")
    

# =====================================================
# DATA PREPROCESSING (from your RandomForestAlgorithm.py)
# =====================================================

COLUMN_MAPPING = {
    'Learner Reference No. (LRN) ex. 136743': 'LRN',
    'Monthly Income': 'MonthlyIncome',
    'Nutritional Status': 'NutritionalStatus',
    'Study Habits': 'StudyHabits',
    'Sleeping Habits': 'SleepingHabits',
    'Father Educational Attainment': 'FatherEducationalAttainment',
    'Mother Educational Attainment': 'MotherEducationalAttainment',
    'Number of Siblings': 'NumberOfSiblings',
    'Access to technology': 'AccessToTechnology'
}

CATEGORICAL_COLUMNS = [
    'Sex', 'Proximity', 'NutritionalStatus',
    'FatherEducationalAttainment', 'MotherEducationalAttainment',
    'FamilyFinancialStatus', 'AccessToTechnology', 'Extracurricular'
]

NUMERIC_COLUMNS = [
    'Age', 'MonthlyIncome', 'Grade1_Average', 'Grade2_Average', 'Grade3_Average', 'Grade4_Average', 'Grade5_Average',
    'Grade6_CurrentGrade', 'StudyHabits', 'SleepingHabits', 'NumberOfSiblings'
]

def categorize_income(income):
    """Categorize monthly income"""
    if pd.isna(income):
        return 'Low income'
    
    try:
        income_val = float(income)
    except (ValueError, TypeError):
        return 'Low income'
    
    if income_val <= 10000:
        return 'Low income'
    elif income_val <= 25000:
        return 'Lower middle class'
    elif income_val <= 50000:
        return 'Middle class'
    elif income_val <= 100000:
        return 'Upper middle class'
    else:
        return 'High income'

def preprocess_data(df):
    """Preprocess data for prediction"""
    df = df.copy()
    
    # Apply column mapping
    df = df.rename(columns=COLUMN_MAPPING)
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Handle numeric columns
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
    
    # Handle missing Grade6_CurrentGrade
    grade_cols = [c for c in ['Grade1_Average', 'Grade2_Average', 'Grade3_Average', 'Grade4_Average', 'Grade5_Average'] if c in df.columns]
    if grade_cols:
        if 'Grade6_CurrentGrade' not in df.columns:
            df['Grade6_CurrentGrade'] = df[grade_cols].median(axis=1)
        else:
            missing_mask = df['Grade6_CurrentGrade'].isna()
            if missing_mask.any():
                df.loc[missing_mask, 'Grade6_CurrentGrade'] = df.loc[missing_mask, grade_cols].median(axis=1)
    
    # Handle FamilyFinancialStatus
    if 'FamilyFinancialStatus' not in df.columns and 'MonthlyIncome' in df.columns:
        df['FamilyFinancialStatus'] = df['MonthlyIncome'].apply(categorize_income)
    
    # Handle missing values in categorical columns
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            if df[col].isnull().sum() > 0:
                mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                df[col] = df[col].fillna(mode_val)
    
    # One-hot encode categorical variables
    categorical_cols_to_encode = [col for col in CATEGORICAL_COLUMNS if col in df.columns]
    if categorical_cols_to_encode:
        df = pd.get_dummies(df, columns=categorical_cols_to_encode, drop_first=True)
    
    # Drop LRN if present
    if 'LRN' in df.columns:
        df = df.drop(columns=['LRN'])
    
    return df

def calculate_completion_score(row):
    """Calculate completion score for a student"""
    weights = {
        'academic': 0.25,
        'habits': 0.20,
        'family': 0.20,
        'tech_extracurricular': 0.15,
        'proximity': 0.10,
        'health_demo': 0.10
    }
    
    total = 0.0
    
    # Academic
    grades = [row.get(f'Grade{i}_Average', np.nan) for i in range(1, 6)]
    grades = [g for g in grades if pd.notna(g)]
    if grades:
        avg_grade = np.mean(grades)
        curr = row.get('Grade6_CurrentGrade', np.nan)
        dynamic_threshold = np.mean(grades)
        acad_score = 0.0
        if avg_grade > dynamic_threshold:
            acad_score += 0.5
        if pd.notna(curr) and curr > dynamic_threshold:
            acad_score += 0.5
        total += weights['academic'] * acad_score
    
    # Study & Sleeping Habits
    study = row.get('StudyHabits', 0)
    sleep = row.get('SleepingHabits', 0)
    habit_score = ((study >= 3) + (sleep >= 3)) / 2
    total += weights['habits'] * habit_score
    
    # Family Background
    family_score = 0.0
    financial = row.get('FamilyFinancialStatus', '')
    if financial in ['Middle class', 'Upper middle class', 'High income']:
        family_score += 1/4
    income = row.get('MonthlyIncome', 0)
    if pd.notna(income) and income >= 15000:
        family_score += 1/4
    if row.get('FatherEducationalAttainment') in ['Secondary', 'Tertiary']:
        family_score += 1/4
    if row.get('MotherEducationalAttainment') in ['Secondary', 'Tertiary']:
        family_score += 1/4
    total += weights['family'] * family_score
    
    # Technology & Extracurricular
    tech_str = str(row.get('AccessToTechnology', '')).lower()
    tech_items = ['laptop', 'cellphone', 'computer', 'tablet']
    tech_count = sum(1 for item in tech_items if item in tech_str)
    tech_score = tech_count / len(tech_items)
    extra_flag = str(row.get('Extracurricular', '')).lower() == 'yes'
    combined_score = (tech_score + float(extra_flag)) / 2
    total += weights['tech_extracurricular'] * combined_score
    
    # Proximity
    proximity = str(row.get('Proximity', '')).lower()
    if proximity == 'near':
        proximity_score = 1.0
    elif proximity == 'far':
        proximity_score = 0.0
    else:
        proximity_score = 0.5
    total += weights['proximity'] * proximity_score
    
    # Health & Demographics
    health_demo_score = 0.0
    nutrition = str(row.get('NutritionalStatus', '')).lower()
    if nutrition in ['normal', 'healthy']:
        health_demo_score += 0.4
    elif nutrition in ['underweight', 'overweight']:
        health_demo_score += 0.2
    siblings = row.get('NumberOfSiblings', 0)
    if pd.notna(siblings):
        if siblings <= 3:
            health_demo_score += 0.4
        elif siblings <= 6:
            health_demo_score += 0.2
    age = row.get('Age', 0)
    if pd.notna(age) and 10 <= age <= 13:
        health_demo_score += 0.2
    sex = str(row.get('Sex', '')).lower()
    if sex in ['female', 'f']:
        health_demo_score += 0.05
    total += weights['health_demo'] * min(1.0, health_demo_score)
    
    return float(max(0.0, min(1.0, total)))

# =====================================================
# API ENDPOINTS
# =====================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/model/info', methods=['GET'])
def model_info():
    """Get model information"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    return jsonify({
        'model_type': 'RandomForestClassifier',
        'features_count': len(feature_columns) if feature_columns else 0,
        'threshold': THRESHOLD,
        'metrics': model_metrics,
        'version': '1.0.0'
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint
    
    Request body:
    {
        "csv_data": "base64_encoded_csv_content",
        "threshold": 0.5 (optional),
        "include_shap": true (optional)
    }
    """
    try:
        # Validate API key
        api_key = request.headers.get('X-API-Key')
        if api_key != API_KEY:
            return jsonify({'error': 'Invalid API key'}), 401
        
        # Check if model is loaded
        if model is None or feature_columns is None:
            return jsonify({'error': 'Model not loaded'}), 503
        
        # Parse request
        data = request.get_json()
        if not data or 'csv_data' not in data:
            return jsonify({'error': 'Missing csv_data in request'}), 400
        
        # Decode CSV data
        csv_base64 = data['csv_data']
        csv_content = base64.b64decode(csv_base64).decode('utf-8')
        
        # Load CSV into DataFrame
        df_original = pd.read_csv(io.StringIO(csv_content))
        
        # Store original identifiers BEFORE preprocessing (use original column name)
        lrn_column = None
        possible_lrn_columns = [
            'LRN',
            'Learner Reference No. (LRN) ex. 136743',
            'Learner Reference No.',
            'learner_reference_no'
        ]

        for col in possible_lrn_columns:
            if col in df_original.columns:
                lrn_column = col
                break

        original_lrns = df_original[lrn_column].tolist() if lrn_column else []
        original_ages = df_original['Age'].tolist() if 'Age' in df_original.columns else []
        
        # Preprocess data
        df_processed = preprocess_data(df_original)

        # Remove 'CompletionRate' if present
        if 'CompletionRate' in df_processed.columns:
            df_processed = df_processed.drop(columns=['CompletionRate'])
            logger.info("Removed 'CompletionRate' column before prediction")
        
        # Align columns with trained model
        X = df_processed.reindex(columns=feature_columns, fill_value=0)

        logger.info("=" * 80)
        logger.info("FEATURE ALIGNMENT DIAGNOSTIC")
        logger.info("=" * 80)

        # 1. Check shape match
        logger.info(f"Expected features: {len(feature_columns)}")
        logger.info(f"Actual features: {X.shape[1]}")
        logger.info(f"Number of students: {X.shape[0]}")

        # 2. Find LRN_1040 row for debugging
        lrn_1040_idx = None
        for idx, lrn in enumerate(original_lrns):
            if str(lrn) == "LRN_1040" or "1040" in str(lrn):
                lrn_1040_idx = idx
                break

        if lrn_1040_idx is not None:
            logger.info(f"\n🔍 DEBUGGING LRN_1040 (Row {lrn_1040_idx}):")
            
            # 3. Check for NaN or zero-filled columns
            student_data = X.iloc[lrn_1040_idx]
            nan_count = student_data.isna().sum()
            zero_count = (student_data == 0).sum()
            
            logger.info(f"NaN values: {nan_count}")
            logger.info(f"Zero-filled values: {zero_count}/{len(student_data)}")
            
            # 4. Print Grade columns specifically
            grade_cols = [col for col in X.columns if 'Grade' in col]
            logger.info(f"\nGrade columns found: {len(grade_cols)}")
            for col in grade_cols:
                logger.info(f"  {col}: {student_data[col]}")
            
            # 5. Print categorical columns (one-hot encoded)
            categorical_encoded = [col for col in X.columns if any(cat in col for cat in ['Sex_', 'Proximity_', 'Financial_', 'Father_', 'Mother_', 'Technology_', 'Extracurricular_'])]
            logger.info(f"\nCategorical columns found: {len(categorical_encoded)}")
            active_categories = [col for col in categorical_encoded if student_data[col] == 1]
            logger.info(f"Active categories: {active_categories}")
            
            # 6. Compare with feature_columns order
            if len(X.columns) != len(feature_columns):
                logger.error("❌ COLUMN COUNT MISMATCH!")
                missing_in_X = set(feature_columns) - set(X.columns)
                extra_in_X = set(X.columns) - set(feature_columns)
                if missing_in_X:
                    logger.error(f"Missing in X: {missing_in_X}")
                if extra_in_X:
                    logger.error(f"Extra in X: {extra_in_X}")
            else:
                column_order_match = list(X.columns) == feature_columns
                logger.info(f"Column order match: {column_order_match}")
                if not column_order_match:
                    logger.warning("⚠️ Column order differs from training!")

        # 7. Make prediction with logging
        logger.info("\n📊 Running model prediction...")
        y_pred_proba = model.predict_proba(X)[:, 1]

        if lrn_1040_idx is not None:
            logger.info(f"LRN_1040 probability: {y_pred_proba[lrn_1040_idx]:.4f}")
            logger.info(f"Expected probability: ~0.7369")
            logger.info(f"Difference: {abs(y_pred_proba[lrn_1040_idx] - 0.7369):.4f}")

        logger.info("=" * 80)
        
        # Get threshold
        threshold = data.get('threshold', THRESHOLD)
        
        # Make predictions
        y_pred_proba = model.predict_proba(X)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate SHAP values if requested
        shap_matrix = None
        if data.get('include_shap', True):
            try:
                logger.info("Computing SHAP values...")
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
                
                # Handle different SHAP output formats
                if isinstance(shap_values, list):
                    shap_matrix = np.array(shap_values[1])
                elif isinstance(shap_values, np.ndarray):
                    if shap_values.ndim == 3:
                        shap_matrix = shap_values[:, :, 1]
                    else:
                        shap_matrix = shap_values
                else:
                    shap_matrix = np.array(shap_values)
                
                logger.info(f"✅ SHAP values computed: {shap_matrix.shape}")
            except Exception as e:
                logger.warning(f"⚠️ SHAP computation failed: {e}")
        
        # Calculate completion scores
        completion_scores = []
        for _, row in df_original.iterrows():
            score = calculate_completion_score(row)
            completion_scores.append(score)
        
        # Format results
        results = []
        for i in range(len(y_pred)):
            # Format LRN (handle scientific notation from Excel)
            raw_lrn = original_lrns[i] if i < len(original_lrns) else f"STU{i+1:03d}"

            # Convert scientific notation to full number
            if isinstance(raw_lrn, (float, np.floating)):
                lrn = f"{raw_lrn:.0f}"
            elif isinstance(raw_lrn, str):
                try:
                    # Try to parse as float in case it's a string 
                    lrn = f"{float(raw_lrn):.0f}"
                except ValueError:
                    lrn = raw_lrn
            else:
                lrn = str(raw_lrn)

            age = original_ages[i] if i < len(original_ages) else None
            
            # Compute grouped SHAP factors
            top_factors = []
            if shap_matrix is not None:
                try:
                    shap_values_row = shap_matrix[i]
                    shap_df = pd.DataFrame({
                        "feature": X.columns,
                        "impact": shap_values_row
                    })
                    
                    # Group by base feature
                    # Preserve Grade columns, group others
                    def get_base_feature(f):
                        if f.startswith('Grade'):
                            return f  # Keep full name like "Grade1_Average"
                        else:
                            return f.split("_")[0]  # Group one-hot encoded 

                    shap_df["base_feature"] = shap_df["feature"].apply(get_base_feature)
                    
                    grouped = (
                        shap_df.groupby("base_feature")["impact"]
                        .agg(lambda x: np.mean(x))
                        .reset_index()
                    )
                    
                    # Filter based on prediction
                    if int(y_pred[i]) == 0:  # At Risk
                        grouped = grouped[grouped["impact"] < 0].sort_values("impact", ascending=True)
                    else:  # Will Complete
                        grouped = grouped[grouped["impact"] > 0].sort_values("impact", ascending=False)
                    
                    top_factors = grouped.head(5).to_dict(orient="records")
                except Exception as e:
                    logger.warning(f"Error processing SHAP for student {i}: {e}")
            
            results.append({
                "lrn": str(lrn),
                "age": int(age) if age and pd.notna(age) else None,
                "result": int(y_pred[i]),
                "prediction": "Likely to Complete" if y_pred[i] == 1 else "At Risk",
                "probability": float(y_pred_proba[i]),
                "score": float(completion_scores[i]),
                "factors": top_factors
            })
        
        # Summary statistics
        at_risk_count = int(sum(1 for pred in y_pred if pred == 0))
        likely_to_complete_count = int(sum(1 for pred in y_pred if pred == 1))
        
        return jsonify({
            "success": True,
            "predictions": results,
            "summary": {
                "total_students": len(results),
                "at_risk_count": at_risk_count,
                "likely_to_complete_count": likely_to_complete_count,
                "at_risk_percentage": float(at_risk_count / len(results) * 100),
                "likely_to_complete_percentage": float(likely_to_complete_count / len(results) * 100),
                "avg_probability": float(sum(y_pred_proba) / len(y_pred_proba))
            }
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500

# =====================================================
# INITIALIZATION
# =====================================================

if __name__ == '__main__':
    logger.info("🚀 Starting ML Prediction API...")
    
    # Load model on startup
    if not load_model():
        logger.error("❌ Failed to load model. Please ensure model files exist.")
    else:
        logger.info("✅ Model loaded successfully")
    
    # Get port from environment variable (Railway sets this)
    port = int(os.environ.get('PORT', 5000))
    
    # Run Flask app
    app.run(host='0.0.0.0', port=port, debug=False)