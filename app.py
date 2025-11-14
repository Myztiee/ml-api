"""
Flask API for Student Completion Prediction Model
Deployed on Railway - Synchronized with RandomForestAlgorithm_new.py
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

# =====================================================
# CONFIGURATION
# =====================================================
API_KEY = os.environ.get('API_KEY', 'your-secure-api-key-here')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'trained_model.pkl')
FEATURES_PATH = os.path.join(BASE_DIR, 'feature_columns.pkl')
METADATA_PATH = os.path.join(BASE_DIR, 'model_metadata.json')
THRESHOLD = 0.50
RANDOM_STATE = 42

# Class labels
CLASS_LABELS = {
    0: "At Risk",
    1: "Likely to Complete"
}

print("📁 Checking model path:", os.path.abspath(MODEL_PATH))
print("📁 Files in current directory:", os.listdir(os.path.dirname(os.path.abspath(__file__))))

# Global variables for model caching
model = None
feature_columns = None
model_metrics = None
label_threshold = None

# =====================================================
# DATA CONFIGURATION (from RandomForestAlgorithm_new.py)
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
    'Access to technology': 'AccessToTechnology',
    'Grade1': 'Grade1_Average',
    'Grade2': 'Grade2_Average',
    'Grade3': 'Grade3_Average',
    'Grade4': 'Grade4_Average',
    'Grade5': 'Grade5_Average',
    'CurrentGrade': 'Grade6_CurrentGrade'
}

PREDICTOR_COLUMNS = [
    'LRN', 'Age', 'Sex', 'Proximity', 'MonthlyIncome',
    'Grade1_Average', 'Grade2_Average', 'Grade3_Average', 'Grade4_Average', 'Grade5_Average', 'Grade6_CurrentGrade',
    'NutritionalStatus', 'StudyHabits', 'SleepingHabits',
    'FatherEducationalAttainment', 'MotherEducationalAttainment',
    'FamilyFinancialStatus', 'NumberOfSiblings', 'AccessToTechnology',
    'Extracurricular'
]

CATEGORICAL_COLUMNS = [
    'Sex', 'Proximity', 'NutritionalStatus',
    'FatherEducationalAttainment', 'MotherEducationalAttainment',
    'FamilyFinancialStatus', 'AccessToTechnology', 'Extracurricular'
]

NUMERIC_COLUMNS = [
    'Age', 'MonthlyIncome', 'Grade1_Average', 'Grade2_Average', 'Grade3_Average', 'Grade4_Average', 'Grade5_Average',
    'Grade6_CurrentGrade', 'StudyHabits', 'SleepingHabits', 'NumberOfSiblings'
]

# =====================================================
# MODEL LOADING
# =====================================================

def load_model():
    """Load the trained model and feature columns"""
    global model, feature_columns, model_metrics, label_threshold
    
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(FEATURES_PATH):
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            with open(FEATURES_PATH, 'rb') as f:
                feature_columns = pickle.load(f)
            
            # Load metadata if available
            if os.path.exists(METADATA_PATH):
                with open(METADATA_PATH, 'r') as f:
                    metadata = json.load(f)
                    model_metrics = metadata.get('metrics', {})
                    label_threshold = metadata.get('label_threshold')
                    logger.info(f"Loaded label_threshold: {label_threshold}")
            
            logger.info("✅ Model loaded successfully")
            logger.info(f"Feature count: {len(feature_columns)}")
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

# Load model at startup
if not load_model():
    logger.error("⚠️ Model failed to load at startup - predictions will fail!")
else:
    logger.info("✅ Model loaded successfully at startup!")

# =====================================================
# DATA PREPROCESSING FUNCTIONS
# =====================================================

def categorize_income(income):
    """
    Categorize monthly income into financial status categories
    Synchronized with RandomForestAlgorithm_new.py
    """
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
    """
    Preprocess data for prediction
    Fully synchronized with RandomForestAlgorithm_new.py
    """
    df = df.copy()
    
    logger.info("Starting data preprocessing...")
    
    # Clean column names - remove extra spaces
    df.columns = df.columns.str.strip()
    
    # Apply column mapping
    df = df.rename(columns=COLUMN_MAPPING)
    
    # Convert numeric columns to proper data types
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Handle the complex Study Habits and Sleeping Habits columns
    study_cols = [col for col in df.columns if 'Study Habits' in col and col != 'StudyHabits']
    sleep_cols = [col for col in df.columns if 'Sleeping Habits' in col and col != 'SleepingHabits']
    
    if study_cols:
        study_numeric_cols = []
        for col in study_cols:
            if df[col].dtype in ['int64', 'float64']:
                study_numeric_cols.append(col)
        
        if study_numeric_cols:
            df['StudyHabits'] = df[study_numeric_cols].mean(axis=1)
            df = df.drop(columns=study_numeric_cols)
    
    if sleep_cols:
        sleep_numeric_cols = []
        for col in sleep_cols:
            if df[col].dtype in ['int64', 'float64']:
                sleep_numeric_cols.append(col)
        
        if sleep_numeric_cols:
            df['SleepingHabits'] = df[sleep_numeric_cols].mean(axis=1)
            df = df.drop(columns=sleep_numeric_cols)
    
    # Clean up any remaining unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    # Handle missing FamilyFinancialStatus column (infer from MonthlyIncome)
    if 'FamilyFinancialStatus' not in df.columns and 'MonthlyIncome' in df.columns:
        df['FamilyFinancialStatus'] = df['MonthlyIncome'].apply(categorize_income)
        logger.info("Created FamilyFinancialStatus from MonthlyIncome")
    
    # Handle missing values in numeric columns
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                logger.info(f"Filled missing values in {col} with median: {median_val}")
    
    # Handle missing Grade6_CurrentGrade
    grade_cols = [c for c in ['Grade1_Average', 'Grade2_Average', 'Grade3_Average', 'Grade4_Average', 'Grade5_Average'] if c in df.columns]
    if grade_cols:
        if 'Grade6_CurrentGrade' not in df.columns:
            df['Grade6_CurrentGrade'] = df[grade_cols].median(axis=1)
            logger.info("Created 'Grade6_CurrentGrade' using median of Grade1_Average-Grade5_Average.")
        else:
            missing_mask = df['Grade6_CurrentGrade'].isna()
            if missing_mask.any():
                df.loc[missing_mask, 'Grade6_CurrentGrade'] = df.loc[missing_mask, grade_cols].median(axis=1)
                logger.info(f"Filled {missing_mask.sum()} missing 'Grade6_CurrentGrade' values using median of Grade1_Average-Grade5_Average.")
    
    # Handle missing values in categorical columns
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            if df[col].isnull().sum() > 0:
                mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                df[col] = df[col].fillna(mode_val)
                logger.info(f"Filled missing values in {col} with mode: {mode_val}")
    
    # One-hot encode categorical variables
    categorical_cols_to_encode = [col for col in CATEGORICAL_COLUMNS if col in df.columns]
    if categorical_cols_to_encode:
        df = pd.get_dummies(df, columns=categorical_cols_to_encode, drop_first=True)
        logger.info(f"One-hot encoded categorical columns: {categorical_cols_to_encode}")
    
    # Drop LRN column if present (identifier, not a feature)
    if 'LRN' in df.columns:
        df = df.drop(columns=['LRN'])
        logger.info("Dropped LRN column (identifier)")
    
    logger.info(f"Preprocessing complete. Final shape: {df.shape}")
    return df


def compute_completion_scores(df):
    """
    Compute the weighted completion score based on RRL-supported predictors
    Synchronized with RandomForestAlgorithm_new.py
    """
    df = df.copy()
    
    # Fill in missing expected columns safely
    default_values = {
        'StudyHabits': 0,
        'SleepingHabits': 0,
        'AccessToTechnology': '',
        'Extracurricular': 'No',
        'FamilyFinancialStatus': 'Low income',
        'FatherEducationalAttainment': 'Primary',
        'MotherEducationalAttainment': 'Primary',
        'Proximity': 'Unknown',
        'NutritionalStatus': 'Unknown',
        'Age': 0,
        'NumberOfSiblings': 0,
        'Sex': 'Unknown',
        'MonthlyIncome': 0,
        'Grade6_CurrentGrade': df[[c for c in df.columns if c in ['Grade1_Average','Grade2_Average','Grade3_Average','Grade4_Average','Grade5_Average']]].mean(axis=1) if any(c in df.columns for c in ['Grade1_Average','Grade2_Average','Grade3_Average','Grade4_Average','Grade5_Average']) else 0,
    }
    
    for col, val in default_values.items():
        if col not in df.columns:
            df[col] = val
    
    # Define weights
    weights = {
        'academic': 0.25,
        'habits': 0.20,
        'family': 0.20,
        'tech_extracurricular': 0.15,
        'proximity': 0.10,
        'health_demo': 0.10
    }
    
    df['completion_score'] = 0.0
    
    # Academic (25%)
    grade_cols = [c for c in ['Grade1_Average', 'Grade2_Average', 'Grade3_Average', 'Grade4_Average', 'Grade5_Average'] if c in df.columns]
    if grade_cols:
        df['avg_grade'] = df[grade_cols].mean(axis=1)
        dynamic_threshold = df['avg_grade'].mean()
        if 'Grade6_CurrentGrade' in df.columns:
            cg_series = df['Grade6_CurrentGrade']
        elif 'CurrentGrade' in df.columns:
            cg_series = df['CurrentGrade']
        else:
            cg_series = df['avg_grade']
        academic_points = (
            (df['avg_grade'] > dynamic_threshold).astype(float)
            + (cg_series > dynamic_threshold).astype(float)
        ) / 2
        df['completion_score'] += weights['academic'] * academic_points
    
    # Study & Sleeping Habits (20%)
    df['habit_score'] = (
        (df['StudyHabits'] >= 3).astype(float) +
        (df['SleepingHabits'] >= 3).astype(float)
    ) / 2
    df['completion_score'] += weights['habits'] * df['habit_score']
    
    # Family Background (20%)
    df['family_score'] = (
        df['FamilyFinancialStatus'].isin(['Middle class', 'Upper middle class', 'High income']).astype(float)
        + (df['MonthlyIncome'] >= 15000).astype(float)
        + df['FatherEducationalAttainment'].isin(['Secondary', 'Tertiary']).astype(float)
        + df['MotherEducationalAttainment'].isin(['Secondary', 'Tertiary']).astype(float)
    ) / 4
    df['completion_score'] += weights['family'] * df['family_score']
    
    # Technology & Extracurricular (15%)
    tech_items = ['laptop', 'cellphone', 'computer', 'tablet']
    df['tech_score'] = df['AccessToTechnology'].apply(
        lambda x: sum(item in str(x).lower() for item in tech_items) / len(tech_items)
    )
    df['extra_score'] = (df['Extracurricular'].str.lower() == 'yes').astype(float)
    df['tech_extra_score'] = (df['tech_score'] + df['extra_score']) / 2
    df['completion_score'] += weights['tech_extracurricular'] * df['tech_extra_score']
    
    # Proximity to School (10%)
    df['proximity_points'] = df['Proximity'].str.lower().map({'near': 1.0, 'far': 0.0}).fillna(0.5)
    df['completion_score'] += weights['proximity'] * df['proximity_points']
    
    # Health & Demographics (10%)
    health_demo_scores = []
    for _, row in df.iterrows():
        score = 0.0
        nutrition = str(row['NutritionalStatus']).lower()
        if nutrition in ['normal', 'healthy']:
            score += 0.4
        elif nutrition in ['underweight', 'overweight']:
            score += 0.2
        
        siblings = row['NumberOfSiblings']
        if pd.notna(siblings):
            if siblings <= 3:
                score += 0.4
            elif siblings <= 6:
                score += 0.2
        
        age = row['Age']
        if pd.notna(age) and 10 <= age <= 13:
            score += 0.2
        
        sex = str(row['Sex']).lower()
        if sex in ['female', 'f']:
            score += 0.05
        
        health_demo_scores.append(min(1.0, score))
    
    df['health_demo_score'] = health_demo_scores
    df['completion_score'] += weights['health_demo'] * df['health_demo_score']
    
    return df['completion_score']


def calculate_completion_score(row):
    """
    Calculate a weighted completion score for a single student record
    Synchronized with RandomForestAlgorithm_new.py
    """
    weights = {
        'academic': 0.25,
        'habits': 0.20,
        'family': 0.20,
        'tech_extracurricular': 0.15,
        'proximity': 0.10,
        'health_demo': 0.10
    }
    
    total = 0.0
    
    # Academic (25%)
    grades = [row.get(f'Grade{i}_Average', np.nan) for i in range(1, 6)]
    grades = [g for g in grades if pd.notna(g)]
    if grades:
        avg_grade = np.mean(grades)
        curr = row.get('Grade6_CurrentGrade', row.get('CurrentGrade', np.nan))
        dynamic_threshold = np.mean(grades)
        acad_score = 0.0
        if avg_grade > dynamic_threshold:
            acad_score += 0.5
        if pd.notna(curr) and curr > dynamic_threshold:
            acad_score += 0.5
        total += weights['academic'] * acad_score
    
    # Study & Sleeping Habits (20%)
    study = row.get('StudyHabits', 0)
    sleep = row.get('SleepingHabits', 0)
    habit_score = ((study >= 3) + (sleep >= 3)) / 2
    total += weights['habits'] * habit_score
    
    # Family Background (20%)
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
    
    # Technology & Extracurricular (15%)
    tech_str = str(row.get('AccessToTechnology', '')).lower()
    tech_items = ['laptop', 'cellphone', 'computer', 'tablet']
    tech_count = sum(1 for item in tech_items if item in tech_str)
    tech_score = tech_count / len(tech_items)
    extra_flag = str(row.get('Extracurricular', '')).lower() == 'yes'
    combined_score = (tech_score + float(extra_flag)) / 2
    total += weights['tech_extracurricular'] * combined_score
    
    # Proximity (10%)
    proximity = str(row.get('Proximity', '')).lower()
    if proximity == 'near':
        proximity_score = 1.0
    elif proximity == 'far':
        proximity_score = 0.0
    else:
        proximity_score = 0.5
    total += weights['proximity'] * proximity_score
    
    # Health & Demographics (10%)
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
        'label_threshold': label_threshold,
        'metrics': model_metrics,
        'version': '2.0.0'
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
        logger.info(f"Loaded CSV with {len(df_original)} rows and {len(df_original.columns)} columns")
        
        # Store original identifiers BEFORE preprocessing
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
        
        # Remove 'CompletionRate' if present (we're predicting, not training)
        if 'CompletionRate' in df_processed.columns:
            df_processed = df_processed.drop(columns=['CompletionRate'])
            logger.info("Removed 'CompletionRate' column before prediction")
        
        # Align columns with trained model
        X = df_processed.reindex(columns=feature_columns, fill_value=0)
        logger.info(f"Aligned features shape: {X.shape}")
        
        # Get threshold
        threshold = data.get('threshold', THRESHOLD)
        
        # Make predictions
        y_pred_proba = model.predict_proba(X)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        logger.info(f"Predictions complete: {len(y_pred)} students")
        
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
                
                # Verify column count
                if shap_matrix.shape[1] != X.shape[1]:
                    logger.warning(f"⚠️ SHAP feature mismatch: shap={shap_matrix.shape[1]} vs X={X.shape[1]}")
                    shap_matrix = None
                    
            except Exception as e:
                logger.warning(f"⚠️ SHAP computation failed: {e}")
                shap_matrix = None
        
        # Calculate completion scores for each student
        completion_scores = []
        for _, row in df_original.iterrows():
            score = calculate_completion_score(row)
            completion_scores.append(score)
        
        # Format results
        results = []
        for i in range(len(y_pred)):
            # Format LRN (handle scientific notation from Excel)
            raw_lrn = original_lrns[i] if i < len(original_lrns) else f"STU{i+1:03d}"
            
            if isinstance(raw_lrn, (float, np.floating)):
                lrn = f"{raw_lrn:.0f}"
            elif isinstance(raw_lrn, str):
                try:
                    lrn = f"{float(raw_lrn):.0f}"
                except ValueError:
                    lrn = raw_lrn
            else:
                lrn = str(raw_lrn)
            
            age = original_ages[i] if i < len(original_ages) else None
            
            # Compute grouped SHAP factors
            top_factors = []
            if shap_matrix is not None and X is not None:
                try:
                    shap_values_row = shap_matrix[i]
                    shap_df = pd.DataFrame({
                        "feature": X.columns,
                        "impact": shap_values_row
                    })
                    
                    # Group one-hot encoded features back to their base name
                    shap_df["base_feature"] = shap_df["feature"].apply(lambda f: f.split("_")[0])
                    
                    grouped = (
                        shap_df.groupby("base_feature")["impact"]
                        .agg(lambda x: np.mean(x))
                        .reset_index()
                    )
                    
                    # Filter based on prediction direction
                    if int(y_pred[i]) == 0:
                        # At Risk - show top negative (risk) factors only
                        grouped = grouped[grouped["impact"] < 0].sort_values("impact", ascending=True)
                        top_factors = grouped.head(5).to_dict(orient="records")
                    else:
                        # Likely to Complete - show top positive (success) factors only
                        grouped = grouped[grouped["impact"] > 0].sort_values("impact", ascending=False)
                        top_factors = grouped.head(5).to_dict(orient="records")
                        
                except Exception as e:
                    logger.warning(f"Error processing SHAP for student {i}: {e}")
                    top_factors = [{"feature": "Error grouping SHAP", "impact": str(e)}]
            
            results.append({
                "lrn": str(lrn),
                "age": int(age) if age and pd.notna(age) else None,
                "result": int(y_pred[i]),
                "prediction": CLASS_LABELS[int(y_pred[i])],
                "probability": float(y_pred_proba[i]),
                "score": float(completion_scores[i]),
                "factors": top_factors
            })
        
        # Summary statistics
        at_risk_count = int(sum(1 for pred in y_pred if pred == 0))
        likely_to_complete_count = int(sum(1 for pred in y_pred if pred == 1))
        
        logger.info(f"Summary - At Risk: {at_risk_count}, Likely to Complete: {likely_to_complete_count}")
        
        return jsonify({
            "success": True,
            "predictions": results,
            "summary": {
                "total_students": len(results),
                "at_risk_count": at_risk_count,
                "likely_to_complete_count": likely_to_complete_count,
                "at_risk_percentage": float(at_risk_count / len(results) * 100) if len(results) > 0 else 0.0,
                "likely_to_complete_percentage": float(likely_to_complete_count / len(results) * 100) if len(results) > 0 else 0.0,
                "avg_probability": float(sum(y_pred_proba) / len(y_pred_proba)) if len(y_pred_proba) > 0 else 0.0
            }
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
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