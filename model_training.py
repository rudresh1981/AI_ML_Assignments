"""
Model Training Module
Contains all machine learning model training, evaluation, and offline training functions
Run this file directly to train all models offline: python model_training.py
"""

import pandas as pd
import numpy as np
import os
import pickle
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, confusion_matrix, 
    matthews_corrcoef, classification_report
)
from sklearn.datasets import load_breast_cancer

# Import model modules from model package
from model import MODEL_MODULES

# Model mapping dictionary
MODEL_MAP = MODEL_MODULES


def load_training_data():
    """
    Load the Breast Cancer Wisconsin dataset from CSV file
    Falls back to sklearn dataset if CSV not found
    
    Returns:
    --------
    tuple : (df, feature_names, target_names)
        DataFrame with data, list of feature names, array of target class names
    """
    try:
        # Try to load from CSV file
        df = pd.read_csv('BreastCancer_Wisconsin.csv')
        
        # Identify target column (case-insensitive)
        # Handle both 'Diagnosis' and 'diagnosis'
        df.columns = df.columns.str.strip()  # Remove any whitespace
        
        if 'Diagnosis' in df.columns:
            target_col = 'Diagnosis'
            # Map M (Malignant) to 0 and B (Benign) to 1
            df['target'] = df[target_col].map({'M': 0, 'B': 1})
            df = df.drop(columns=[target_col])
            target_names = np.array(['malignant', 'benign'])
        elif 'diagnosis' in df.columns:
            target_col = 'diagnosis'
            # Map M (Malignant) to 0 and B (Benign) to 1
            df['target'] = df[target_col].map({'M': 0, 'B': 1})
            df = df.drop(columns=[target_col])
            target_names = np.array(['malignant', 'benign'])
        elif 'target' in df.columns:
            target_col = 'target'
            target_names = np.array(['malignant', 'benign'])
        else:
            raise ValueError('CSV must contain "diagnosis" or "Diagnosis" or "target" column')
        
        # Remove ID column if present (case-insensitive)
        id_cols = [col for col in df.columns if 'id' in col.lower() or 'unnamed' in col.lower()]
        if id_cols:
            df = df.drop(columns=id_cols)
        
        # Get feature names (all columns except target)
        feature_names = [col for col in df.columns if col != 'target']
        
        return df, feature_names, target_names, 'CSV'
        
    except FileNotFoundError:
        # Fallback to sklearn dataset
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df, data.feature_names, data.target_names, 'sklearn'
    
    except Exception as e:
        # On any other error, fall back to sklearn dataset
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df, data.feature_names, data.target_names, 'sklearn'


def prepare_data(df, target_col='target', test_size=0.1, random_state=42):
    """
    Prepare data for training by splitting into train/test sets
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with features and target
    target_col : str
        Name of the target column
    test_size : float
        Fraction of data to use for testing (default 0.1 = 10%)
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    tuple : (X_train, X_test, y_train, y_test)
    """
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Train/test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=int(random_state), 
        stratify=y
    )
    
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """
    Apply z-score normalization to features
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    X_test : array-like
        Test features
        
    Returns:
    --------
    tuple : (X_train_scaled, X_test_scaled, scaler)
        Scaled features and the fitted scaler object
    """
    # Z-score normalization (StandardScaler)
    # Formula: z = (x - mean) / std_dev
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Fit on training data
    X_test_scaled = scaler.transform(X_test)        # Transform test data using training statistics
    
    
    return X_train_scaled, X_test_scaled, scaler


def train_model(model_name, X_train, y_train):
    """
    Train a machine learning model
    
    Parameters:
    -----------
    model_name : str
        Name of the model from MODEL_MAP
    X_train : array-like
        Training features (should be scaled)
    y_train : array-like
        Training labels
        
    Returns:
    --------
    tuple : (model, success, error_message)
        Trained model object, success flag, and error message if any
    """
    try:
        # Instantiate model using the module's get_model() function
        model_module = MODEL_MAP[model_name]
        model = model_module.get_model()
        
        # Train model
        model.fit(X_train, y_train)
        
        return model, True, None
        
    except Exception as e:
        return None, False, str(e)


def evaluate_model(model, X_test, y_test, target_names):
    """
    Evaluate a trained model and calculate metrics
    
    Parameters:
    -----------
    model : trained model object
        The trained machine learning model
    X_test : array-like
        Test features (should be scaled)
    y_test : array-like
        True test labels
    target_names : array-like
        Names of target classes
        
    Returns:
    --------
    dict : Dictionary containing all evaluation metrics and predictions
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Get probability scores if available
    y_proba = None
    try:
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, 'decision_function'):
            y_proba = model.decision_function(X_test)
    except Exception:
        y_proba = None
    
    # Calculate metrics
    metrics = {}
    metrics['Accuracy'] = accuracy_score(y_test, y_pred)
    
    # AUC (requires probability scores)
    if y_proba is not None:
        try:
            metrics['AUC'] = roc_auc_score(y_test, y_proba)
        except Exception:
            metrics['AUC'] = None
    else:
        metrics['AUC'] = None
    
    metrics['Precision'] = precision_score(y_test, y_pred, zero_division=0)
    metrics['Recall'] = recall_score(y_test, y_pred, zero_division=0)
    metrics['F1'] = f1_score(y_test, y_pred, zero_division=0)
    metrics['MCC'] = matthews_corrcoef(y_test, y_pred)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Classification report
    try:
        report = classification_report(y_test, y_pred, target_names=target_names, zero_division=0)
    except Exception:
        report = None
    
    return {
        'metrics': metrics,
        'confusion_matrix': cm,
        'classification_report': report,
        'predictions': y_pred,
        'probabilities': y_proba
    }


def predict_new_data(model, scaler, test_data, feature_names, target_names):
    """
    Make predictions on new test data
    
    Parameters:
    -----------
    model : trained model object
        The trained machine learning model
    scaler : StandardScaler object
        Fitted scaler from training
    test_data : pandas.DataFrame
        New test data with features
    feature_names : list
        List of feature names expected by the model
    target_names : array-like
        Names of target classes
        
    Returns:
    --------
    tuple : (result_df, predictions, probabilities, error_message)
        Results dataframe with predictions, predictions array, probabilities, and error if any
    """
    try:
        # Check for missing features
        missing_features = set(feature_names) - set(test_data.columns)
        if missing_features:
            return None, None, None, f"Missing required features: {missing_features}"
        
        # Select only required features in correct order
        X_test_user = test_data[feature_names]
        
        # Handle missing values
        X_test_user = X_test_user.fillna(X_test_user.median())
        
        # Scale the test data using training scaler
        X_test_user_scaled = scaler.transform(X_test_user)
        
        # Make predictions
        predictions = model.predict(X_test_user_scaled)
        
        # Get probabilities if available
        probabilities = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X_test_user_scaled)
            pred_df = pd.DataFrame({
                'Prediction': [target_names[int(p)] for p in predictions],
                'Prediction_Code': predictions,
                f'Probability_{target_names[0]}': probabilities[:, 0],
                f'Probability_{target_names[1]}': probabilities[:, 1]
            })
        else:
            pred_df = pd.DataFrame({
                'Prediction': [target_names[int(p)] for p in predictions],
                'Prediction_Code': predictions
            })
        
        # Combine with original test data
        result_df = pd.concat([test_data.reset_index(drop=True), pred_df], axis=1)
        
        return result_df, predictions, probabilities, None
        
    except Exception as e:
        return None, None, None, str(e)


def get_model_list():
    """
    Get list of available models
    
    Returns:
    --------
    list : List of model names
    """
    return list(MODEL_MAP.keys())


def train_and_save_all_models(output_dir='trained_models'):
    """
    Train all models and save artifacts (Offline Training)
    
    This function orchestrates the complete training workflow:
    1. Load training data
    2. Split into train/test sets (90/10)
    3. Apply z-score normalization
    4. Train all 6 models
    5. Evaluate each model
    6. Save models, scaler, and metadata
    
    Parameters:
    -----------
    output_dir : str
        Directory to save trained models and artifacts (default: 'trained_models')
        
    Returns:
    --------
    dict : Dictionary containing training results for all models
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("OFFLINE MODEL TRAINING")
    print("=" * 60)
    
    # Load training data
    print("\n1. Loading training data...")
    train_df, feature_names, target_names, data_source = load_training_data()
    print(f"   Data source: {data_source}")
    print(f"   Samples: {train_df.shape[0]}, Features: {train_df.shape[1]-1}")
    
    # Prepare data (90% train, 10% test)
    print("\n2. Preparing data (90% train, 10% test)...")
    X_train, X_test, y_train, y_test = prepare_data(
        train_df, target_col='target', test_size=0.1, random_state=42
    )
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    # Save original data sample (first 10 rows)
    original_sample = X_train.head(10).copy()
    original_sample['target'] = y_train.iloc[:10].values
    original_sample_path = os.path.join(output_dir, 'data_sample_original.csv')
    original_sample.to_csv(original_sample_path, index=False)
    print(f"   Original data sample saved: {original_sample_path}")
    
    # Scale features
    print("\n3. Applying z-score normalization...")
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Save scaled data sample (first 10 rows)
    scaled_sample = pd.DataFrame(
        X_train_scaled[:10], 
        columns=X_train.columns
    )
    scaled_sample['target'] = y_train.iloc[:10].values
    scaled_sample_path = os.path.join(output_dir, 'data_sample_scaled.csv')
    scaled_sample.to_csv(scaled_sample_path, index=False)
    print(f"   Scaled data sample saved: {scaled_sample_path}")
    
    
    # Save scaler
    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"   Scaler saved: {scaler_path}")
    
    # Save feature names and target names
    metadata = {
        'feature_names': list(feature_names),
        'target_names': list(target_names),
        'n_features': len(feature_names),
        'n_samples_train': len(X_train),
        'n_samples_test': len(X_test),
        'data_source': data_source,
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'test_size': 0.1,
        'random_state': 42
    }
    
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"   Metadata saved: {metadata_path}")
    
    # Train all models
    print("\n4. Training models...")
    print("-" * 60)
    
    model_results = {}
    model_list = get_model_list()
    
    for idx, model_name in enumerate(model_list, 1):
        print(f"\n   [{idx}/{len(model_list)}] Training {model_name}...")
        
        # Get model module
        model_module = MODEL_MAP[model_name]
        
        # Train and evaluate model using module's train_and_evaluate function
        model, results = model_module.train_and_evaluate(
            X_train_scaled, X_test_scaled, y_train, y_test
        )
        
        if not results['success']:
            print(f"   Training failed: {results.get('error', 'Unknown error')}")
            continue
        
        # Extract metrics
        metrics = results['metrics']
        
        # Save model
        model_filename = model_name.replace(' ', '_').lower() + '.pkl'
        model_path = os.path.join(output_dir, model_filename)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Store results
        model_results[model_name] = {
            'filename': model_filename,
            'metrics': {k: float(v) if v is not None else None for k, v in metrics.items()},
            'confusion_matrix': results['confusion_matrix']
        }
        
        print(f"   Model saved: {model_path}")
        print(f"      Accuracy: {metrics['Accuracy']:.4f}")
        print(f"      AUC: {metrics['AUC']:.4f}" if metrics['AUC'] else "      AUC: N/A")
        print(f"      F1-Score: {metrics['F1']:.4f}")
    
    # Save model results
    results_path = os.path.join(output_dir, 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump(model_results, f, indent=4)
    print(f"\n   Training results saved: {results_path}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nAll artifacts saved in: {output_dir}/")
    print(f"  - {len(model_list)} model files (.pkl)")
    print(f"  - 1 scaler file (scaler.pkl)")
    print(f"  - 1 metadata file (metadata.json)")
    print(f"  - 1 results file (training_results.json)")
    print(f"  - 2 data sample files (original & scaled CSV)")
    print("\nYou can now use these artifacts in the Streamlit app for predictions.")
    
    return model_results


# Main execution block for offline training
if __name__ == '__main__':
    print("\n Starting Offline Model Training...\n")
    
    # Run offline training
    results = train_and_save_all_models()
    
    # Print summary
    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE SUMMARY")
    print("=" * 60)
    for model_name, result in results.items():
        metrics = result['metrics']
        print(f"\n{model_name}:")
        print(f"  Accuracy:  {metrics['Accuracy']:.4f}")
        print(f"  Precision: {metrics['Precision']:.4f}")
        print(f"  Recall:    {metrics['Recall']:.4f}")
        print(f"  F1-Score:  {metrics['F1']:.4f}")
        if metrics['AUC']:
            print(f"  AUC:       {metrics['AUC']:.4f}")
    
    print("\nTraining complete! Run 'streamlit run app.py' to use the models.\n")
