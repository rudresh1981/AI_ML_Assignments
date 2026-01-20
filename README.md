# ML Assignment 2 - Breast Cancer Classification

A Streamlit web application for breast cancer classification using pre-trained machine learning models.

## Features

- **Offline Model Training**: Train models separately and save as artifacts
- **Multiple ML Models**: Logistic Regression, Decision Tree, k-NN, Naive Bayes, Random Forest, and XGBoost
- **Pre-trained Model Loading**: Load saved models for fast predictions
- **UCI Breast Cancer Dataset**: 569 samples with 30 features
- **Z-score Normalization**: StandardScaler preprocessing
- **Test Data Prediction**: Upload CSV files for batch predictions
- **Performance Metrics**: Accuracy, AUC, Precision, Recall, F1-Score, and MCC

## Architecture

The application follows an **offline training + online prediction** architecture:

1. **Offline Training** (`model_training.py` - run as script):
   - Trains all 6 models on Breast Cancer Wisconsin dataset
   - Applies 90/10 train/test split with z-score normalization
   - Saves model artifacts (`.pkl` files), scaler, and metadata

2. **Online Prediction** (`app.py` - Streamlit web app):
   - Loads pre-trained models from `trained_models/` directory
   - Provides web interface for uploading test data
   - Generates predictions using selected pre-trained model

## Workflow

### Step 1: Train Models Offline

First, ensure you have the training data:
- Place `BreastCancer_Wisconsin.csv` in the project directory, OR
- The script will automatically use sklearn's built-in dataset as fallback

```bash
# Activate virtual environment
.\venv\Scripts\Activate.ps1  # Windows PowerShell
# or
source venv/bin/activate  # Linux/Mac

# Train all models and save artifacts
python model_training.py
```

This creates a `trained_models/` directory containing:
- `logistic_regression.pkl`, `decision_tree.pkl`, `k_nn.pkl`, etc. (6 model files)
- `scaler.pkl` (StandardScaler for feature normalization)
- `metadata.json` (feature names, training info, dataset details)
- `training_results.json` (model performance metrics)

### Step 2: Run Streamlit App

```bash
# Run the prediction app
streamlit run app.py
```

The app will:
1. Load all pre-trained models from `trained_models/` directory
2. Display model performance metrics from training
3. Allow you to select a model and upload test data CSV
4. Generate predictions and provide downloadable results

## Requirements

See `requirements.txt` for all dependencies:
- streamlit 1.52.2
- pandas 2.3.3
- numpy 2.4.1
- scikit-learn 1.8.0
- xgboost 3.1.3

## Local Development

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1
# The activation script will automatically install requirements

# OR manually install dependencies
pip install -r requirements.txt

# Step 1: Train models offline
python model_training.py

# Step 2: Run the prediction app
streamlit run app.py
```

## Test Data Format

Your test CSV file should contain the same 30 features as the training data:
- mean radius, mean texture, mean perimeter, mean area, mean smoothness...
- (See app for complete list of feature names)
- No target column needed - predictions will be generated

## Deployment to Streamlit Community Cloud

### Prerequisites
1. Push code to GitHub repository
2. Include `trained_models/` directory in repository (or train models on cloud)

### Steps
1. Go to https://share.streamlit.io/
2. Sign in with GitHub
3. Click "New app"
4. Select repository: `rudresh1981/AI_ML_Assignments`
5. Branch: `main`
6. Main file path: `ML_Assignment_2/app.py`
7. Click "Deploy"

### Important Notes
- Ensure `requirements.txt` is in the same directory as `app.py`
- If models are too large for GitHub, you may need to:
  - Use Git LFS for large files
  - Train models directly on Streamlit Cloud (add startup script)
  - Store models in cloud storage (S3, GCS) and download on startup

## Project Structure

```
ML_Assignment_2/
├── model_training.py           # ML training module + offline training script
├── app.py                      # Streamlit web app (prediction only)
├── requirements.txt            # Python dependencies
├── trained_models/             # Pre-trained model artifacts
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── k_nn.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── scaler.pkl
│   ├── metadata.json
│   └── training_results.json
├── BreastCancer_Wisconsin.csv  # Training dataset (optional)
├── .gitignore
└── README.md
```

## MLOps Artifacts

The `trained_models/` directory contains all necessary artifacts for model deployment:
- **Model Files**: Serialized trained models (`.pkl`)
- **Scaler**: Fitted StandardScaler for feature preprocessing
- **Metadata**: Feature names, target classes, training configuration
- **Training Results**: Performance metrics for model selection

## Author

Rudresh Ramalingappa - ML Assignment 2 - Breast Cancer Classification
