## Problem Statement

This project focuses on evaluating multiple machine learning models for **binary classification of breast cancer tumors** as either **Malignant (cancerous)** or **Benign (non-cancerous)**. The goal is to compare the performance of six different machine learning algorithms and identify the most effective model for accurate diagnosis prediction based on tumor characteristics.

The evaluation includes comprehensive metrics such as Accuracy, Precision, Recall, F1-Score, AUC (Area Under the ROC Curve), and MCC (Matthews Correlation Coefficient) to provide a holistic view of each model's predictive capabilities.

---

## Dataset Description

### Source
The dataset is sourced from **Kaggle - Breast Cancer Wisconsin (Diagnostic) Dataset**, which contains features computed from digitized images of fine needle aspirate (FNA) of breast masses. https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

### Dataset Characteristics

- **Total Records**: 569 samples
- **Number of Features**: 30 numerical features
- **Target Variable**: Binary classification (Malignant = 0, Benign = 1)
- **Training Set**: 512 samples (90% of total data)
- **Test Set**: 57 samples (10% of total data)
- **Split Method**: Stratified train-test split with random_state=42

### Features Description

The 30 features are computed for each cell nucleus and consist of three sets of 10 measurements:
1. **Mean values** (radius1, texture1, perimeter1, area1, smoothness1, compactness1, concavity1, concave_points1, symmetry1, fractal_dimension1)
2. **Standard error** (radius2, texture2, ..., fractal_dimension2)
3. **Worst/largest values** (radius3, texture3, ..., fractal_dimension3)

All features are continuous numerical values representing geometric and texture properties of cell nuclei.

### Class Distribution

**Overall Dataset (569 samples):**
- Benign (1): ~63% of samples
- Malignant (0): ~37% of samples

**Test Set (57 samples):**
- Benign (1): 36 samples (63.2%)
- Malignant (0): 21 samples (36.8%)

**Balance Status**: The dataset shows a **moderate class imbalance** with approximately 1.7:1 ratio (Benign:Malignant). While not severely imbalanced, stratified sampling was used during train-test split to maintain class distribution consistency.

### Data Preprocessing Requirements

**Nature of Data**: 
- All features are continuous numerical measurements with varying scales
- Features have different units and ranges (e.g., area vs. smoothness)
- No categorical variables present

**Normalization/Standardization**: 
- **Z-score normalization (StandardScaler)** was applied to all features
- **Formula**: z = (x - mean) / standard_deviation
- **Reason**: Different features have vastly different scales (e.g., area ranges in thousands while fractal dimension is less than 1)
- **Method**: Scaler fitted on training data only, then applied to test data to prevent data leakage
- **Importance**: Essential for distance-based algorithms (k-NN, Logistic Regression) and improves convergence for gradient-based methods

---

## Models Used

Six machine learning models were evaluated for breast cancer classification:

| ML Model Name          | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|------------------------|----------|--------|-----------|--------|--------|--------|
| **k-NN**              | 0.9825   | 0.9927 | 0.9730    | 1.0000 | 0.9863 | 0.9626 |
| **Random Forest**     | 0.9649   | 0.9947 | 0.9722    | 0.9722 | 0.9722 | 0.9246 |
| **Logistic Regression** | 0.9649   | 0.9894 | 0.9722    | 0.9722 | 0.9722 | 0.9246 |
| **XGBoost**           | 0.9474   | 0.9921 | 0.9714    | 0.9444 | 0.9577 | 0.8886 |
| **Naive Bayes**       | 0.9474   | 0.9907 | 0.9714    | 0.9444 | 0.9577 | 0.8886 |
| **Decision Tree**     | 0.9298   | 0.9345 | 0.9706    | 0.9167 | 0.9429 | 0.8545 |

**Note**: Models are ranked by Accuracy score.

---

## Observations

| ML Model Name          | Performance Observations |
|------------------------|--------------------------|
| **k-NN**              | **Best overall performer** with highest accuracy (98.25%) and perfect recall (100%), meaning it correctly identified all malignant cases. Excellent AUC (0.9927) indicates strong discriminative ability. Only 1 false positive occurred. Ideal for this critical healthcare application where missing malignant cases is costly. |
| **Random Forest**     | **Second-best performance** with very high accuracy (96.49%) and the highest AUC (0.9947). Balanced precision and recall (97.22% each) demonstrate consistent performance across both classes. Ensemble method provides robust predictions with minimal overfitting. Excellent choice for production deployment. |
| **Logistic Regression** | Strong performer with 96.49% accuracy and excellent interpretability. High AUC (0.9894) shows good probability calibration. Balanced precision-recall trade-off makes it reliable for clinical decision support. Computationally efficient and provides probability estimates for risk assessment. |
| **XGBoost**           | Good performance (94.74% accuracy) with exceptional AUC (0.9921), second only to Random Forest. Slightly lower recall (94.44%) means it missed 2 malignant cases. Advanced gradient boosting provides competitive results but may require more computational resources. |
| **Naive Bayes**       | Achieved 94.74% accuracy with excellent AUC (0.9907). Despite strong probabilistic assumptions, performs well on this dataset. Fast training and prediction makes it suitable for real-time applications. Same recall (94.44%) as XGBoost with 2 missed malignant cases. |
| **Decision Tree**     | Lowest accuracy (92.98%) among evaluated models but still respectable. Lower recall (91.67%) means 3 malignant cases were misclassified. Highly interpretable with clear decision rules. More prone to overfitting compared to ensemble methods. Best used as a baseline or for rule extraction. |

### Key Insights

1. **Distance-based method (k-NN) excelled** due to well-separated clusters in the normalized feature space
2. **Ensemble methods (Random Forest, XGBoost)** demonstrated robust performance with high AUC scores
3. **All models achieved >92% accuracy**, indicating the dataset has strong predictive signals
4. **Recall is critical** in medical diagnosis - k-NN's perfect recall (no missed malignant cases) makes it particularly valuable
5. **High AUC scores (>0.93)** across all models show excellent probability calibration for risk stratification

---

## Project Structure

```
AI_ML_Assignments/
├── app.py                          # Streamlit web application
├── model_training.py               # Offline model training script
├── requirements.txt                # Python dependencies
├── BreastCancer_Wisconsin.csv      # Dataset
├── test.csv                        # Test data template (57 samples)
├── model/                          # Model implementations
│   ├── __init__.py
│   ├── logistic_regression.py
│   ├── decision_tree.py
│   ├── k_nn.py
│   ├── naive_bayes.py
│   ├── random_forest.py
│   └── xgboost_model.py
└── trained_models/                 # Pre-trained models & artifacts
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── k-nn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    ├── xgboost.pkl
    ├── scaler.pkl                  # StandardScaler for normalization
    ├── metadata.json               # Dataset and training metadata
    ├── training_results.json       # Model performance metrics
    ├── test.csv                    # Test data (no labels)
    └── test_with_labels.csv        # Test data (with labels)
```

---

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/rudresh1981/AI_ML_Assignments.git
cd AI_ML_Assignments
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1    # Windows PowerShell
# OR
source venv/bin/activate        # Linux/Mac
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Usage

### Option 1: Run Streamlit Web App (Recommended)

Launch the interactive web application:
```bash
streamlit run app.py
```

**Features:**
- Select from 6 pre-trained models via sidebar
- Download test.csv template (57 actual test samples)
- Upload CSV files for batch predictions
- View detailed prediction results and probabilities
- Download prediction results as CSV
- Interactive model performance comparison

### Option 2: Offline Model Training

Retrain all models from scratch:
```bash
python model_training.py
```

**This will:**
- Load Breast Cancer Wisconsin dataset
- Split data (90% train, 10% test)
- Apply z-score normalization
- Train all 6 models
- Evaluate and save models with metrics
- Generate test.csv template
- Save artifacts in `trained_models/` directory

---

## Testing Predictions

### Using Streamlit App:
1. Launch app: `streamlit run app.py`
2. Download `test.csv` from sidebar (57 test samples)
3. Upload the CSV file
4. View predictions and download results

### Test Data Format:
The CSV must contain 30 features in this exact order:
```
radius1, texture1, perimeter1, area1, smoothness1, compactness1, 
concavity1, concave_points1, symmetry1, fractal_dimension1,
radius2, texture2, perimeter2, area2, smoothness2, compactness2,
concavity2, concave_points2, symmetry2, fractal_dimension2,
radius3, texture3, perimeter3, area3, smoothness3, compactness3,
concavity3, concave_points3, symmetry3, fractal_dimension3
```

---

## Model Details

| Model | Algorithm Type | Key Hyperparameters |
|-------|---------------|---------------------|
| **Logistic Regression** | Linear classifier | max_iter=1000 |
| **Decision Tree** | Tree-based | criterion='gini', max_depth=None |
| **k-NN** | Instance-based | n_neighbors=5, metric='minkowski' |
| **Naive Bayes** | Probabilistic | GaussianNB (default) |
| **Random Forest** | Ensemble (Bagging) | n_estimators=100, max_depth=None |
| **XGBoost** | Ensemble (Boosting) | n_estimators=100, learning_rate=0.1 |

---

## Technologies Used

- **Python 3.12**
- **Streamlit** - Interactive web application
- **scikit-learn** - ML algorithms and preprocessing
- **XGBoost** - Gradient boosting framework
- **pandas** - Data manipulation
- **NumPy** - Numerical computing
- **pickle** - Model serialization

---

## Performance Metrics Explained

- **Accuracy**: Overall correct predictions / Total predictions
- **Precision**: True Positives / (True Positives + False Positives) - How many predicted malignant cases are actually malignant
- **Recall (Sensitivity)**: True Positives / (True Positives + False Negatives) - How many actual malignant cases were correctly identified
- **F1-Score**: Harmonic mean of Precision and Recall
- **AUC**: Area Under ROC Curve - Model's ability to distinguish between classes
- **MCC**: Matthews Correlation Coefficient - Balanced measure considering all confusion matrix elements

---

## Conclusion

The evaluation demonstrates that **k-NN achieves the best performance** for breast cancer classification with 98.25% accuracy and perfect recall, making it the recommended model for clinical deployment where missing malignant cases is unacceptable. **Random Forest** and **Logistic Regression** serve as excellent alternatives, offering balanced performance with strong interpretability.

All models achieve >92% accuracy, confirming the dataset's strong predictive signals and the effectiveness of z-score normalization for this medical classification task.

---

## Author

**Rudresh Ramalingappa**  
GitHub: [@rudresh1981](https://github.com/rudresh1981)

---

## License

This project is for educational purposes as part of ML/AI coursework.

---

## Acknowledgments

- Dataset: Breast Cancer Wisconsin (Diagnostic) from Kaggle
- UCI Machine Learning Repository
- scikit-learn documentation and community`
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
