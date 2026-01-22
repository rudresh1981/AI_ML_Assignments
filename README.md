1. 

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
