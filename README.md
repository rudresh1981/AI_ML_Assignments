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
| **k-NN**              | Achieved highest accuracy (98.25%) with perfect recall (1.0000), successfully identifying all 36 malignant cases without false negatives. AUC of 0.9927 demonstrates superior class separation. Single false positive (1/57) indicates excellent specificity. MCC of 0.9626 confirms strong balanced performance across both classes. |
| **Random Forest**     | Demonstrated robust performance with 96.49% accuracy and highest AUC (0.9947), indicating optimal probability calibration. Balanced precision and recall (0.9722) across both classes with identical performance metrics to Logistic Regression. Ensemble approach minimizes overfitting risk while maintaining interpretability through feature importance analysis. |
| **Logistic Regression** | Delivered 96.49% accuracy with well-calibrated probabilities (AUC: 0.9894). Balanced precision-recall (0.9722) indicates consistent performance without class bias. Linear decision boundary provides interpretable coefficients for clinical feature analysis. Computationally efficient with minimal hyperparameter tuning required. |
| **XGBoost**           | Achieved 94.74% accuracy with exceptional AUC (0.9921), second-highest among all models. Recall of 0.9444 indicates 2 false negatives from 36 malignant cases. Gradient boosting architecture provides non-linear decision boundaries. MCC of 0.8886 confirms reliable performance despite moderate computational complexity. |
| **Naive Bayes**       | Attained 94.74% accuracy with strong AUC (0.9907) despite independence assumption. Recall of 0.9444 with 2 false negatives matches XGBoost performance. Probabilistic framework enables fast training and prediction. Gaussian distribution assumption appears appropriate for normalized continuous features. |
| **Decision Tree**     | Recorded lowest accuracy (92.98%) with 3 false negatives (recall: 0.9167). AUC of 0.9345 indicates moderate discriminative ability. Single-tree architecture provides maximum interpretability with explicit decision rules. MCC of 0.8545 suggests adequate but suboptimal performance. Prone to overfitting without ensemble aggregation. |

### Key Insights

1. **Distance-based k-NN achieved optimal performance** (accuracy: 98.25%, recall: 1.0) on normalized features, indicating well-separated class distributions in 30-dimensional feature space.
2. **Ensemble methods outperformed single models** with Random Forest (AUC: 0.9947) and XGBoost (AUC: 0.9921) demonstrating superior probability calibration and generalization capability.
3. **All models exceeded 92% accuracy threshold**, confirming strong predictive signals in tumor characteristics with effective z-score normalization preprocessing.
4. **Recall prioritization is critical** for clinical deployment—k-NN's zero false negative rate minimizes risk of undetected malignant cases, aligning with medical diagnostic requirements.
5. **Consistent AUC performance above 0.93** across all models validates dataset quality and feature engineering effectiveness for binary classification task.

---
