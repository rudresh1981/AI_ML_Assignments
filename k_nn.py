"""
k-Nearest Neighbors Model Training
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, matthews_corrcoef


def get_model():
    """
    Get k-NN model instance
    
    Returns:
    --------
    KNeighborsClassifier : Model instance
    """
    return KNeighborsClassifier()


def get_model_name():
    """
    Get the display name of the model
    
    Returns:
    --------
    str : Model name
    """
    return 'k-NN'


def train_and_evaluate(X_train, X_test, y_train, y_test):
    """
    Train k-NN model and compute evaluation metrics
    
    Parameters:
    -----------
    X_train : array-like
        Training features (should be scaled)
    X_test : array-like
        Test features (should be scaled)
    y_train : array-like
        Training labels
    y_test : array-like
        Test labels
        
    Returns:
    --------
    tuple : (model, results_dict)
        Trained model object and dictionary containing evaluation metrics
    """
    try:
        # Instantiate and train model
        model = get_model()
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Get probability scores
        y_proba = None
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
        
        # AUC score (if probability available)
        auc = None
        if y_proba is not None:
            try:
                auc = roc_auc_score(y_test, y_proba)
            except:
                pass
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Matthews correlation coefficient
        mcc = matthews_corrcoef(y_test, y_pred)
        
        # Compile results
        results = {
            'model_name': get_model_name(),
            'success': True,
            'metrics': {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1': f1,
                'AUC': auc,
                'MCC': mcc
            },
            'confusion_matrix': cm.tolist(),
            'predictions': {
                'y_pred': y_pred.tolist(),
                'y_proba': y_proba.tolist() if y_proba is not None else None,
                'y_test': y_test.tolist()
            }
        }
        
        return model, results
        
    except Exception as e:
        results = {
            'model_name': get_model_name(),
            'success': False,
            'error': str(e),
            'metrics': None
        }
        return None, results
