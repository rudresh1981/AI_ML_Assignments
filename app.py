import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os

# Import only the prediction function from model_training module
from model_training import predict_new_data

st.set_page_config(page_title='ML Assignment 2 — Breast Cancer Classification', layout='wide')

st.title('ML Assignment 2 — Breast Cancer Classification App')
st.write('Pre-trained models on Breast Cancer Wisconsin Dataset. Upload test data for predictions.')

# Constants
MODELS_DIR = 'trained_models'

# Load metadata and models
@st.cache_resource
def load_model_artifacts():
    """Load all pre-trained models, scaler, and metadata"""
    
    if not os.path.exists(MODELS_DIR):
        return None, None, None, None, "Models directory not found. Please run model_training.py first."
    
    # Load metadata
    metadata_path = os.path.join(MODELS_DIR, 'metadata.json')
    if not os.path.exists(metadata_path):
        return None, None, None, None, "Metadata file not found. Please run model_training.py first."
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Load scaler
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
    if not os.path.exists(scaler_path):
        return None, None, None, None, "Scaler file not found. Please run model_training.py first."
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Load training results
    results_path = os.path.join(MODELS_DIR, 'training_results.json')
    training_results = None
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            training_results = json.load(f)
    
    # Load all models
    models = {}
    if training_results:
        for model_name, result in training_results.items():
            model_path = os.path.join(MODELS_DIR, result['filename'])
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    models[model_name] = pickle.load(f)
    
    if not models:
        return None, None, None, None, "No trained models found. Please run model_training.py first."
    
    return models, scaler, metadata, training_results, None

# Load artifacts
models, scaler, metadata, training_results, error_msg = load_model_artifacts()

if error_msg:
    st.error(f'{error_msg}')
    st.stop()

st.success('Pre-trained models loaded successfully!')

# Display model information
st.info(f'**Training Dataset:** {metadata["data_source"]} | '
        f'{metadata["n_samples_train"]} training samples | '
        f'{metadata["n_features"]} features | '
        f'Trained on: {metadata["training_date"]}')

feature_names = metadata['feature_names']
target_names = metadata['target_names']

st.write(f'**Target Classes:** {target_names[0]} (Malignant=0), {target_names[1]} (Benign=1)')

# Show available models and their performance
with st.expander("View Pre-Trained Models Performance"):
    if training_results:
        for model_name, result in training_results.items():
            st.write(f"**{model_name}**")
            metrics = result['metrics']
            cols = st.columns(6)
            cols[0].metric('Accuracy', f"{metrics['Accuracy']:.4f}")
            cols[1].metric('Precision', f"{metrics['Precision']:.4f}")
            cols[2].metric('Recall', f"{metrics['Recall']:.4f}")
            cols[3].metric('F1', f"{metrics['F1']:.4f}")
            if metrics['AUC']:
                cols[4].metric('AUC', f"{metrics['AUC']:.4f}")
            cols[5].metric('MCC', f"{metrics['MCC']:.4f}")
            st.divider()

# Sidebar for model selection
st.sidebar.header('Model Selection')
model_names_list = list(models.keys())
selected_model_name = st.sidebar.selectbox('Choose pre-trained model', options=model_names_list)
selected_model = models[selected_model_name]

# Display selected model performance
st.sidebar.subheader('Selected Model Performance')
if training_results and selected_model_name in training_results:
    metrics = training_results[selected_model_name]['metrics']
    st.sidebar.metric('Accuracy', f"{metrics['Accuracy']:.4f}")
    st.sidebar.metric('F1-Score', f"{metrics['F1']:.4f}")
    if metrics['AUC']:
        st.sidebar.metric('AUC', f"{metrics['AUC']:.4f}")

# Sidebar - Download test data template
st.sidebar.divider()
st.sidebar.subheader('📥 Test Data Template')
test_template_path = os.path.join(MODELS_DIR, 'test.csv')
if os.path.exists(test_template_path):
    with open(test_template_path, 'rb') as f:
        test_csv_data = f.read()
    
    st.sidebar.download_button(
        label="📄 Download test.csv",
        data=test_csv_data,
        file_name='test.csv',
        mime='text/csv',
        help='Download the actual test data used during model training'
    )
    st.sidebar.caption(f'57 samples with {len(feature_names)} features')

# Sidebar - Upload test data
st.sidebar.divider()
st.sidebar.subheader('📤 Upload Test Data')
uploaded_test_file = st.sidebar.file_uploader('Upload CSV file', type=['csv'], key='test_upload', help=f'CSV must have {len(feature_names)} features')

with st.sidebar.expander("View Required Features"):
    st.caption(", ".join(feature_names))

# Main content area - Prediction results
if uploaded_test_file is not None:
    st.header('Prediction Results')
    test_data = pd.read_csv(uploaded_test_file)
    st.write(f'**Test data shape:** {test_data.shape}')
    st.dataframe(test_data.head())
    
    # Use the prediction function from model_training module with pre-trained model
    result_df, predictions, probabilities, error_msg = predict_new_data(
        selected_model, scaler, test_data, feature_names, target_names
    )
    
    if error_msg:
        st.error(f'{error_msg}')
        st.write('Please ensure your CSV has the correct format and features.')
    else:
        st.success(f'Predictions completed for {len(test_data)} samples using {selected_model_name}!')
        st.subheader('Prediction Results')
        st.dataframe(result_df)
        
        # Download predictions
        csv = result_df.to_csv(index=False)
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name=f'predictions_{selected_model_name.replace(" ", "_")}.csv',
            mime='text/csv'
        )
        
        # Summary statistics
        st.subheader('Prediction Summary')
        col1, col2 = st.columns(2)
        with col1:
            st.metric('Total Predictions', len(predictions))
            st.metric(f'{target_names[0]} (Malignant)', int((predictions == 0).sum()))
        with col2:
            st.metric(f'{target_names[1]} (Benign)', int((predictions == 1).sum()))
            if probabilities is not None:
                st.metric('Avg Confidence', f"{probabilities.max(axis=1).mean():.4f}")
else:
    # Display instructions when no file is uploaded
    st.header('Get Started')
    st.info('👈 Use the sidebar to:\n\n1. **Download** the test data template (test.csv)\n2. **Upload** your test CSV file\n3. View predictions and results here')
    
    st.markdown('---')
    st.subheader('How to Use')
    st.markdown('''
    **Step 1:** Select a model from the sidebar  
    **Step 2:** Download the test.csv template (optional)  
    **Step 3:** Upload your CSV file with the required features  
    **Step 4:** View predictions and download results
    ''')
