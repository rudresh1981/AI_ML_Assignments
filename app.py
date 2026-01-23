import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
import time

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

# Sidebar - Instructions at the top with download option
st.sidebar.header('📋 How to Use')
st.sidebar.markdown('''
1. **Select** a model below
2. **Download** test template
3. **Upload** your CSV file
4. **View** predictions in main area
''')

# Download test data template
test_template_path = os.path.join(MODELS_DIR, 'test.csv')
if os.path.exists(test_template_path):
    with open(test_template_path, 'rb') as f:
        test_csv_data = f.read()
    
    st.sidebar.download_button(
        label="📄 Download BreastCancer_test.csv",
        data=test_csv_data,
        file_name='BreastCancer_test.csv',
        mime='text/csv',
        help='Test data (57 samples, 30 features)',
        use_container_width=True
    )

st.sidebar.divider()

# Sidebar for model selection
st.sidebar.header('Model Selection')
model_names_list = list(models.keys())
selected_model_name = st.sidebar.selectbox('Select model', options=model_names_list)
selected_model = models[selected_model_name]

# Show available models and their performance (always visible - compact view)
st.subheader('Training Performance comparison')
if training_results:
    # Create a dataframe for compact table view
    performance_data = []
    for model_name, result in training_results.items():
        metrics = result['metrics']
        # Add visual indicator for selected model
        display_name = f"➤ {model_name}" if model_name == selected_model_name else model_name
        performance_data.append({
            'Model': display_name,
            'Accuracy': f"{metrics['Accuracy']:.4f}",
            'Precision': f"{metrics['Precision']:.4f}",
            'Recall': f"{metrics['Recall']:.4f}",
            'F1': f"{metrics['F1']:.4f}",
            'AUC': f"{metrics['AUC']:.4f}" if metrics['AUC'] else 'N/A',
            'MCC': f"{metrics['MCC']:.4f}"
        })
    
    performance_df = pd.DataFrame(performance_data)
    st.dataframe(performance_df, use_container_width=True, hide_index=True)
    

st.divider()

# Sidebar - Upload test data
st.sidebar.divider()
st.sidebar.subheader('Upload Data for Prediction')
uploaded_test_file = st.sidebar.file_uploader('Upload CSV file', type=['csv'], key='test_upload', help=f'CSV must have {len(feature_names)} features')

with st.sidebar.expander("Required Features"):
    st.caption(", ".join(feature_names))

# Main content area - Prediction results
if uploaded_test_file is not None:
    st.header('Prediction Results')
    test_data = pd.read_csv(uploaded_test_file)
    st.write(f'**Test data shape:** {test_data.shape}')
    st.dataframe(test_data.head())
    
    # Measure prediction time
    start_time = time.time()
    
    # Use the prediction function from model_training module with pre-trained model
    result_df, predictions, probabilities, error_msg = predict_new_data(
        selected_model, scaler, test_data, feature_names, target_names
    )
    
    # Calculate prediction time
    end_time = time.time()
    prediction_time = end_time - start_time
    avg_prediction_time = prediction_time / len(test_data) if len(test_data) > 0 else 0
    
    if error_msg:
        st.error(f'{error_msg}')
        st.write('Please ensure the uploaded file has features. See feature list in sidebar or the test data template.')
    else:
        st.success(f'Predictions completed for {len(test_data)} input file using {selected_model_name}!')
        
        # Container for predictions (this will be targeted for scrolling)
        with st.container():
            st.subheader(f'Prediction Results - {selected_model_name}')
            st.dataframe(result_df)
            
            # Scroll to this section after render
            st.components.v1.html(
                """
                <script>
                    window.parent.document.querySelector('[data-testid="stVerticalBlock"]').scrollIntoView({behavior: 'smooth'});
                </script>
                """,
                height=0,
            )
        
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
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric('Total Predictions', len(predictions))
            st.metric(f'{target_names[0]} (Malignant)', int((predictions == 0).sum()))
        with col2:
            st.metric(f'{target_names[1]} (Benign)', int((predictions == 1).sum()))
            if probabilities is not None:
                st.metric('Avg Confidence', f"{probabilities.max(axis=1).mean():.4f}")
        with col3:
            st.metric('Total Time', f"{prediction_time:.4f} sec")
            st.metric('Avg Time/Sample', f"{avg_prediction_time*1000:.2f} ms")
else:
    # Display message when no file is uploaded
    st.info('Upload test data ,CSV file, in sidebar to get started with predictions.')

