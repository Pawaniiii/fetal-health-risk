# app.py
import streamlit as st
import pandas as pd
import joblib
import json
from pathlib import Path

st.set_page_config(page_title='Fetal Health Predictor', layout='centered')
st.title('Fetal / Pregnancy Risk — Demo')

MODEL_PATH = Path('model.joblib')
FEATURES_PATH = Path('features.json')

if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
    st.warning('Model or features.json not found.')

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    with open(FEATURES_PATH, 'r') as f:
        features = json.load(f)
    return model, features

if MODEL_PATH.exists() and FEATURES_PATH.exists():
    model, features = load_model()

    st.sidebar.header('Prediction mode')
    mode = st.sidebar.radio('Choose input mode', ['Manual single record', 'CSV batch file'])

    if mode == 'Manual single record':
        values = {}
        cols = st.columns(2)
        for i, feat in enumerate(features):
            with cols[i%2]:
                values[feat] = st.number_input(feat, value=0.0)
        if st.button('Predict'):
            X = pd.DataFrame([values])[features]
            pred = model.predict(X)
            st.write('Prediction:', pred[0])

    else:
        uploaded = st.file_uploader('CSV file', type=['csv'])
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            missing = [c for c in features if c not in df.columns]
            if missing:
                st.error(f'Missing columns: {missing}')
            else:
                preds = model.predict(df[features])
                out = df.copy()
                out['prediction'] = preds
                st.write(out)
                st.download_button('Download predictions CSV', out.to_csv(index=False), 'predictions.csv')
