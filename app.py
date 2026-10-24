"""Streamlit user interface. Launch with:  streamlit run app.py"""
from pathlib import Path
import pandas as pd
import streamlit as st
from predictor import predict, FEATURES

st.set_page_config(page_title='Transaction demand planner')
st.title('Banking transaction-demand planner')
st.write('Estimates whether a customer will make **Low (0-49)**, **Medium (50-199)** or **High (200+)** '
         'transactions in the next three months (Nov 2015 - Jan 2016).')
st.warning('Academic prototype trained on historical 2015 data. Probabilities are model estimates, not guarantees. '
           'Use only for aggregate capacity-planning research - never for credit, fraud or service-eligibility '
           'decisions about individuals.')

examples_file = Path(__file__).parent / 'outputs' / 'valid_examples.csv'
with st.expander('Input guidance'):
    st.write(f'Upload a CSV with these {len(FEATURES)} engineered feature columns (one row per customer). '
             'month_01 ... month_12 are transaction counts for Nov 2014 ... Oct 2015. Counts must be whole, '
             'non-negative numbers. Income and financial columns may be left blank. '
             'Create real feature rows with the feature-engineering section of the notebook.')
    st.code(', '.join(FEATURES))
    st.download_button('Download example CSV (synthetic customers)', examples_file.read_bytes(), 'valid_examples.csv')

source = st.radio('Choose input', ['Use the synthetic example customers', 'Upload my own CSV'])
if source == 'Upload my own CSV':
    uploaded = st.file_uploader('Customer feature CSV', type='csv')
    if uploaded is None:
        st.stop()
    try:
        data = pd.read_csv(uploaded)
    except Exception:
        st.error('Could not read that file. Please upload a comma-separated .csv file.')
        st.stop()
else:
    data = pd.read_csv(examples_file)

try:
    results = predict(data)
except ValueError as error:
    st.error(f'Input problem: {error}')
    st.stop()

st.subheader('Predictions')
st.dataframe(results.drop(columns='explanation'), hide_index=True)
for sentence in results['explanation']:
    st.write(sentence)
st.caption('Files are processed in memory and not stored. Run locally; do not upload real personal data to a public service.')
