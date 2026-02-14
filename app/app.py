import streamlit as st
import joblib
import pandas as pd

# --- CONFIGURAÇÃO ---
st.set_page_config(page_title="Previsão de Diamantes", page_icon="💎")

st.title("💎 Previsão de Preço de Diamantes")
st.markdown("**Modelo:** Decision Tree Regressor (Pipeline com Pré-processamento)")

# Carregar modelo (Pipeline completo)
try:
    pipeline = joblib.load("models/diamond_price_model.joblib")
except FileNotFoundError:
    st.error("Modelo não encontrado! Execute 'python train.py' primeiro.")
    st.stop()

# --- INPUTS DO USUÁRIO ---
with st.form("form_diamante"):
    col1, col2 = st.columns(2)
    
    with col1:
        carat = st.number_input("Carat (Quilates)", 0.2, 5.0, 0.7)
        depth = st.number_input("Depth (%)", 43.0, 79.0, 61.0)
        table = st.number_input("Table (%)", 43.0, 95.0, 57.0)
        cut = st.selectbox("Corte (Cut)", ['Ideal', 'Premium', 'Very Good', 'Good', 'Fair'])
        
    with col2:
        x = st.number_input("X (mm)", 0.0, 11.0, 5.0)
        y = st.number_input("Y (mm)", 0.0, 59.0, 5.0)
        z = st.number_input("Z (mm)", 0.0, 32.0, 3.0)
        color = st.selectbox("Cor", ['D', 'E', 'F', 'G', 'H', 'I', 'J'])
        clarity = st.selectbox("Claridade", ['IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'I1'])

    submit = st.form_submit_button("Calcular Preço 💰")

if submit:
    # Cria o DataFrame Exatamente como ele vem do Seaborn
    # O Pipeline vai tratar as categorias automaticamente
    input_data = pd.DataFrame({
        'carat': [carat],
        'cut': [cut],
        'color': [color],
        'clarity': [clarity],
        'depth': [depth],
        'table': [table],
        'x': [x],
        'y': [y],
        'z': [z]
    })
    
    prediction = pipeline.predict(input_data)[0]
    st.success(f"Preço Estimado: **US$ {prediction:,.2f}**")