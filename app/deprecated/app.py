"""Aplicativo Streamlit para previsão de estoque de materiais com otimização de hiperparâmetros."""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
root_path = os.path.dirname(__file__)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from materials_stock_libs.models.models import SalesModels
from materials_stock_libs.utils import generate_future_dates
from PIL import Image

import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Previsão de Estoque", layout="wide")
img = Image.open(os.path.join(root_path, "logo.png"))

# Sidebar
st.sidebar.image(img, caption="Empresa", use_column_width=True)
st.sidebar.header("Parâmetros")
uploaded_file = st.sidebar.file_uploader("Carregue um Excel com dados de estoque, ele deve estar no formato de coluna 'data', 'material' e 'quantidade'", type="xlsx")

if uploaded_file:
    # ==============================
    # Carregar dados
    # ==============================
    try:
        df = pd.read_excel(uploaded_file, engine='openpyxl')
        df['data'] = pd.to_datetime(df['data'])
        materiais = df['material'].unique()
    except Exception as e:
        st.error(f"Erro ao encontrar a coluna 'data': {e}. Certifique-se de que o Excel contém as colunas ['data', 'material', 'quantidade'].")
        st.stop()

    
    material = st.sidebar.selectbox("Escolha o tipo de material", materiais)

    modelo_tipo = st.sidebar.selectbox("Modelo de regressão", ["linear", "rf"])
    otimizar = st.sidebar.checkbox("Otimizar hiperparâmetros (Optuna)", value=False)

    dias_futuros = st.sidebar.slider("Dias para prever", 30, 365, 180)

    # ==============================
    # Criar modelo
    # ==============================
    modelo_estoque = SalesModels(df)

    # Otimizar hiperparâmetros se marcado
    rf_params, prophet_params = {}, {}
    if otimizar:
        st.sidebar.write("Otimizando modelos, aguarde...")
        if modelo_tipo == "rf":
            rf_params = modelo_estoque.optimize_regression(material, model_type="rf", n_trials=20)
        prophet_params = modelo_estoque.optimize_forecast(material, n_trials=20)
        st.sidebar.success("Otimização concluída!")

    # Treinar modelos com hiperparâmetros
    modelo_estoque.train_regression_model(material, model_type=modelo_tipo, **rf_params)
    modelo_estoque.train_forecast_model(material, **prophet_params)

    # ==============================
    # Fazer previsões
    # ==============================
    future_dates = generate_future_dates(df['data'].max(), days=dias_futuros)
    y_pred = modelo_estoque.predict_stock(material, future_dates)
    zero_date = modelo_estoque.predict_zero_stock_date(material, days_ahead=dias_futuros)

    # ==============================
    # Visualizações
    # ==============================
    st.subheader(f"Previsão de Estoque para {material}")

    fig = go.Figure()

    # Linha do histórico
    fig.add_trace(go.Scatter(
        x=df['data'],
        y=df['quantidade'],
        mode='lines+markers',
        name='Histórico'
    ))

    # Linha da previsão
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=y_pred,
        mode='lines',
        line=dict(dash='dash'),
        name='Previsão'
    ))

    # Linha do estoque zerado
    if zero_date:
        fig.add_trace(go.Scatter(
            x=[zero_date, zero_date],
            y=[0, max(df['quantidade'].max(), max(y_pred))],
            mode='lines',
            line=dict(color='red', dash='dot'),
            name=f"Estoque Zera: {zero_date.date()}"
        ))

    fig.update_layout(
        xaxis_title='Data',
        yaxis_title='Quantidade',
        legend_title='Legenda',
        width=900,
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    # --- Gráfico de Estoque Atual por Material ---
    st.subheader("Estoque Atual por Tipo de Material")

    estoque_atual = df.groupby('material')['quantidade'].last().reset_index()

    fig2 = px.bar(
        estoque_atual,
        x='material',
        y='quantidade',
        labels={'quantidade': 'Quantidade', 'material': 'Material'},
        text='quantidade'
    )
    fig2.update_traces(textposition='outside')
    fig2.update_layout(yaxis=dict(title='Quantidade'), xaxis=dict(title='Material'))

    st.plotly_chart(fig2, use_container_width=True)
