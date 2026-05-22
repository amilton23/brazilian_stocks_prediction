"""Módulo para visualização de dados localmente ou via streamlit."""

############################################################################################################
### 1. IMPORTS
############################################################################################################

import pandas as pd
import plotly.express as px
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

############################################################################################################
### 2. VISUALIZAÇÃO DE DADOS
############################################################################################################

def plot_top_n_barplot(
    df,
    cols_group=['UF', 'EAN'],
    col_value='RS_PC',
    date_col=None,
    ano=None,
    top_n_dim=10,
    top_n_subdim=5,
    title=None,
    ylabel=None,
    xlabel=None
):
    """
    Gráfico interativo de barras com Plotly, mostrando os Top N produtos (EAN)
    por participação dentro das Top N categorias (UF, Cidade, etc.).
    
    Adaptação:
        - Permite filtrar o dataframe por ano (coluna de data + ano desejado).
        - Ideal para uso em Streamlit.

    Exemplo de uso:
        df.drop_duplicates(inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.dropna(inplace=True)
        fig = plot_top_n_barplot(
            df,
            cols_group=['UF', 'PRODUCT_DESC'],
            col_value='RS_PC',
            # date_col='DT_EMISSAO',
            # ano=2025,
            top_n_dim=5,
            top_n_subdim=10,
            title="Top 10 produtos por participação dentro das 5 maiores UFs",
            ylabel="Participação (%)",
            xlabel="Produto"
        )
        fig.show()
        st.plotly_chart(fig, use_container_width=True)
    """

    df = df.copy()

    # --- Se o usuário informar uma coluna de data e ano, aplica o filtro ---
    if date_col is not None and ano is not None:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df[df[date_col].dt.year == int(ano)]
        if df.empty:
            raise ValueError(f"Nenhum dado encontrado para o ano {ano} na coluna '{date_col}'.")

    # --- Agrupa e calcula soma ---
    df_grouped = df.groupby(cols_group, as_index=False)[col_value].sum()

    # --- Seleciona as top categorias principais (ex: UFs) ---
    top_dims = (
        df_grouped.groupby(cols_group[0])[col_value]
        .sum()
        .nlargest(top_n_dim)
        .index
    )
    df_grouped = df_grouped[df_grouped[cols_group[0]].isin(top_dims)]

    # --- Calcula participação (%) dentro de cada grupo principal ---
    df_grouped['share'] = (
        df_grouped.groupby(cols_group[0])[col_value]
        .transform(lambda x: x / x.sum())
        * 100
    )

    # --- Seleciona os top subgrupos (ex: EANs) dentro de cada grupo principal ---
    top_data = (
        df_grouped.sort_values([cols_group[0], 'share'], ascending=[True, False])
        .groupby(cols_group[0])
        .head(top_n_subdim)
    )

    # --- Criação do gráfico Plotly ---
    fig = px.bar(
        top_data,
        x=cols_group[1],
        y='share',
        color=cols_group[0],
        text='share',
        barmode='group',
        labels={
            'share': ylabel or 'Participação (%)',
            cols_group[1]: xlabel or cols_group[1],
            cols_group[0]: cols_group[0],
        },
        title=title or (
            f"Top {top_n_subdim} {cols_group[1]} "
            f"por participação nas {top_n_dim} maiores {cols_group[0]}"
            + (f" — Ano {ano}" if ano else "")
        ),
    )

    # --- Ajustes visuais ---
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig.update_layout(
        xaxis_tickangle=-45,
        plot_bgcolor='white',
        yaxis=dict(showgrid=True, gridcolor='lightgray'),
        legend_title_text=cols_group[0],
        title_x=0.5,
        height=600,
    )

    return fig

def plot_boxplot(df, columns=None, hue=None, figsize=(12, 8)):
    """
    Gera um boxplot (diagrama de caixa) para as colunas selecionadas de um DataFrame
    com a opção de incluir uma variável de hue para colorir os dados.
    Se a lista de 'columns' (colunas) não for fornecida, todas as colunas numéricas
    serão utilizadas. Os gráficos são dispostos em linhas de 3 colunas.

    Argumentos:
        df (DataFrame): O DataFrame contendo os dados.
        columns (list, optional): Lista de nomes de colunas a serem plotadas no boxplot. 
            Se não fornecida, todas as colunas numéricas serão utilizadas.
        hue (str, optional): Nome da coluna a ser usada como hue (categorias de cor). 
            Padrão é None.
        figsize (tuple, optional): Tamanho da figura (largura, altura). Padrão é (12, 8).

    Retorna:
        Exibe o gráfico boxplot.

    Exemplo de Uso:
        plot_boxplot(
            df=data, 
            columns=['Feature_A', 'Feature_B'], 
            hue='Category',
            figsize=(15, 6)
        )
    """
    
    if columns is None:
        columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    if not all(col in df.columns for col in columns):
        raise ValueError("Some columns are not present in the DataFrame.")
    
    n_columns = len(columns)
    n_rows = int(np.ceil(n_columns / 3))

    fig, axes = plt.subplots(n_rows, 3, figsize=(figsize[0], n_rows * figsize[1] / 3))
    
    axes = axes.flatten()
    
    for i, column in enumerate(columns):
        ax = axes[i]
        sns.boxplot(data = df, x=df[column], hue=hue, ax=ax)
        ax.set_title(f'Boxplot of {column}', fontsize=12)
        ax.set_xlabel('')
        ax.set_ylabel('')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off') 

    plt.tight_layout()
    plt.show()

def plot_treemap(df, group_cols=['UF', 'SEGMENTO_PROD'], col_value='RS_PC', date_col=None, ano=None):
    """Treemap para análise de participação de mercado (market share).
    
    Exemplo de uso:
        fig_treemap = plot_treemap(df, group_cols=['UF', 'SEGMENTO_PROD'], col_value='RS_PC', date_col='DT_EMISSAO', ano=2025)
        fig_treemap.show()
        st.plotly_chart(fig_treemap, use_container_width=True)
    
    
    """
    df = df.copy()
    if date_col and ano:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df[df[date_col].dt.year == int(ano)]

    df_treemap = df.groupby(group_cols, as_index=False)[col_value].sum()
    fig = px.treemap(
        df_treemap,
        path=group_cols,
        values=col_value,
        color=col_value,
        color_continuous_scale='Viridis',
        title=f"Market Share — {ano if ano else 'Geral'}"
    )
    return fig