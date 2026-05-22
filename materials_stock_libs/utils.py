"""Módulo utilitário para funções auxiliares."""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import hashlib

import re
from tqdm import tqdm
import gc
from typing import Union, List

from datetime import datetime, timedelta

############################################################################################################
### 1. FUNÇÕES UTILITÁRIAS GERAIS
############################################################################################################

# 1.1. FUNÇÕES DE ANÁLISE EXPLORATÓRIA
def contagem_frequencia_histograma(
    df: pd.DataFrame, 
    hist: bool = True, 
    hist_cols: list = None, 
    use_cols_filter: bool = False,
    OUTPUTS_PATH: str = os.getcwd()
):
    """
    Calcula a contagem de frequência para todas as colunas e, opcionalmente, 
    gera histogramas para colunas numéricas.
    """
    
    cols_para_frequencia = df.columns

    if use_cols_filter and hist_cols is not None:
        cols_para_frequencia = [col for col in hist_cols if col in df.columns]
    
    if hist:
        if hist_cols is None or hist_cols == []:
            hist_cols_final = df.select_dtypes(include=['number']).columns.tolist()
        else:
            hist_cols_final = hist_cols
    else:
        hist_cols_final = []

    for col in cols_para_frequencia:
        print("#" * 50)
        print(f"Contagem de frequência da coluna {col}:")
        print(df[col].value_counts(dropna=False).sort_index(ascending=True))

        if hist and col in hist_cols_final:
            plt.figure(figsize=(10, 6)) 
            sns.histplot(df[col].dropna(), kde=True)
            plt.title(f'Distribuição de {col}')
            plt.xticks(rotation=90)
            
            plt.show()
            
            if 'OUTPUTS_PATH' in globals():
                filepath = os.path.join(OUTPUTS_PATH, f'histograma_col_{col}.png')
                plt.savefig(filepath, bbox_inches='tight')
            else:
                print("Aviso: OUTPUTS_PATH não está definida, o gráfico não foi salvo.")

            plt.close()
    print("#" * 50)
    print("\n")

def unicos(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            print("#"*50)
            print(f"Coluna - {col}:")
            print(f"Contagem únicos: {len(df[col].unique())}")
            print(f"Valores únicos: {df[col].unique()}")
            print("#"*50)
            print(f"\n")

# 1.2. FUNÇÕES DE GERAÇÃO DE HASH
def gerar_sha256(df: pd.DataFrame, colunas: list, hash_col_name: str = "sha256") -> pd.Series:
    """
    Gera uma hash numérica de 256 dígitos a partir da concatenação das colunas informadas.

    Args:
        df (pd.DataFrame): DataFrame de entrada.
        colunas (list): Lista de colunas a serem concatenadas.
        hash_col_name (str): Nome da coluna de saída (caso queira adicionar ao df).

    Returns:
        pd.Series: Série contendo a hash numérica de 256 dígitos por linha.
    """
    if not all(col in df.columns for col in colunas):
        raise ValueError(f"Algumas colunas não foram encontradas no DataFrame: {set(colunas) - set(df.columns)}")

    concat_series = df[colunas].astype(str).agg("".join, axis=1)

    # Função auxiliar para gerar hash numérica de 80 dígitos
    def hash_numeric_256(x: str) -> str:
        # Gera SHA-256 hexadecimal (64 caracteres hexadecimais)
        h = hashlib.sha256(x.encode("utf-8")).hexdigest()
        # Converte para inteiro e mantém apenas os 64 primeiros dígitos decimais
        numeric_hash = str(int(h, 16))
        return numeric_hash.zfill(80)  # garante comprimento fixo

    hash_series = concat_series.map(hash_numeric_256)
    df[hash_col_name] = hash_series

    return hash_series

def filtrar_colunas(df: pd.DataFrame, filtro: Union[str, List[str]]) -> pd.DataFrame:
    """
    Filtra as colunas de um DataFrame com base em um ou mais critérios.

    Parâmetros:
        df : pd.DataFrame
            O DataFrame a ser filtrado.
        filtro : str | list[str]
            Um critério de filtro (ex: 'RS') ou uma lista de critérios (ex: ['RS', 'EAN']).

    Retorna:
        pd.DataFrame
            Um novo DataFrame contendo apenas as colunas que atendem ao(s) critério(s).
    """
    # Garante que 'filtro' seja sempre uma lista
    if isinstance(filtro, str):
        filtros = [filtro]
    else:
        filtros = filtro

    # Seleciona colunas que contêm qualquer um dos filtros
    colunas_filtradas = [
        col for col in df.columns
        if any(f in col for f in filtros)
    ]

    # Retorna apenas as colunas filtradas
    return df[colunas_filtradas]

# 1.2. Funções de teste de modelo
# Criar dados sintéticos
def make_synthetic_data(start_date:str='2022-01-01', n_months=36, seed=42):
    np.random.seed(seed)
    dates = pd.date_range(start=start_date, periods=n_months, freq='MS')
    trend = np.linspace(50, 150, n_months)                      # tendência linear
    seasonality = 15 * np.sin(2 * np.pi * np.arange(n_months) / 12)  # sazonalidade anual
    noise = np.random.normal(0, 5, n_months)                    # ruído
    y = trend + seasonality + noise

    df = pd.DataFrame({
        'DT_EMISSAO': dates,
        'UNIDADES': y,
        'UF': ['CE'] * n_months,
        'canal_venda': np.random.choice(['Online', 'Loja', 'Distribuidor'], size=n_months)
    })
    return df

############################################################################################################
### 2. FUNÇÕES DE TRANSFORMAÇÃO DE DADOS
############################################################################################################

# 2.1. FUNÇÕES PARA LIMPEZA DE DADOS
def normalizar_codigos(df, coluna, como='int', output_col:str=''):
    """
    Versão otimizada para grandes volumes.
    Converte strings como '7,89295E+12' ou '7.896714E+12' para inteiros ou strings normalizadas.
    Funciona de forma vetorizada (sem apply).

    Args:
        df (pd.DataFrame): dataframe de entrada
        coluna (str): nome da coluna a normalizar
        como (str): 'int' ou 'str'
    """
    # Cria uma cópia local para segurança
    serie = df[coluna].astype(str).str.strip()

    # Substitui vírgula por ponto (compatível com float)
    serie = serie.str.replace(',', '.', regex=False)

    # Converte valores vazios, 'nan', etc. para NaN reais
    serie = serie.replace(['', 'nan', 'None', 'NaT'], np.nan)

    # Identifica quais entradas são numéricas (float ou científica)
    mask_num = serie.str.match(r'^\d+(\.\d+)?(e[\+\-]?\d+)?$', case=False, na=False)

    # Inicializa coluna de saída
    serie_out = pd.Series(np.nan, index=serie.index)

    # Converte apenas os valores numéricos
    if mask_num.any():
        serie_num = serie[mask_num].astype(float).round().astype('Int64')
        serie_out.loc[mask_num] = serie_num

    # Preenche o restante (que já são códigos puros) convertendo para inteiro se possível
    mask_rest = ~mask_num & serie.notna()
    if mask_rest.any():
        try:
            serie_rest = serie[mask_rest].astype('Int64')
        except Exception:
            serie_rest = pd.to_numeric(serie[mask_rest], errors='coerce').astype('Int64')
        serie_out.loc[mask_rest] = serie_rest

    # Retorna no formato desejado
    if como == 'str':
        serie_out = serie_out.astype('Int64').astype(str)
    else:
        serie_out = serie_out.astype('Int64')

    # Atribui ao dataframe
    if output_col:
        df[output_col] = serie_out
    else:
        df[coluna + '_norm'] = serie_out
    return df

def merge_grande(df1, df2, chave='EAN_norm', how='inner', chunksize=100_000):
    """
    Realiza merge em blocos de forma otimizada e sem gerar colunas com sufixos (_x/_y).
    - Remove duplicatas na chave.
    - Elimina colunas duplicadas (mantém as do primeiro DF).
    - Mostra progresso com tqdm.
    """

    if chave not in df1.columns or chave not in df2.columns:
        raise KeyError(f"A coluna '{chave}' deve existir em ambos os DataFrames.")

    df2_unique_cols = [col for col in df2.columns if col not in df1.columns or col == chave]
    df2 = df2[df2_unique_cols]

    total_rows = len(df1)
    n_chunks = (total_rows // chunksize) + int(total_rows % chunksize != 0)
    merged_chunks = []

    print(f"\n🔄 Iniciando merge em blocos sem sufixos...")
    print(f"➡️ Base A: {len(df1):,} linhas | Base B: {len(df2):,} linhas")
    print(f"➡️ Modo: {how} | Chunk size: {chunksize:,} | Total blocos: {n_chunks}\n")

    for start in tqdm(range(0, total_rows, chunksize), total=n_chunks, desc="Processando blocos"):
        end = min(start + chunksize, total_rows)
        chunk = df1.iloc[start:end]

        merged_chunk = pd.merge(chunk, df2, on=chave, how=how, suffixes=("", ""))  # Evita sufixos
        merged_chunks.append(merged_chunk)

        del chunk, merged_chunk
        gc.collect()

    df_merged = pd.concat(merged_chunks, ignore_index=True)
    print(f"\n✅ Merge finalizado com {len(df_merged):,} linhas totais.")
    return df_merged

def preparar_chave(df, col='EAN_norm'):
    df[col] = (
        df[col]
        .astype(str)
        .str.strip()
        .str.replace(',', '.', regex=False)
        .str.replace('\.0$', '', regex=True)  # remove ".0" em floats convertidos
    )
    return df

def validate_str(value):
    if not isinstance(value, str):
        raise ValueError("Parâmetro esperado como string.")
    return value.strip()

def filter_segment(df, filter_col: str = None, segment_value: str = None) -> pd.DataFrame:
    """
    Filtra o DataFrame para um segmento específico (ex: 'UF' == 'CE' ou 'material' == 'Aço').
    """
    if not filter_col or not segment_value:
        raise ValueError("Parâmetros 'filter_col' e 'segment_value' devem ser informados.")

    filter_col = validate_str(filter_col)
    segment_value = validate_str(segment_value)

    df_filtered = df[df[filter_col] == segment_value].copy()
    if df_filtered.empty:
        raise ValueError(f"Nenhum dado encontrado para {filter_col} = '{segment_value}'.")
    return df_filtered

# 2.2 FUNÇÕES PARA TRATAMENTO PARA SÉRIES TEMPORAIS

def generate_future_dates(start_date, days=180):
    """Gera uma lista de datas futuras."""
    return pd.date_range(start=start_date, periods=days, freq='D')

def pivotar_dados(
    df,
    col_unique='EAN',
    cols_to_pivot=['UNIDADES', 'RS_PC', 'RS_PR', 'RS_PPP'],
    data_type="yearmonth",
    id_cols=None,   # colunas que identificam a série além de col_unique (opcional)
    verbose=False
):
    """
    Versão robusta de pivotar_dados:
    - melt das colunas que começam com os prefixes em cols_to_pivot
    - extrai a parte numérica da data (YYYYMM ou YYYYMMDD)
    - converte VALOR para numérico (tratando vírgulas)
    - pivot e garante agregação final por id_cols + DT_EMISSAO
    - retorna DataFrame "long" pivotado por DT_EMISSAO com colunas QUANTIDADE, RS_PC, ...
    """

    df = df.copy()

    # Se id_cols não informadas, use col_unique + colunas comuns úteis (ajuste conforme seu dataset)
    if id_cols is None:
        # manter ao menos a chave e colunas frequentemente úteis (Product, UF)
        cand = []
        for c in ['PRODUCT_DESC', 'UF', 'EAN']:
            if c in df.columns and c != col_unique:
                cand.append(c)
        id_cols = [col_unique] + cand

    # identifica colunas que começam com os prefixes (case-insensitive)
    prefixes = tuple(cols_to_pivot)
    cols_to_melt = [c for c in df.columns if any(c.startswith(p) for p in prefixes)]
    if verbose:
        print(f"[pivotar_dados] colunas a melt: {len(cols_to_melt)}")

    # id_vars: todas as colunas exceto as a melt
    id_vars = [c for c in df.columns if c not in cols_to_melt]

    # melt
    df_melt = df.melt(id_vars=id_vars, value_vars=cols_to_melt,
                      var_name='VARIAVEL', value_name='VALOR')

    # extrai tipo (prefix) e a parte numérica da data (YYYYMM ou YYYYMMDD) mais flexível
    pattern_prefix = r'^(' + '|'.join(map(re.escape, cols_to_pivot)) + r')'
    pattern_date = r'(\d{6,8})'  # procura 6 ou 8 dígitos em qualquer lugar

    def extrair_tipo(var):
        m = re.search(pattern_prefix, var)
        return m.group(1) if m else None

    def extrair_data(var):
        m = re.search(pattern_date, var)
        return m.group(1) if m else None

    df_melt['TIPO'] = df_melt['VARIAVEL'].astype(str).apply(extrair_tipo)
    df_melt['DT_EMISSAO_RAW'] = df_melt['VARIAVEL'].astype(str).apply(extrair_data)

    # remove linhas sem match
    df_melt = df_melt.dropna(subset=['TIPO', 'DT_EMISSAO_RAW']).copy()
    if df_melt.empty:
        raise ValueError("Nenhuma coluna correspondente aos prefixes em 'cols_to_pivot' foi encontrada.")

    # limpar VALOR: substituir vírgula por ponto e converter para numérico
    def to_numeric_val(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, str):
            # remover espaços e pontos de milhar possíveis, trocar vírgula por ponto
            s = x.strip()
            # se houver vírgula e ponto, inferir: se ponto aparece antes da vírgula, assume-se formatação BR (1.234,56)
            if ',' in s and '.' in s:
                # remover pontos de milhar
                s = s.replace('.', '')
                s = s.replace(',', '.')
            else:
                s = s.replace(',', '.')
            # remover quaisquer símbolos não numéricos no início/fim
            s = re.sub(r'[^\d\.\-]', '', s)
            try:
                return float(s) if s != '' else np.nan
            except:
                return np.nan
        else:
            try:
                return float(x)
            except:
                return np.nan

    df_melt['VALOR'] = df_melt['VALOR'].apply(to_numeric_val)

    # converte DT_EMISSAO_RAW para datetime conforme data_type
    if data_type == "yearmonth":
        # algumas strings podem ter YYYYMM ou YYYYMMDD; pegamos primeiro 6 (YYYYMM)
        df_melt['DT_EMISSAO'] = pd.to_datetime(df_melt['DT_EMISSAO_RAW'].str[:6], format='%Y%m', errors='coerce')
    elif data_type == "yearmonthday":
        df_melt['DT_EMISSAO'] = pd.to_datetime(df_melt['DT_EMISSAO_RAW'].str[:8], format='%Y%m%d', errors='coerce')
    else:
        # tentativa genérica
        df_melt['DT_EMISSAO'] = pd.to_datetime(df_melt['DT_EMISSAO_RAW'], errors='coerce')

    # remove linhas com DT inválida
    df_melt = df_melt.dropna(subset=['DT_EMISSAO'])

    # pivot: agregamos explicitamente por id_cols + DT_EMISSAO
    group_index = [c for c in id_cols if c in df_melt.columns] + ['DT_EMISSAO']
    if verbose:
        print(f"[pivotar_dados] agrupando por: {group_index}")

    df_pivot = (
        df_melt
        .groupby(group_index + ['TIPO'], as_index=False)['VALOR']
        .sum()
        .pivot_table(index=group_index, columns='TIPO', values='VALOR', aggfunc='sum')
        .reset_index()
    )

    # Flatten columns (pivot_table cria columns TIPO)
    df_pivot.columns.name = None
    df_pivot = df_pivot.rename_axis(None, axis=1)

    # renomear colunas (UNIDADES -> QUANTIDADE, outros em maiúscula)
    rename_dict = {}
    for col in list(df_pivot.columns):
        if col.upper() in [c.upper() for c in cols_to_pivot]:
            key = col.upper()
            rename_dict[col] = 'QUANTIDADE' if key == 'UNIDADES' else key

    if rename_dict:
        df_pivot = df_pivot.rename(columns=rename_dict)

    # Por segurança, agregamos novamente por group_index (caso existam duplicatas residuais)
    numeric_cols = [c for c in df_pivot.columns if c not in group_index]
    if numeric_cols:
        agg_dict = {c: 'sum' for c in numeric_cols}
        df_pivot = df_pivot.groupby(group_index, as_index=False).agg(agg_dict)

    # ordena por chave e data
    df_pivot = df_pivot.sort_values(group_index).reset_index(drop=True)

    return df_pivot

def pivotar_por_categoria(
    df,
    col_categoria='EAN',
    cols_to_pivot=['UNIDADES', 'RS_PC', 'RS_PR', 'RS_PPP'],
    data_type="yearmonth",
    id_cols=None,
    mostrar_preview=False
):
    resultados = {}
    valores_unicos = df[col_categoria].unique()

    print(f"🔹 Iniciando processamento por '{col_categoria}' ({len(valores_unicos)} valores únicos)...")

    for valor in tqdm(valores_unicos, desc=f"Processando {col_categoria}", unit="categoria"):
        df_cat = df[df[col_categoria] == valor].copy()
        if not df_cat.empty:
            try:
                df_pivot = pivotar_dados(
                    df=df_cat,
                    col_unique=col_categoria,
                    cols_to_pivot=cols_to_pivot,
                    data_type=data_type,
                    id_cols=id_cols
                )
                resultados[valor] = df_pivot
                if mostrar_preview:
                    print(f"\n📦 {col_categoria}: {valor}")
                    print(df_pivot.head())
            except Exception as e:
                print(f"⚠️ Erro ao processar {col_categoria}={valor}: {e}")

    print(f"\n✅ Processamento concluído! ({len(resultados)} DataFrames gerados)")
    return resultados
