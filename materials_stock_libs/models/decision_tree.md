# Estrutura Hierárquica do Módulo de Regressão

O módulo ***decision_tree.py*** fornece uma classe unificada (`DecisionTreeModels`) para treinar, avaliar e fazer previsões usando modelos de regressão clássicos (Regressão Linear, Random Forest e LightGBM), com foco em dados segmentados e validação de séries temporais.

---

## Funções Auxiliares (Nível Raiz)

Funções utilitárias de suporte.

* ### **`normalize_horizon(...)`**
    * **Objetivo:** Padronizar strings de horizonte de tempo (ex: '90 days' para '90D').
    * **Etapa do Pipeline:** `[PRÉ-PROCESSAMENTO (Helper)]`

---

## Classe Principal: `DecisionTreeModels`

Classe que encapsula a lógica de filtragem de dados, divisão temporal, treinamento, otimização e inferência para diferentes modelos de regressão.

* ### **`__init__(...)`**
    * **Objetivo:** Inicializar a classe, armazenando o DataFrame principal, os nomes das colunas relevantes (data, alvo, filtro), o valor do segmento (se houver) e a *seed* de aleatoriedade.
    * **Etapa do Pipeline:** `[SETUP]`

* ### **`time_train_test_split(...)`**
    * **Objetivo:** Método estático para dividir um DataFrame temporal. Suporta uma divisão *holdout* simples (treino/teste) ou uma validação *walk-forward* (gerando múltiplos splits).
    * **Etapa do Pipeline:** `[PRÉ-PROCESSAMENTO (Split de Dados)]`

* ### **`_validate_str(...)`**
    * **Objetivo:** Método de utilidade interna para garantir que os identificadores de segmento sejam strings.
    * **Etapa do Pipeline:** `[PRÉ-PROCESSAMENTO (Helper)]`

* ### **`filter_segment(...)`**
    * **Objetivo:** Filtrar o DataFrame principal para isolar os dados de um único segmento (ex: `UF == 'CE'`). Essencial antes de treinar um modelo específico para esse segmento.
    * **Etapa do Pipeline:** `[PRÉ-PROCESSAMENTO (Filtro de Dados)]`

* ### **`train_model(...)`**
    * **Objetivo:** Orquestrar o treinamento de um modelo. Ele filtra os dados para o segmento, converte datas em *timestamps* ordinais (para uso em modelos não-temporais) e treina o modelo escolhido ('linear', 'rf' ou 'lgbm').
    * **Etapa do Pipeline:** `[TREINAMENTO]`

* ### **`predict_future(...)`**
    * **Objetivo:** Realizar inferência. Carrega um modelo previamente treinado para um segmento e o utiliza para prever valores para um conjunto de datas futuras.
    * **Etapa do Pipeline:** `[INFERÊNCIA]`

* ### **`optimize_model(...)`**
    * **Objetivo:** Encontrar os melhores hiperparâmetros para um tipo de modelo ('linear', 'rf', 'lgbm') usando **Optuna**. Utiliza `TimeSeriesSplit` para validar os parâmetros de forma temporalmente consciente, minimizando o RMSE médio.
    * **Etapa do Pipeline:** `[TREINAMENTO / AVALIAÇÃO (Otimização)]`

# Exemplo de uso da classe:

```python
    ###################################################
    ### IMPORTS
    ###################################################

    import pandas as pd
    import numpy as np
    import optuna
    import lightgbm as lgb

    from typing import Generator, Tuple
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_squared_error
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor

    import matplotlib.pyplot as plt
    from datetime import datetime, timedelta
    from ..utils import make_synthetic_data
 
    ###################################################
    # 1️. GERAÇÃO DE DADOS TEMPORAIS SINTÉTICOS
    ###################################################

    df = make_synthetic_data()

    print(df.head())

    ###################################################
    # 2. INSTANCIAR E PREPARAR O MODELO
    ###################################################

    # Instancia a classe principal
    trainer = DecisionTreeModels(
        df=df,
        date_col='DT_EMISSAO',
        target_col='UNIDADES',
        filter_col='UF',
        segment_value='CE',
        seed=42
    )

    ###################################################
    # 3️. TREINAR UM MODELO
    ###################################################

    # Escolha o tipo de modelo: 'linear', 'rf', ou 'lgbm'
    model_type = 'lgbm'   # altere para 'rf' ou 'linear' se quiser testar outros

    model = trainer.train_model(
        filter_col='UF',
        segment_value='CE',
        model_type=model_type
    )

    print(f"✅ Modelo '{model_type}' treinado com sucesso!")

    ###################################################
    # 4️. FAZER PREVISÕES FUTURAS (3 MESES)
    ###################################################

    future_dates = pd.date_range(df['DT_EMISSAO'].max() + pd.offsets.MonthBegin(1), periods=3, freq='MS')
    forecast = trainer.predict_future(segment_value='CE', future_dates=future_dates)

    print(forecast)

    ###################################################
    # 5️. VISUALIZAÇÃO DOS RESULTADOS
    ###################################################

    plt.figure(figsize=(12,5))
    plt.plot(df['DT_EMISSAO'], df['UNIDADES'], label='Histórico', color='black', alpha=0.7)
    plt.plot(forecast['ds'], forecast['pred'], 'o--', label=f'Previsão ({model_type.upper()})', color='tab:blue')
    plt.axvline(df['DT_EMISSAO'].iloc[-1], color='gray', linestyle='--', alpha=0.7)
    plt.title(f"Forecast com {model_type.upper()} — Exemplo de Séries Temporais")
    plt.xlabel("Data")
    plt.ylabel("Unidades")
    plt.legend()
    plt.show()

    ###################################################
    # 6️. (OPCIONAL) — OTIMIZAÇÃO DE HIPERPARÂMETROS COM OPTUNA
    ###################################################

    # ⚠️ Rodar esse trecho leva alguns minutos dependendo do modelo
    # Exemplo para Random Forest

    best_params = trainer.optimize_model(
        filter_col='UF',
        segment_value='CE',
        model_type='rf',     # ou 'lgbm' ou 'linear'
        n_trials=10,         # aumente para busca mais profunda
        n_splits=3
    )

    print("\nMelhores hiperparâmetros encontrados:")
    print(best_params)

    # Retreinar o modelo caso melhores parâmetros
    model = trainer.train_model(
        filter_col='UF',
        segment_value='CE',
        model_type=model_type,
        **best_params
    )

    ###################################################
    # 7. SALVANDO E CARREGANDO O MODELO
    ###################################################

    # Salvar o modelo treinado
    import joblib
    joblib.dump(trainer.models['CE'], 'lightgbm_CE_model.pkl')

    # Carregar novamente
    loaded_model = joblib.load('lightgbm_CE_model.pkl')
```