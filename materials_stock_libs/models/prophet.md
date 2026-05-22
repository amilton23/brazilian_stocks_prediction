# Estrutura Hierárquica do Módulo de Prophet
## Visão Geral

O módulo `prophet.py` implementa um pipeline completo de previsão temporal usando:

- **Prophet** (modelo base)  
- **LSTM** (para modelagem dos resíduos não capturados pelo Prophet)  
- **Pipeline híbrido** → *yhat_final = yhat_prophet + residual_lstm*

Totalmente preparado para:
- Treinamento
- Validação *walk-forward*
- Otimização com Optuna
- Inferência em produção
- Salvamento/Carregamento de modelos (MLOps-ready)

---

## Estrutura Hierárquica do Módulo

```python
prophet.py
├── Funções auxiliares
│ └── plot_prophet
│
├── Classe LSTMResidualModel
│ ├── init
│ └── forward
│
└── Classe ProphetModels
├── init
├── time_train_test_split_monthly
├── train_prophet
├── optimize_prophet_hyperparams
├── preprocess_features
├── fit_scaler
├── create_sequences
├── walk_forward_train
├── infer_prophet
├── infer_hybrid
├── save_models
└── load_models
```

---

## Funções Auxiliares

### `plot_prophet(df, test_size=0.2, is_sample=True, verbose=False)`

#### **Objetivo**
Plotar previsões de forma automática, detectando a coluna `yhat` e adicionando linha de corte entre treino e teste.

#### **Principais características**
- Detecta automaticamente `yhat`
- Ajusta eixo temporal por mês
- Desenha linha vertical representando `train/test split`
- Usada tanto para Prophet puro quanto para modelo híbrido

#### **Etapa do pipeline**
`[AVALIAÇÃO / VISUALIZAÇÃO]`

---

### `make_synthetic_monthly_data(periods=36, seed=42)`

#### **Objetivo**
Criar um conjunto de dados falsos (sintéticos) com tendência e sazonalidade para testes e demonstração.

#### **Principais características**
- Permite definir o número de períodos (padrão: 36 meses)
- Inclui tendência crescente e sazonalidade
- Gera valores aleatórios com distribuição normal
- Pode ser ajustado para simular diferentes cenários

#### **Etapa do pipeline**
`[PRÉ-PROCESSAMENTO (Criação de Dados)]`

---

## Classe `LSTMResidualModel`

Camada neural PyTorch especializada em modelar o **resíduo do Prophet**.

### `__init__(num_features, cat_cardinalities, embed_dims, hidden_size, num_layers, dropout)`

#### Objetivo
Definir:
- Camada de embedding para categóricos
- LSTM
- Camada densa final
- Inicialização robusta dos pesos (Xavier)

---

### `forward(x_num, x_cat)`

#### Objetivo
Executar o fluxo forward:
1. Concatena features numéricas e embeddings categóricos  
2. Passa pela LSTM  
3. Saída: predição do resíduo (`y_residual_hat`)

---

## Classe Principal: `ProphetModels`

Responsável por todo o pipeline de previsões.

---

### `__init__(...)`

#### Objetivo
Configurar ambiente:
- Seeds determinísticos
- Caminhos de modelos
- Variáveis internas:
  - `prophet_model_`
  - `lstm_model_`
  - `encoders_`
  - `scaler_`
  - `pipeline_mode_` (`prophet` | `hybrid`)

#### Etapa
`[SETUP]`

---

### `time_train_test_split_monthly(...)`

#### Objetivo
Gerar splits temporais mensais (holdout ou walk-forward).

#### Características
- Validação rigorosa
- Suporte a:
  - `test_months`
  - `n_splits`
  - `step_size`

#### Etapa
`[PRÉ-PROCESSAMENTO / SPLIT]`

---

### `train_prophet(train_df, **prophet_kwargs)`

#### Objetivo
Treinar o modelo Prophet usando parâmetros personalizados.

#### Destaques
- Armazena internamente em `self.prophet_model_`
- Suporte a logging via `verbose`
- Preparado para produção

#### Etapa
`[TREINAMENTO PROPHET]`

---

### `optimize_prophet_hyperparams(...)`

#### Objetivo
Otimizar Prophet com **Optuna**.

#### Características
- Usa validação cruzada
- Minimiza RMSE
- Atualiza automaticamente `self.prophet_model_`

#### Etapa
`[OTIMIZAÇÃO / TREINAMENTO]`

---

### `preprocess_features(...)`

#### Objetivo
Preparar dataset para modelo híbrido:
- Label Encoding (opcional)
- Sazonalidade: `month_sin` e `month_cos`
- Converte coluna `ds` para datetime

#### Etapa
`[FEATURE ENGINEERING]`

---

### `fit_scaler(df, numerical_cols, scaler_class=MinMaxScaler)`

#### Objetivo
Ajustar scaler e armazenar em `self.scaler_`.

#### Etapa
`[NORMALIZAÇÃO]`

---

### `create_sequences(...)`

#### Objetivo
Gerar janelas (`window_size`) para LSTM.

#### Output
- Tensores PyTorch para:
  - X_num  
  - X_cat  
  - y_residual  

#### Etapa
`[FORMATAÇÃO PARA LSTM]`

---

### `walk_forward_train(...)`

#### Objetivo
Treinar a parte híbrida Prophet+LSTM em esquema walk-forward.

#### Características principais
- Suporte a incremental training
- Usa yhat já existente (`use_existing_yhat=True`)
- Seleciona melhor modelo global
- Calcula RMSE e MAE por split

#### Etapa
`[TREINAMENTO HÍBRIDO]`

---

### `infer_prophet(...)`

#### Objetivo
Gerar:
- `yhat` (previsão Prophet)
- `residual` (erro base)

#### Etapa
`[INFERÊNCIA PROPHET PURO]`

---

### `infer_hybrid(...)`

#### Objetivo
Gerar predições do pipeline híbrido.

#### Output
`df_pred` contendo:
- `ds`
- `yhat_prophet`
- `yhat` (final híbrida)
- `residual_pred`
- `y` (real, se existente)

#### Regras especiais
- Resíduos negativos corrigidos para zero
- Janelamento dinâmico via `deque`

#### Etapa
`[INFERÊNCIA HÍBRIDA]`

---

### `save_models(verbose=False)`

#### Objetivo
Salvar:
- Prophet (`json`)
- LSTM (`.pth`)
- Encoders (`.pkl`)
- Scaler (`.pkl`)

#### Etapa
`[SERIALIZAÇÃO / DEPLOY]`

---

### `load_models(...)`

#### Objetivo
Carregar componentes salvos e reconstruir pipeline.

#### Etapa
`[SETUP DE INFERÊNCIA]`

---

# Exemplo de Uso Completo — ProphetModels (Pipeline Prophet + LSTM)

```python
import pandas as pd
import numpy as np
import random
import hashlib
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

from materials_stock_libs.models.prophet import (
    ProphetModels,
    LSTMResidualModel,
    plot_prophet
)

# ============================================================
# 1. Dados sintéticos realistas
# ============================================================
def make_realistic_external_sales():
    produtos = ["GLICONIL (MDQ)", "PANADOL", "NEOSALDINA"]
    ufs = ["SP", "RJ", "MG", "BA"]
    meses = pd.date_range("2022-01-01", periods=24, freq="MS")

    registros = []
    for prod in produtos:
        eans = [str(random.randint(78910000, 78999999)) for _ in range(3)]

        for uf in ufs:
            for dt in meses:
                qtd = (
                    100
                    + np.sin(dt.month / 2) * 20
                    + np.random.randn() * 8
                    + meses.get_loc(dt) * 1.8
                )

                registros.append({
                    "PRODUCT_DESC": prod,
                    "UF": uf,
                    "DT_EMISSAO": dt,
                    "QUANTIDADE": qtd,
                    "RS_PC": qtd * random.uniform(1.2, 2.5),
                    "RS_PPP": qtd * random.uniform(0.8, 1.5),
                    "RS_PR": qtd * random.uniform(2.8, 4.0),
                    "EAN": random.choice(eans)
                })
    return pd.DataFrame(registros)

df_ext_pivot = make_realistic_external_sales()


# ============================================================
# 2. Agregação igual ao pipeline real
# ============================================================
df_agg = (
    df_ext_pivot
    .groupby(["PRODUCT_DESC", "UF", "DT_EMISSAO"], as_index=False)
    .agg({
        "QUANTIDADE": "sum",
        "RS_PC": "sum",
        "RS_PPP": "sum",
        "RS_PR": "sum",
        "EAN": ["first", lambda x: list(set(x))]
    })
)

df_agg.columns = [
    "PRODUCT_DESC", "UF", "DT_EMISSAO",
    "QUANTIDADE", "RS_PC", "RS_PPP", "RS_PR",
    "EAN", "EAN_LIST"
]

df_agg = (
    df_agg
    .drop_duplicates(subset=["PRODUCT_DESC", "UF", "DT_EMISSAO"])
    .sort_values(["PRODUCT_DESC", "UF", "DT_EMISSAO"])
    .reset_index(drop=True)
)

# ============================================================
# 3. Loop REALISTA por produto + UF
# ============================================================
produtos = df_agg["PRODUCT_DESC"].unique()
ufs = df_agg["UF"].unique()

min_points = 24
HORIZONTE_PREVISAO = 3
JANELAMENTO = 6
LSTM_EPOCHS = 5

for prod in produtos:
    for uf in ufs:
        print(f"\n=== Treinando {prod} | {uf} ===")

        df_prod = df_agg[(df_agg["PRODUCT_DESC"] == prod) & (df_agg["UF"] == uf)].copy()

        if df_prod.shape[0] < min_points:
            print(f"⚠️ Poucos dados para {prod}-{uf}")
            continue

        df_prod = df_prod.rename(columns={"DT_EMISSAO": "ds", "QUANTIDADE": "y"})

        # ====================================================
        # Instancia pipeline
        # ====================================================
        pm = ProphetModels(
            date_col="ds",
            target_col="y",
            seed=42,
            model_dir=f"modelos/{prod}_{uf}"
        )

        # ====================================================
        # Split temporal
        # ====================================================
        df_split = df_prod.rename(columns={"ds": "DT_EMISSAO"})
        train_df, test_df = pm.time_train_test_split_monthly(df_split, test_months=0.2)[0]
        train_df = train_df.rename(columns={"DT_EMISSAO": "ds"})
        test_df = test_df.rename(columns={"DT_EMISSAO": "ds"})

        # ====================================================
        # Treina Prophet
        # ====================================================
        best_params = dict(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=1.5
        )
        pm.train_prophet(train_df, **best_params)

        # ====================================================
        # Inferência interna para gerar resíduos
        # ====================================================
        forecast_full = pm.infer_prophet(train_df, horizon=HORIZONTE_PREVISAO)

        # ====================================================
        # Pré-processamento LSTM
        # ====================================================
        categorical_cols = []
        numerical_cols = ["residual", "month_sin", "month_cos"]

        df_pre, _ = pm.preprocess_features(forecast_full, categorical_cols)
        df_pre, _ = pm.fit_scaler(df_pre, numerical_cols, scaler_class=MinMaxScaler)

        # ====================================================
        # Treino do LSTM (híbrido)
        # ====================================================
        results_df, _ = pm.walk_forward_train(
            df=df_pre,
            numerical_cols=numerical_cols,
            categorical_cols=categorical_cols,
            cat_cardinalities=[],
            embed_dims=[],
            window_size=JANELAMENTO,
            horizon=HORIZONTE_PREVISAO,
            lstm_epochs=LSTM_EPOCHS,
            incremental=True,
            use_existing_yhat=True
        )

        pm.save_models()

        # ====================================================
        # Carregar pipeline — simulação de produção
        # ====================================================
        pm_prod = ProphetModels(
            date_col="ds",
            target_col="y",
            model_dir=f"modelos/{prod}_{uf}"
        )

        pm_prod.load_models(
            model_class=LSTMResidualModel,
            categorical_cols=categorical_cols,
            numerical_cols=numerical_cols
        )

        # ====================================================
        # Inferência Prophet + Hybrid
        # ====================================================
        forecast_prod = pm_prod.infer_prophet(df_prod, horizon=HORIZONTE_PREVISAO)

        df_pred = pm_prod.infer_hybrid(
            df=forecast_prod,
            forecast_all=forecast_prod,
            categorical_cols=categorical_cols,
            feature_cols=numerical_cols,
            window=JANELAMENTO
        )

        print(df_pred.tail())

        # ====================================================
        # Plot
        # ====================================================
        plot_prophet(df_pred)

print("\nPipeline completo finalizado!")
```