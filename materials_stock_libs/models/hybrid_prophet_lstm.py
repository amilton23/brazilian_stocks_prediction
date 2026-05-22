###############################################################
# PREVISÃO HÍBRIDA DE ESTOQUE: Prophet + LSTM (PyTorch)
###############################################################
# Autor: José Amilton Cardoso Filho | linkedin.com/in/amiltoncofh/
# Data: 2025-10-08
# GitHub: github.com/amilton23
# Objetivo: Prever estoque de produtos usando Prophet + LSTM Residual
# Frameworks: PyTorch, Prophet, Scikit-learn, Matplotlib, Pandas, Numpy
###############################################################

# Requisitos de environment e teste na raiz do repositório (env.yaml e README.md - seção 4).

import os
import random
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tqdm import tqdm
from collections import deque

from typing import Generator, Tuple, List

# Prophet
from prophet import Prophet

# PyTorch (LSTM)
import torch
import torch.nn as nn

# Scalers
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

#######################################################
# 0. Configurações iniciais
#######################################################

# Reprodutibilidade
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

#######################################################
# 1. Geração de dados mensais sintéticos
#######################################################
def make_synthetic_monthly_data(start='2023-05-01', periods=25, seed=SEED):
    """Cria uma série mensal (maio/2023 a maio/2025) com tendência e sazonalidade."""
    rng = pd.date_range(start=start, periods=periods, freq='MS')  # monthly start
    t = np.arange(periods)
    trend = 2 * t                              # tendência linear
    seasonal = 10 * np.sin(2 * np.pi * t / 12) # sazonalidade anual
    noise = np.random.normal(0, 3, size=periods)
    y = 100 + trend + seasonal + noise
    return pd.DataFrame({'ds': rng, 'y': y})

df = make_synthetic_monthly_data()
plt.figure(figsize=(10,4))
plt.plot(df['ds'], df['y'], marker='o')
plt.title('Série mensal sintética (maio/2023 a maio/2025)')
plt.show()



#######################################################
# 2. Split temporal com walk-forward mensal
#######################################################
def time_train_test_split_monthly(
    df: pd.DataFrame,
    test_months: int = 3,
    n_splits: int = None,
    step_size: int = None,
    walk_forward: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame] | Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
    """
    Divide dados temporais mensais para treino/teste ou validação walk-forward.
    
    Args:
        df (pd.DataFrame): dataframe com coluna 'ds' (datetime) e 'y'
        test_months (int): número de meses para teste
        n_splits (int, opcional): número de divisões no walk-forward
        step_size (int, opcional): avanço em meses
        walk_forward (bool): se True, gera janelas sequenciais (train, val)
        
    Returns:
        - Se walk_forward=False → (train_df, test_df)
        - Se walk_forward=True → generator (train_df, val_df)
    """
    if 'ds' not in df.columns:
        raise ValueError("O DataFrame precisa conter a coluna 'ds' (datetime).")

    df = df.sort_values('ds').reset_index(drop=True)
    n = len(df)
    if n < test_months:
        raise ValueError("Número de meses de teste maior que o total de períodos disponíveis.")

    if not walk_forward:
        split_idx = n - test_months
        train_df = df.iloc[:split_idx].reset_index(drop=True)
        test_df = df.iloc[split_idx:].reset_index(drop=True)
        return train_df, test_df

    # Walk-forward
    if n_splits is None and step_size is None:
        raise ValueError("Defina 'n_splits' ou 'step_size' para walk-forward.")

    if n_splits is not None:
        # garante que n_splits seja compatível com número de observações
        step_size = max(1, (n - test_months) // n_splits)

    for start in range(0, n - test_months, step_size):
        end_train = start + (n - test_months)
        if end_train >= n - test_months:
            break
        train_df = df.iloc[:end_train].reset_index(drop=True)
        val_df = df.iloc[end_train:end_train + test_months].reset_index(drop=True)
        yield train_df, val_df

#######################################################
# 3. Teste da função
#######################################################
train_df, test_df = time_train_test_split_monthly(df, test_months=3)
print(f"Treino: {train_df['ds'].iloc[0].date()} → {train_df['ds'].iloc[-1].date()} "
      f"({len(train_df)} meses)")
print(f"Teste:  {test_df['ds'].iloc[0].date()} → {test_df['ds'].iloc[-1].date()} "
      f"({len(test_df)} meses)")

print("\n===== Walk-forward validation (mensal) =====")
for i, (train_df, val_df) in enumerate(
    time_train_test_split_monthly(df, test_months=3, n_splits=4, walk_forward=True)
):
    print(f"Split {i+1}: Train até {train_df['ds'].iloc[-1].date()} | "
          f"Val de {val_df['ds'].iloc[0].date()} até {val_df['ds'].iloc[-1].date()}")

#######################################################
# 3. Treinar Prophet (captura tendência + sazonalidade)
#######################################################
def train_prophet(train_df, **prophet_kwargs):
    """Treina Prophet e retorna objeto treinado + forecast completo (inclui histórico)."""
    model = Prophet(**prophet_kwargs)
    model.fit(train_df[['ds','y']])
    # Retorne modelo; forecast será gerado para avaliações/predições posteriormente
    return model

prophet_params = dict(
    daily_seasonality=False,
    weekly_seasonality=False,
    yearly_seasonality=True,
    seasonality_mode='additive',
    changepoint_prior_scale=0.05
)

prophet_model = train_prophet(train_df, **prophet_params)

# Gera forecast sobre treino+test para visualização/resíduos
horizon = test_df.shape[0]
future_all = prophet_model.make_future_dataframe(periods=horizon, freq='MS', include_history=True)
forecast_all = prophet_model.predict(future_all)

# Plot: histórico vs fit do Prophet
plt.figure(figsize=(12,4))
plt.plot(train_df['ds'], train_df['y'], label='histórico (train)', color='black', alpha=0.6)
plt.plot(forecast_all['ds'], forecast_all['yhat'], label='prophet yhat', color='orange')
plt.axvline(train_df['ds'].iloc[-1], color='gray', linestyle='--', label='split')
plt.legend()
plt.title('Prophet - ajuste e forecast (yhat)')
plt.show()

#######################################################
# 4. Calcular resíduos (observado - prophet on train)
#    - Para treinar LSTM, usamos os resíduos do período de treino
#######################################################
# Obter yhat alinhado com séries históricas
# forecast_all contains both history and future. We'll align with original df by 'ds'
merged = train_df.merge(forecast_all[['ds','yhat']], on='ds', how='left')
merged['residual'] = merged['y'] - merged['yhat']

plt.figure(figsize=(12,3))
plt.plot(merged['ds'], merged['residual'], label='residual (train)')
plt.title('Resíduos (train)')
plt.show()

#######################################################
# 5. Preparar dataset janelado para LSTM (usamos apenas treino)
#    - inputs: lagged residuals + optional exogenous features (dayofweek, month, special flags)
#    - target: next-step residual
#######################################################

def preprocess_features(df, categorical_cols, fit_encoders=True, encoders=None):
    df = df.copy()

    # Label encode das categorias
    if fit_encoders:
        encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        # salvar encoders
        for col, le in encoders.items():
            joblib.dump(le, f"encoder_{col}.pkl")
    else:
        # usar encoders existentes
        for col in categorical_cols:
            le = encoders[col]
            df[col] = le.transform(df[col].astype(str))

    # Calendar features
    df['month'] = df['ds'].dt.month
    df['month_sin'] = np.sin(2*np.pi*(df['month']-1)/12)
    df['month_cos'] = np.cos(2*np.pi*(df['month']-1)/12)

    return df, encoders

categorial_cols = []  # Nenhuma categoria neste exemplo sintético
merged, encoders = preprocess_features(merged, categorical_cols=categorial_cols, fit_encoders=True)

#######################################################
### 6. Estruturação temporal para LSTM
#######################################################

# Normalização dos resíduos
scaler = MinMaxScaler(feature_range=(-1, 1))
residual_scaled = scaler.fit_transform(merged[["residual"]])

# SOMENTE NUMÉRICAS
def create_sequences(data, window=6):
    """Cria janelas temporais (X, y)"""
    xs, ys = [], []
    for i in range(len(data) - window):
        x = data[i:(i+window)]
        y = data[i+window]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

WINDOW = 6
X, y = create_sequences(residual_scaled, window=WINDOW)

# Split treino/teste
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Conversão para tensores
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


#######################################################
### 7. Treino e teste do modelo LSTM em PyTorch
#######################################################

# ##########################
# 7.1. Definição do modelo LSTM em PyTorch
# ##########################

class LSTMResidualModel(nn.Module):
    def __init__(self, num_features, cat_cardinalities, embed_dims, hidden_size=64, num_layers=2):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=card, embedding_dim=dim)
            for card, dim in zip(cat_cardinalities, embed_dims)
        ])
        total_embed_dim = sum(embed_dims)
        self.lstm = nn.LSTM(num_features + total_embed_dim, hidden_size,
                            num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x_num, x_cat):
        embeds = [emb(x_cat[:, :, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat([x_num] + embeds, dim=2)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def walk_forward_train(
    df: pd.DataFrame,
    encoders: dict,
    model_class: nn.Module,
    prophet_model,
    numerical_cols: list,
    categorical_cols: list,
    cat_cardinalities: list,
    embed_dims: list,
    window_size: int = 24,  # meses no treino
    horizon: int = 3,       # meses no teste
    lstm_epochs: int = 10,
    incremental: bool = True
):
    """
    Realiza walk-forward training incremental com Prophet + LSTM híbrido.

    Args:
        df (pd.DataFrame): dataframe com colunas ['ds', 'y', ...].
        encoders (dict): encoders fixos para variáveis categóricas.
        model_class (nn.Module): classe do modelo LSTM.
        prophet_model: modelo Prophet previamente treinado.
        numerical_cols (list): features numéricas.
        categorical_cols (list): features categóricas.
        cat_cardinalities (list): cardinalidades das categorias.
        embed_dims (list): dimensões dos embeddings.
        window_size (int): meses de treino por janela.
        horizon (int): meses de previsão (teste).
        lstm_epochs (int): número de épocas por janela.
        incremental (bool): se True, mantém pesos do modelo a cada janela.

    Returns:
        pd.DataFrame com métricas (RMSE, MAE) por janela temporal.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    # inicializa modelo e otimizador
    model = model_class(
        num_features=len(numerical_cols),
        cat_cardinalities=cat_cardinalities,
        embed_dims=embed_dims,
        hidden_size=64,
        num_layers=2
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Loop temporal (rolling window)
    for end_idx in range(window_size, len(df) - horizon):
        train_df = df.iloc[:end_idx].copy()
        test_df = df.iloc[end_idx:end_idx + horizon].copy()

        # ===============================
        # 1️⃣ Preprocessamento consistente
        # ===============================
        train_df, _ = preprocess_features(train_df, categorical_cols, fit_encoders=False, encoders=encoders)
        test_df, _ = preprocess_features(test_df, categorical_cols, fit_encoders=False, encoders=encoders)

        # ===============================
        # 2️⃣ Prophet — previsão base
        # ===============================
        prophet_model.fit(train_df[['ds', 'y']])
        future = prophet_model.make_future_dataframe(periods=horizon, freq='MS', include_history=False)
        forecast = prophet_model.predict(future)

        # ===============================
        # 3️⃣ Cálculo dos resíduos
        # ===============================
        merged = test_df.merge(forecast[['ds', 'yhat']], on='ds', how='left')
        merged['residual'] = merged['y'] - merged['yhat']

        # Adiciona novamente as features de calendário e categorias para merged
        merged, _ = preprocess_features(merged, categorical_cols, fit_encoders=False, encoders=encoders)

        # ===============================
        # 4️⃣ Preparar dados p/ LSTM
        # ===============================
        def encode_features(df_):
            # Numéricas
            X_num = torch.tensor(df_[numerical_cols].values, dtype=torch.float32).to(device)

            # Categóricas (se houver)
            if categorical_cols:
                X_cat = torch.tensor(np.stack([
                    df_[col].values for col in categorical_cols
                ], axis=1), dtype=torch.long).to(device)
            else:
                X_cat = torch.zeros((len(df_), 0), dtype=torch.long).to(device)

            # Target
            y = torch.tensor(df_['residual'].values, dtype=torch.float32).view(-1, 1).to(device)
            return X_num, X_cat, y

        X_num_train, X_cat_train, y_train = encode_features(train_df)
        X_num_test, X_cat_test, y_test = encode_features(merged)

        # ===============================
        # 5️⃣ Treinamento incremental
        # ===============================
        if not incremental:
            model = model_class(
                num_features=len(numerical_cols),
                cat_cardinalities=cat_cardinalities,
                embed_dims=embed_dims,
                hidden_size=64,
                num_layers=2
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(lstm_epochs):
            model.train()
            optimizer.zero_grad()
            out = model(X_num_train, X_cat_train)
            loss = criterion(out, y_train)
            loss.backward()
            optimizer.step()

        train_losses = []
        val_losses = []
        # Armazenar losses para visualização posterior (opcional)
        train_losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            val_out = model(X_num_test, X_cat_test)
            val_loss = criterion(val_out, y_test)
            val_losses.append(val_loss.item())

        # ===============================
        # 6️⃣ Previsão com LSTM residual
        # ===============================
        model.eval()
        with torch.no_grad():
            pred_residual = model(X_num_test, X_cat_test).cpu().numpy().flatten()

        hybrid_forecast = merged['yhat'].values + pred_residual
        y_true = merged['y'].values

        # ===============================
        # 7️⃣ Avaliação
        # ===============================
        rmse = np.sqrt(mean_squared_error(y_true, hybrid_forecast))
        mae = mean_absolute_error(y_true, hybrid_forecast)
        results.append({
            'train_end': train_df['ds'].iloc[-1],
            'test_start': test_df['ds'].iloc[0],
            'test_end': test_df['ds'].iloc[-1],
            'rmse': rmse,
            'mae': mae
        })

        print(f"[{test_df['ds'].iloc[0].date()} → {test_df['ds'].iloc[-1].date()}] "
              f"RMSE={rmse:.4f} | MAE={mae:.4f}")

    return pd.DataFrame(results), model, train_losses, val_losses

# ##########################
# 7.2. Pipeline de treino
# ##########################

categorical_cols = ['UF', 'canal_venda']  # ajuste conforme seu dataset
numerical_cols = ['residual', 'month_sin', 'month_cos']

merged, encoders = preprocess_features(merged, categorical_cols, fit_encoders=True)
scaler = MinMaxScaler(feature_range=(-1, 1))
merged[numerical_cols] = scaler.fit_transform(merged[numerical_cols])

WINDOW = 6
X_num, X_cat, y = create_sequences(merged, numerical_cols, categorical_cols, window=WINDOW)

# Split
train_size = int(len(X_num) * 0.8)
X_num_train, X_num_test = X_num[:train_size], X_num[train_size:]
X_cat_train, X_cat_test = X_cat[:train_size], X_cat[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Tensors
X_num_train = torch.tensor(X_num_train, dtype=torch.float32)
X_num_test = torch.tensor(X_num_test, dtype=torch.float32)
X_cat_train = torch.tensor(X_cat_train, dtype=torch.long)
X_cat_test = torch.tensor(X_cat_test, dtype=torch.long)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# ##########################
# 7.3. Treinamento do LSTM
# ##########################

cat_cardinalities = [len(encoders[col].classes_) for col in categorical_cols]
embed_dims = [min(50, int(card**0.25) * 4) for card in cat_cardinalities]

results_df, model, train_losses, val_losses = walk_forward_train(
    df=df,
    encoders=encoders,
    model_class=LSTMResidualModel,
    prophet_model=Prophet(**prophet_params),
    numerical_cols=numerical_cols,
    categorical_cols=categorical_cols,
    cat_cardinalities=cat_cardinalities,
    embed_dims=embed_dims,
    window_size=24,
    horizon=3,
    lstm_epochs=10,
    incremental=True
)

#######################################################
# 7.3. Avaliação e inferência LSTM (com embeddings)
#######################################################

model.eval()
with torch.no_grad():
    pred_test = model(X_num_test, X_cat_test).cpu().numpy()

# Inversão da normalização (resíduos)
pred_test_inv = scaler.inverse_transform(pred_test)
y_test_inv = scaler.inverse_transform(y_test)

# RMSE do LSTM (resíduos)
rmse_lstm = np.sqrt(mean_squared_error(y_test_inv, pred_test_inv))
print(f"\n✅ RMSE LSTM (resíduos): {rmse_lstm:.4f}")

# Visualização do histórico de perda
plt.figure(figsize=(8,4))
plt.plot(train_losses, label='Treino', color='blue')
plt.plot(val_losses, label='Validação', color='orange', linestyle='--')
plt.title('Evolução do Loss durante o Treinamento (LSTM)')
plt.xlabel('Época')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

#######################################################
# 8. INFERÊNCIA HÍBRIDA (walk-forward sobre o período de teste)
#    - Para cada passo futuro:
#         (a) obter Prophet yhat (já temos forecast_all)
#         (b) criar features para LSTM (últimos WINDOW steps), escalar e prever residual
#         (c) soma: final_forecast = yhat + pred_residual
#######################################################

from collections import deque

# 8.1. Merge com previsão Prophet
full = forecast_all[['ds', 'yhat']].merge(df[['ds', 'y']].drop_duplicates(), on='ds', how='left')
full['residual'] = full['y'] - full['yhat']
full, _ = preprocess_features(full, categorical_cols, fit_encoders=False, encoders=encoders)

# 8.2. Aplicar label encoders (os mesmos do treino)
for col, encoder in encoders.items():
    full[col] = encoder.transform(full[col].astype(str))

# 8.3. Escalonar features numéricas
feature_cols = ['residual', 'month_sin', 'month_cos']
full[feature_cols] = full[feature_cols].fillna(0)
full[feature_cols] = scaler.transform(full[feature_cols])

# 8.4. Inicializar janela deslizante (últimos WINDOW passos antes do horizonte)
preds_hybrid, preds_prophet = [], []
dates_horizon = full['ds'].iloc[-horizon:].values
start_idx = len(full) - horizon - WINDOW
rolling_window_num = deque(maxlen=WINDOW)
rolling_window_cat = deque(maxlen=WINDOW)

# 8.5. Inicializa janelas numéricas e categóricas
for i in range(start_idx, start_idx + WINDOW):
    rolling_window_num.append(full[feature_cols].iloc[i].values)
    rolling_window_cat.append(full[categorical_cols].iloc[i].values)

# 8.6. Inferência iterativa (walk-forward)
model.eval()
for i in range(len(dates_horizon)):
    # Preparar entrada
    X_num_input = np.array(rolling_window_num).reshape(1, WINDOW, len(feature_cols))
    X_cat_input = np.array(rolling_window_cat).reshape(1, WINDOW, len(categorical_cols))

    # Converter em tensores
    X_num_tensor = torch.tensor(X_num_input, dtype=torch.float32)
    X_cat_tensor = torch.tensor(X_cat_input, dtype=torch.long)

    # Predição residual
    with torch.no_grad():
        pred_res = model(X_num_tensor, X_cat_tensor).cpu().numpy().reshape(-1)[0]

    # Combina previsão Prophet + LSTM residual
    yhat = full['yhat'].iloc[start_idx + WINDOW + i]
    hybrid_forecast = yhat + pred_res
    preds_hybrid.append(hybrid_forecast)
    preds_prophet.append(yhat)

    # Atualiza janelas (próximo passo)
    scaled_pred_res = scaler.transform([[pred_res, *full[["month_sin", "month_cos"]].iloc[start_idx + WINDOW + i].values]])[0]
    rolling_window_num.append(scaled_pred_res)
    rolling_window_cat.append(full[categorical_cols].iloc[start_idx + WINDOW + i].values)

# Conversão para arrays e datas
preds_hybrid = np.array(preds_hybrid)
preds_prophet = np.array(preds_prophet)
dates_horizon = pd.to_datetime(dates_horizon)

#######################################################
# 9. Avaliação
#######################################################
# True y on horizon
y_true = full['y'].iloc[-horizon:].values
mask = ~np.isnan(y_true)

rmse_prophet = np.sqrt(mean_squared_error(y_true[mask], preds_prophet[mask]))
rmse_hybrid = np.sqrt(mean_squared_error(y_true[mask], preds_hybrid[mask]))
mae_prophet = mean_absolute_error(y_true[mask], preds_prophet[mask])
mae_hybrid = mean_absolute_error(y_true[mask], preds_hybrid[mask])

print(f"\n Avaliação Híbrida Prophet + LSTM:")
print(f"   Prophet RMSE: {rmse_prophet:.4f} | Hybrid RMSE: {rmse_hybrid:.4f}")
print(f"   Prophet MAE:  {mae_prophet:.4f} | Hybrid MAE:  {mae_hybrid:.4f}")

#######################################################
# 10. Visualização completa (histórico + prophet + hybrid)
#######################################################
plt.figure(figsize=(14,6))
# plot history
plt.plot(full['ds'], full['y'], label='Observado (hist)', color='black', alpha=0.6)
# prophet yhat full
plt.plot(full['ds'], full['yhat'], label='Prophet (yhat)', color='orange', alpha=0.9)
# hybrid forecast only for horizon
plt.plot(dates_horizon, preds_hybrid, label='Hybrid (Prophet + LSTM residual)', color='tab:blue', linestyle='--', linewidth=2)
# mark vertical line for split
plt.axvline(train_df['ds'].iloc[-1], color='gray', linestyle='--', label='Split')
plt.legend()
plt.title('Histórico + Prophet + Hybrid forecast (horizon)')
plt.xlabel('ds')
plt.ylabel('y / forecast')
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

#######################################################
# 11. Save models
#######################################################
MODELS_PATH = os.path.join(os.getcwd(), 'models')
os.makedirs(MODELS_PATH, exist_ok=True)

# Prophet
prophet_model.save(os.path.join(MODELS_PATH, "prophet_model.json"))

# PyTorch
torch.save(model.state_dict(), os.path.join(MODELS_PATH, 'lstm_residual_embeddings.pth'))

# Encoders e scaler
for col, le in encoders.items():
    joblib.dump(le, f"{MODELS_PATH} + '/encoder_{col}.pkl'")
joblib.dump(scaler, f'{MODELS_PATH}/scaler.pkl')

print(f"\n💾 Modelos e artefatos salvos em {MODELS_PATH}")

# Para carregar (.py de inferência):
# model.load_state_dict(torch.load('/tmp/lstm_residual_model.pth'))
# model.eval()