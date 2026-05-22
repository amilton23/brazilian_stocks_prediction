"""Módulo para modelos de previsão de estoque/vendas utilizando modelos baseados em modelos de séries temporais."""

############################################################################################################
### 1. IMPORTS
############################################################################################################

# COMMON
import os
import random
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import time
from tqdm import tqdm
from typing import List, Tuple
import copy

# 1. PROPHET
from prophet.diagnostics import cross_validation, performance_metrics
from prophet import Prophet
import optuna
from pandas.tseries.frequencies import to_offset
from prophet.serialize import model_to_json, model_from_json

# PyTorch (LSTM)
import torch
import torch.nn as nn

# Scalers
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

# from ..utils import make_synthetic_data # Removido para o exemplo funcionar

import logging
logger = logging.getLogger("forecast_pipeline")
# Silencia logs verbosos
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
logging.getLogger('prophet').setLevel(logging.ERROR) 
logger.propagate = False

import warnings
warnings.filterwarnings("ignore")

def plot_prophet(df, test_size: float = 0.2, is_sample: bool = True, verbose: bool = False):
    """
    Plota as previsões do modelo (Prophet ou Híbrido) automaticamente.
    Detecta a coluna 'yhat' como previsão e desenha uma linha vertical
    separando o conjunto de treino e teste (por padrão 80% treino, 20% teste).

    Parâmetros:
        df : pd.DataFrame
            DataFrame contendo as colunas ['ds', 'y', 'yhat'].
        test_size : float, padrão 0.2
            Proporção do dataset considerada como teste (para traçar linha divisória).
    """
    # --- Validação básica ---
    if 'ds' not in df.columns or 'y' not in df.columns:
        raise ValueError("O DataFrame deve conter colunas 'ds' e 'y'.")
    if 'yhat' not in df.columns:
        raise ValueError("A coluna 'yhat' (previsão do modelo) não foi encontrada no DataFrame.")
    if not 0 < test_size < 1:
        raise ValueError("O parâmetro 'test_size' deve estar entre 0 e 1 (ex.: 0.2 para 20%).")

    # --- Preparação dos dados ---
    df = df.copy().sort_values('ds')

    # --- Gráfico ---
    plt.figure(figsize=(14, 6))

    # Série real observada
    plt.plot(df['ds'], df['y'], label='Histórico observado', color='black', alpha=0.6)

    # Previsão detectada automaticamente (Prophet ou Híbrido)
    plt.plot(
        df['ds'],
        df['yhat'],
        label='Previsão do modelo',
        color='tab:blue',
        linestyle='--',
        linewidth=2,
        marker='o'
    )
 
    if is_sample:
        total_points = len(df)
        split_index = int(total_points * (1 - test_size))
        split_date = df['ds'].iloc[split_index]

        if verbose:
            print(f"📊 Tamanho total do dataset: {total_points} pontos")
            print(f"📈 Linha divisória treino/teste em: {split_date.date()} ({int((1-test_size)*100)}% treino / {int(test_size*100)}% teste)")

        # Linha vertical de separação treino/teste
        plt.axvline(split_date, color='red', linestyle='--', linewidth=1.8, label='Divisão Treino/Teste')

    # Eixo X formatado mensalmente
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b/%Y'))
    plt.xticks(rotation=45)

    # Configurações gerais
    plt.title('Previsão Temporal do modelo')
    plt.xlabel('Data (Mensal)')
    plt.ylabel('Quantidade Prevista')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

############################################################################################################
### 2. CLASSE DO MODELO LSTM (NÍVEL DO MÓDULO)
############################################################################################################

class LSTMResidualModel(nn.Module):
    def __init__(self, num_features, cat_cardinalities, embed_dims, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = True

        # Embeddings categóricos
        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality, emb_dim)
            for cardinality, emb_dim in zip(cat_cardinalities, embed_dims)
        ]) if cat_cardinalities else None

        total_input_dim = num_features
        if self.embeddings is not None:
            total_input_dim += sum(embed_dims)

        # LSTM com dropout
        self.lstm = nn.LSTM(
            input_size=total_input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Camada densa final
        self.fc = nn.Linear(hidden_size, 1)

        # Inicialização de pesos
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(self, x_num, x_cat=None):
        # Garante formato 3D
        if x_num.dim() == 2:
            x_num = x_num.unsqueeze(1)
        elif x_num.dim() == 1:
            x_num = x_num.unsqueeze(0).unsqueeze(0)

        # Embeddings categóricos
        if self.embeddings is not None and x_cat is not None and x_cat.numel() > 0:
            if x_cat.dim() == 2:
                x_cat = x_cat.unsqueeze(1)
            embeds = [emb(x_cat[:, :, i]) for i, emb in enumerate(self.embeddings)]
            embeds = torch.cat(embeds, dim=-1)
            x = torch.cat([x_num, embeds], dim=-1)
        else:
            x = x_num

        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out


############################################################################################################
### 3. CLASSE DO PIPELINE PROPHET
############################################################################################################

class ProphetModels:
    def __init__(self, date_col:str='DT_EMISSAO', target_col:str='QUANTIDADE', seed=42, model_dir='trained_models'):
        """
        Inicializa o pipeline, definindo caminhos e atributos de estado.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        self.seed = seed
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        self.date_col = date_col
        self.target_col = target_col

        # --- Atributos de Estado (Padrão Scikit-learn) ---
        # Estes atributos são inicializados como None e preenchidos por métodos .fit() ou .load()
        self.prophet_model_ = None
        self.lstm_model_ = None
        self.encoders_ = None
        self.scaler_ = None
        self.pipeline_mode_ = "prophet" # 'prophet' ou 'hybrid'
        self.prophet_config_ = {}

    #######################################################
    ### 1. Geração e Split de Dados
    #######################################################
    
    def time_train_test_split_monthly(self, 
                                      df: pd.DataFrame, 
                                      date_col: str = "DT_EMISSAO",
                                      test_months: float = 0.2,
                                      n_splits: int = None,
                                      step_size: int = None,
                                      walk_forward: bool = False
                                     ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Divide dados temporais mensais em treino/teste (holdout ou walk-forward).
        (Método mantido como na sua última versão)
        """
        
        # --- Validação Básica ---
        if date_col not in df.columns:
            raise ValueError(f"❌ O DataFrame precisa conter a coluna temporal '{date_col}'")

        df_sorted = df.sort_values(date_col).reset_index(drop=True)
        n = len(df_sorted)

        
        if n == 0:
            raise ValueError("❌ O DataFrame está vazio.")
        if 1.0 <= test_months:
            raise ValueError(f"❌ O DataFrame ({n} linhas) é menor ou igual ao número de meses de teste ({test_months}).")

        splits_list = []

        # --- Modo HOLDOUT (Padrão) ---
        if not walk_forward:
            split_idx = round(n*(1 - test_months))
            train_df = df_sorted.iloc[:split_idx].reset_index(drop=True)
            test_df = df_sorted.iloc[split_idx:].reset_index(drop=True)
            
            #print(f"✅ Split holdout criado: treino={len(train_df)}, teste={len(test_df)}")
            
            splits_list.append((train_df, test_df))
            return splits_list

        # --- Modo WALK-FORWARD ---
        else:
            if n_splits is None and step_size is None:
                raise ValueError("Para walk_forward=True, defina 'n_splits' ou 'step_size'.")

            if n_splits:
                step_size = max(1, (round(n*(1 - test_months))) // n_splits)
            
            if step_size is None:
                step_size = 1 

            min_train_size = round(n*test_months) 
            
            for end_train_idx in range(min_train_size, round(n*(1 - test_months)) + 1, step_size):
                
                start_test_idx = end_train_idx
                end_test_idx = end_train_idx + round(n*test_months) 
                
                if end_test_idx > n:
                    continue 

                train_df = df_sorted.iloc[0:start_test_idx].reset_index(drop=True)
                test_df = df_sorted.iloc[start_test_idx:end_test_idx].reset_index(drop=True)
                
                if not train_df.empty and not test_df.empty:
                    splits_list.append((train_df, test_df))

            if not splits_list:
                print(f"⚠️ Nenhum split walk-forward foi gerado com os parâmetros: test_months={round(n*test_months) }, step_size={step_size}")
            else:
                print(f"✅ {len(splits_list)} splits walk-forward criados.")

            return splits_list


    #######################################################
    ### 2️. Prophet
    #######################################################
    
    def train_prophet(self, train_df, verbose=False, **prophet_kwargs):
        """Treina Prophet e ARMAZENA o modelo em self.prophet_model_."""
        self.prophet_config_ = prophet_kwargs

        if verbose:
            print("Treinando modelo Prophet...")
        model = Prophet(**prophet_kwargs)
        model.fit(train_df[['ds', 'y']])
        self.prophet_model_ = model # <-- ARMAZENA O ESTADO
        
        if verbose:
            print("✅ Modelo Prophet treinado e salvo em 'self.prophet_model_'.")
        return model

    def optimize_prophet_hyperparams(self, df, n_trials: int = 25, horizon_months: int = 3, return_model: bool = True, verbose: bool = False):
        """
        Otimiza hiperparâmetros do Prophet e ARMAZENA o melhor modelo.
        """
        df = df.rename(columns={self.date_col: "ds", self.target_col: "y"})[["ds", "y"]].dropna().sort_values("ds")

        if len(df) < 24:
            raise ValueError("Dataset muito pequeno para validação cruzada (mínimo: 24 pontos).")

        horizon_str = f"{horizon_months * 30} days"

        def objective(trial):
            params = {
                "changepoint_prior_scale": trial.suggest_float("changepoint_prior_scale", 0.001, 0.5, log=True),
                "seasonality_prior_scale": trial.suggest_float("seasonality_prior_scale", 0.01, 10.0, log=True),
                "holidays_prior_scale": trial.suggest_float("holidays_prior_scale", 0.01, 10.0, log=True),
                "seasonality_mode": trial.suggest_categorical("seasonality_mode", ["additive", "multiplicative"]),
                "changepoint_range": trial.suggest_float("changepoint_range", 0.7, 0.95),
            }

            try:
                model = Prophet(**params)
                model.fit(df)
                df_cv = cross_validation(model, horizon=horizon_str, parallel="processes")
                rmse_mean = performance_metrics(df_cv, rolling_window=1)["rmse"].mean()
            except Exception as e:
                print(f"⚠️ Falha no trial {trial.number}: {e}")
                rmse_mean = np.inf
            return rmse_mean

        study = optuna.create_study(direction="minimize")
        with tqdm(total=n_trials, desc="Optuna Prophet Optimization") as pbar:
            def cb(study, trial): pbar.update(1)
            study.optimize(objective, n_trials=n_trials, callbacks=[cb])

        best_params = study.best_params
        print(f"\n🏆 Melhor RMSE: {study.best_value:.4f}\nMelhores parâmetros: {best_params}")

        if return_model:
            best_model = Prophet(**best_params).fit(df)
            self.prophet_model_ = best_model # <-- ARMAZENA O ESTADO
            
            if verbose:
                print("✅ Melhor modelo Prophet treinado e salvo em 'self.prophet_model_'.")
            return best_params, best_model
        return best_params

    #######################################################
    ### 3️. Pré-processamento e criação de sequências
    #######################################################
    
    def preprocess_features(self, df, categorical_cols=None, fit_encoders=True, encoders=None, verbose: bool = False):
        """
        Aplica LabelEncoder (se existirem variáveis categóricas) e adiciona features sazonais.
        Se fit_encoders=True, armazena os encoders em self.encoders_.
        Retorna (df, encoders_to_use).
        """
        df = df.copy()
        encoders_to_use = None

        # Se não houver variáveis categóricas, apenas cria features sazonais
        if not categorical_cols:
            df['ds'] = pd.to_datetime(df['ds'])

            # Features sazonais numéricas
            if 'ds' in df.columns and pd.api.types.is_datetime64_any_dtype(df['ds']):
                df['month'] = df['ds'].dt.month
                df['month_sin'] = np.sin(2 * np.pi * (df['month'] - 1) / 12)
                df['month_cos'] = np.cos(2 * np.pi * (df['month'] - 1) / 12)

            # Nenhum encoder necessário
            return df, {}
        else:
            # Caso existam colunas categóricas, aplica o mesmo comportamento original
            if fit_encoders:
                self.encoders_ = {}
                
                if verbose:
                    print("Ajustando LabelEncoders...")
                for col in categorical_cols:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    self.encoders_[col] = le
                
                if verbose:
                    print(f"✅ {len(self.encoders_)} encoders ajustados e salvos em 'self.encoders_'.")
                encoders_to_use = self.encoders_
            else:
                if encoders is None:
                    if getattr(self, 'encoders_', None) is None:
                        raise ValueError("Encoders não ajustados. Chame com fit_encoders=True ou passe 'encoders'.")
                    encoders_to_use = self.encoders_
                else:
                    encoders_to_use = encoders

                for col in categorical_cols:
                    if col not in encoders_to_use:
                        raise ValueError(f"Encoder para coluna '{col}' não encontrado.")
                    le = encoders_to_use[col]
                    df[col] = le.transform(df[col].astype(str))

        # Adiciona features sazonais
        df['ds'] = pd.to_datetime(df['ds'])
        if 'ds' in df.columns and pd.api.types.is_datetime64_any_dtype(df['ds']):
            df['month'] = df['ds'].dt.month
            df['month_sin'] = np.sin(2 * np.pi * (df['month'] - 1) / 12)
            df['month_cos'] = np.cos(2 * np.pi * (df['month'] - 1) / 12)

        return df, encoders_to_use

    def fit_scaler(self, df, numerical_cols, scaler_class=MinMaxScaler, verbose: bool = False, **scaler_kwargs):
        """
        Ajusta e ARMAZENA o scaler (ex: MinMaxScaler) em self.scaler_
        """
        
        if verbose:
            print("Ajustando Scaler...")
        self.scaler_ = scaler_class(**scaler_kwargs)
        df[numerical_cols] = self.scaler_.fit_transform(df[numerical_cols])
        
        if verbose:
            print(f"✅ Scaler ({scaler_class.__name__}) ajustado e salvo em 'self.scaler_'.")
        return df, self.scaler_

    def create_sequences(self, X_num, X_cat, y, window_size):
        X_num_seq, X_cat_seq, y_seq = [], [], []

        n = len(X_num)
        if n <= window_size:
            if verbose:
                print(f"⚠️ [create_sequences] Dados insuficientes: {n=} <= {window_size=}. Pulando criação de janelas.")
            # Retorna tensores vazios (ou uma janela fake)
            return (
                torch.zeros((0, window_size, X_num.shape[-1]), dtype=torch.float32),
                torch.zeros((0, window_size, X_cat.shape[-1] if X_cat.numel() > 0 else 0), dtype=torch.long),
                torch.zeros((0, 1), dtype=torch.float32)
            )

        for i in range(n - window_size):
            X_num_seq.append(X_num[i:i + window_size])
            X_cat_seq.append(X_cat[i:i + window_size])
            y_seq.append(y[i + window_size])

        X_num_seq = torch.stack(X_num_seq)
        X_cat_seq = torch.stack(X_cat_seq)
        y_seq = torch.stack(y_seq)

        return X_num_seq, X_cat_seq, y_seq
    
    #######################################################
    ### 4️. Treinamento Walk-Forward (Prophet + LSTM)
    #######################################################
    
    def walk_forward_train(
        self,
        df,
        numerical_cols,
        categorical_cols=None,
        cat_cardinalities=None,
        embed_dims=None,
        window_size=24,
        horizon=3,
        lstm_epochs=10,
        incremental=True,
        use_existing_yhat=True,  # ⚙️ novo parâmetro opcional
        verbose: bool = False
    ):
        """
        Treina o modelo LSTM residual.
        Se use_existing_yhat=True, usa as previsões Prophet já existentes (colunas 'yhat' e 'residual')
        e NÃO recalcula o Prophet internamente.
        """

        # --- Validações iniciais ---
        if self.prophet_model_ is None:
            raise ValueError("⚠️ O modelo Prophet não foi treinado. Chame train_prophet() primeiro.")

        categorical_cols = categorical_cols or []
        has_categorical = len(categorical_cols) > 0

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if verbose:
            print(f"🧠 Treinando LSTM no device: {device}")
            print(f"   🏷️ Variáveis categóricas: {categorical_cols if has_categorical else 'nenhuma'}")

        # --- Inicializa modelo base ---
        model = LSTMResidualModel(
            num_features=len(numerical_cols),
            cat_cardinalities=cat_cardinalities if has_categorical else [],
            embed_dims=embed_dims if has_categorical else [],
            hidden_size=64,
            num_layers=2
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        best_model_global = None
        best_rmse_global = float("inf")
        results = []

        # --- Loop Walk-Forward ---
        pbar = tqdm(range(window_size + 1, len(df) - horizon + 1), desc="Walk-Forward Training", leave=True)

        for end_idx in pbar:
            train_df = df.iloc[:end_idx].copy()
            test_df = df.iloc[end_idx:end_idx + horizon].copy()

            for col in numerical_cols + ['y', 'ds']:
                if col not in df.columns:
                    raise ValueError(f"Coluna esperada '{col}' ausente em df.")

            # --- Usa previsões Prophet existentes (sem recalcular) ---
            if use_existing_yhat:
                if not {'yhat', 'residual'}.issubset(df.columns):
                    raise ValueError("Colunas 'yhat' e 'residual' não encontradas no DataFrame.")
                merged = test_df.copy()
            else:
                # Recalcula Prophet se necessário (modo compatibilidade)
                prophet_model_local = Prophet(**self.prophet_config_)
                prophet_model_local.fit(train_df[['ds', 'y']])
                future = pd.DataFrame({'ds': test_df['ds'].values})
                forecast = prophet_model_local.predict(future)
                merged = test_df.merge(forecast[['ds', 'yhat']], on='ds', how='left')
                merged['residual'] = merged['y'] - merged['yhat']
                merged['residual'] = merged['residual'].fillna(0)

            # --- Encoding e tensores ---
            def encode_features(df_):
                X_num = torch.tensor(df_[numerical_cols].values, dtype=torch.float32).to(device)
                if has_categorical:
                    X_cat = torch.tensor(
                        np.stack([df_[col].values for col in categorical_cols], axis=1),
                        dtype=torch.long
                    ).to(device)
                else:
                    X_cat = torch.zeros((len(df_), 0), dtype=torch.long).to(device)
                y = torch.tensor(df_['residual'].values, dtype=torch.float32).view(-1, 1).to(device)
                return X_num, X_cat, y

            X_num_train, X_cat_train, y_train = encode_features(train_df)
            X_num_test, X_cat_test, y_test = encode_features(test_df)

            # --- Criação de janelas ---
            if hasattr(self, "create_sequences"):
                X_num_train, X_cat_train, y_train = self.create_sequences(X_num_train, X_cat_train, y_train, window_size)
                X_num_test, X_cat_test, y_test = self.create_sequences(X_num_test, X_cat_test, y_test, window_size=1)
            else:
                print("⚠️ Método create_sequences() não encontrado, prosseguindo sem janelas.")

            if not incremental:
                model = LSTMResidualModel(
                    num_features=len(numerical_cols),
                    cat_cardinalities=cat_cardinalities if has_categorical else [],
                    embed_dims=embed_dims if has_categorical else [],
                    hidden_size=64,
                    num_layers=2
                ).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            # --- Treinamento por época ---
            epoch_hist = {"split_date": [], "epoch": [], "train_loss": [], "val_loss": [], "rmse": [], "mae": []}
            best_rmse_split = float("inf")
            best_model_split = None

            for epoch in tqdm(range(1, lstm_epochs + 1), desc=f'Split {end_idx}', leave=False):
                model.train()
                optimizer.zero_grad()
                out = model(X_num_train, X_cat_train)
                loss = criterion(out, y_train)
                loss.backward()
                optimizer.step()

                model.eval()
                with torch.no_grad():
                    val_out = model(X_num_test, X_cat_test)
                    val_loss = criterion(val_out, y_test)
                    pred_res = val_out.cpu().numpy().flatten()

                valid_len = len(pred_res)
                y_true = test_df['y'].values[-valid_len:]
                yhat = test_df['yhat'].values[-valid_len:] if 'yhat' in test_df.columns else np.zeros(valid_len)
                hybrid = yhat + pred_res

                mask = ~np.isnan(y_true)
                if mask.any():
                    rmse = np.sqrt(mean_squared_error(y_true[mask], hybrid[mask]))
                    mae = mean_absolute_error(y_true[mask], hybrid[mask])
                else:
                    rmse, mae = np.nan, np.nan

                epoch_hist["split_date"].append(train_df['ds'].iloc[-1])
                epoch_hist["epoch"].append(epoch)
                epoch_hist["train_loss"].append(loss.item())
                epoch_hist["val_loss"].append(val_loss.item())
                epoch_hist["rmse"].append(rmse)
                epoch_hist["mae"].append(mae)

                if not np.isnan(rmse) and rmse < best_rmse_split:
                    best_rmse_split = rmse
                    best_model_split = copy.deepcopy(model)

            if best_model_split is not None and best_rmse_split < best_rmse_global:
                best_rmse_global = best_rmse_split
                best_model_global = copy.deepcopy(best_model_split)

            results.append({
                'train_end': train_df['ds'].iloc[-1],
                'test_start': test_df['ds'].iloc[0],
                'test_end': test_df['ds'].iloc[-1],
                'best_rmse': best_rmse_split,
                'best_mae': mae
            })

        results_df = pd.DataFrame(results)
        self.lstm_model_ = best_model_global
        self.pipeline_mode_ = "hybrid"
        
        if verbose:
            print(f"🏆 Melhor RMSE Global: {best_rmse_global:.4f}")
        return results_df, best_model_global


    #######################################################
    ### 5. INFERÊNCIAS
    #######################################################

    def infer_prophet(self, df, horizon: int = 3, verbose: bool = False):
        """
        Realiza inferência SOMENTE com o modelo Prophet salvo em 'self.prophet_model_'.
        Mantém todas as colunas originais do DataFrame e adiciona as previsões ('yhat')
        e o resíduo ('residual'), sem alterar a ordem dos dados.
        """
        if self.prophet_model_ is None:
            raise ValueError("Modelo Prophet (self.prophet_model_) não foi treinado ou carregado.")

        if verbose:
            print(f"Executando inferência Prophet (horizonte={horizon})...")

        # Cria dataframe futuro
        future = self.prophet_model_.make_future_dataframe(periods=horizon, freq='MS', include_history=True)

        # Gera previsões
        forecast = self.prophet_model_.predict(future)

        # Mantém todas as colunas originais do df e adiciona previsões e resíduos
        forecast_all = (
            df.merge(forecast[['ds', 'yhat']], on='ds', how='right')  # garante inclusão do horizonte futuro
            .assign(residual=lambda x: x['y'] - x['yhat'])
        )

        if verbose:
            print("✅ Previsão Prophet concluída.")
        return forecast_all

    def infer_hybrid(self, df, forecast_all, categorical_cols, feature_cols, window=6, verbose: bool = False):
        """
        Realiza inferência HÍBRIDA (Prophet + LSTM) em TODO o dataset.
        Usa rolling window para gerar previsões ao longo de toda a série temporal.
        Valores negativos de previsão são ajustados para zero (não faz sentido ter estoque/vendas negativas).
        """
        if verbose:
            print("🚀 Executando inferência Híbrida (Prophet + LSTM) em toda a série...")

        # --- Validação de componentes ---
        if self.prophet_model_ is None or self.lstm_model_ is None:
            raise ValueError("❌ Pipeline híbrido incompleto. Treine ou carregue Prophet e LSTM antes.")
        if self.scaler_ is None:
            raise ValueError("❌ Scaler não encontrado. Execute fit_scaler() antes da inferência.")
        if 'yhat' not in forecast_all.columns:
            raise ValueError("❌ A coluna 'yhat' (previsão Prophet) é obrigatória no forecast_all.")

        # --- Preparação dos dados ---
        cols_to_keep = ['ds', 'yhat'] + categorical_cols
        full = forecast_all[cols_to_keep].merge(df[['ds', 'y']], on='ds', how='left')

        # Calcula resíduos (útil para escala e continuidade)
        full['residual'] = full['y'] - full['yhat']
        full = full.fillna(0)

        # Aplica pré-processamento categórico e escala
        full, _ = self.preprocess_features(full, categorical_cols, fit_encoders=False)
        full[feature_cols] = self.scaler_.transform(full[feature_cols])

        self.lstm_model_.eval()

        preds_hybrid, preds_prophet, dates_all = [], [], []

        # --- Rolling window sobre TODO o dataset ---
        rolling_window_num = deque(maxlen=window)
        rolling_window_cat = deque(maxlen=window)

        # Inicializa janelas com os primeiros pontos
        for i in range(window):
            rolling_window_num.append(full[feature_cols].iloc[i].values)
            rolling_window_cat.append(full[categorical_cols].iloc[i].values)

        for i in range(window, len(full)):
            # Prepara tensores
            X_num_input = np.array(rolling_window_num).reshape(1, window, len(feature_cols))
            X_cat_input = np.array(rolling_window_cat).reshape(1, window, len(categorical_cols)) \
                if categorical_cols else np.zeros((1, window, 0))

            X_num_tensor = torch.tensor(X_num_input, dtype=torch.float32)
            X_cat_tensor = torch.tensor(X_cat_input, dtype=torch.long)

            # Inferência do LSTM
            with torch.no_grad():
                pred_res = self.lstm_model_(X_num_tensor, X_cat_tensor).cpu().numpy().reshape(-1)[0]

            yhat_prophet = full['yhat'].iloc[i]
            hybrid_forecast = yhat_prophet + pred_res

            preds_hybrid.append(hybrid_forecast)
            preds_prophet.append(yhat_prophet)
            dates_all.append(full['ds'].iloc[i])

            # Atualiza janelas com novos valores
            next_features_num = full[feature_cols].iloc[i].values.copy()
            next_features_cat = full[categorical_cols].iloc[i].values if categorical_cols else []
            next_features_num[feature_cols.index('residual')] = pred_res

            rolling_window_num.append(next_features_num)
            if categorical_cols:
                rolling_window_cat.append(next_features_cat)

        # --- Cria DataFrame final com todas as previsões ---
        df_pred = pd.DataFrame({
            'ds': pd.to_datetime(dates_all),
            'yhat_prophet': preds_prophet,
            'yhat': preds_hybrid,  # previsão híbrida final
            'residual_pred': np.array(preds_hybrid) - np.array(preds_prophet)
        })

        # --- Adiciona valores reais (y) quando disponíveis ---
        if 'y' in df.columns:
            df_real = df[['ds', 'y']].drop_duplicates(subset='ds', keep='last')
            df_pred = df_pred.merge(df_real, on='ds', how='left')  # left join preserva previsões futuras

            # Opcional: preencher futuros com 0 em vez de NaN
            # df_pred['y'] = df_pred['y'].fillna(0)

        else:
            df_pred['y'] = np.nan

        # --- Pós-processamento: força valores negativos para 0 ---
        num_negativos = (df_pred['yhat'] < 0).sum()
        if num_negativos > 0:
            if verbose:
                print(f"⚠️ Ajustando {num_negativos} previsões negativas para zero (não faz sentido físico).")
            df_pred.loc[df_pred['yhat'] < 0, 'yhat'] = 0.0

        if verbose:
            print(f"✅ Inferência concluída: {len(df_pred)} pontos previstos (janela {window})")
        return df_pred

    #######################################################
    ### 6. Salvamento e Carregamento de Estado
    #######################################################

    def save_models(self, verbose: bool = False):
        """
        Salva o ESTADO ATUAL do pipeline (modelos, encoders, scaler) no model_dir.
        Não recebe mais argumentos.
        """
        os.makedirs(self.model_dir, exist_ok=True)
        if verbose:
            print(f"💾 Salvando artefatos em {self.model_dir}...")

        # 1️. Prophet
        if self.prophet_model_ is not None:
            prophet_path = os.path.join(self.model_dir, "prophet_model.json")
            with open(prophet_path, "w") as fout:
                fout.write(model_to_json(self.prophet_model_))
            if verbose:
                print(f"✅ Modelo Prophet salvo.")
        
        # 2️. LSTM
        if self.lstm_model_ is not None:
            lstm_path = os.path.join(self.model_dir, "lstm_residual.pth")
            torch.save(self.lstm_model_.state_dict(), lstm_path)
            if verbose:
                print(f"✅ Modelo LSTM salvo.")
            self.pipeline_mode_ = "hybrid"
        else:
            self.pipeline_mode_ = "prophet"
            if verbose:
                print("ℹ️ Nenhum modelo LSTM (modo Prophet puro).")
        
        # 3️. Encoders
        if self.encoders_ is not None:
            for col, le in self.encoders_.items():
                enc_path = os.path.join(self.model_dir, f"encoder_{col}.pkl")
                joblib.dump(le, enc_path)
            if verbose:
                print(f"✅ {len(self.encoders_)} encoders salvos.")
            
        # 4️. Scaler
        if self.scaler_ is not None:
            scaler_path = os.path.join(self.model_dir, "scaler.pkl")
            joblib.dump(self.scaler_, scaler_path)
            if verbose:
                print(f"✅ Scaler salvo.")
            
        if verbose:
            print(f"💾 Salvamento concluído (modo: {self.pipeline_mode_.upper()}).")

    def load_models(self, model_class=LSTMResidualModel, categorical_cols=None, numerical_cols=None, verbose: bool = False):
        """
        Carrega o ESTADO do pipeline (modelos, encoders, scaler) para 'self'.
        Não retorna mais um dicionário.
        """
        if verbose:
            print(f"🔄 Carregando artefatos de {self.model_dir}...")
        
        # Caminhos esperados
        prophet_path = os.path.join(self.model_dir, "prophet_model.json")
        lstm_path = os.path.join(self.model_dir, "lstm_residual.pth")
        scaler_path = os.path.join(self.model_dir, "scaler.pkl")

        # 1️. Prophet
        if os.path.exists(prophet_path):
            with open(prophet_path, "r") as fin:
                self.prophet_model_ = model_from_json(fin.read())
            if verbose:
                print("✅ Modelo Prophet carregado.")
        else:
            if verbose:
                print("❌ Nenhum modelo Prophet encontrado.")

        # 2️. Encoders e Scaler
        encoders = {}
        if categorical_cols:
            for col in categorical_cols:
                enc_path = os.path.join(self.model_dir, f"encoder_{col}.pkl")
                if os.path.exists(enc_path):
                    encoders[col] = joblib.load(enc_path)
            if encoders:
                self.encoders_ = encoders
                if verbose:
                    print(f"✅ {len(self.encoders_)} LabelEncoders carregados.")

        if os.path.exists(scaler_path):
            self.scaler_ = joblib.load(scaler_path)
            if verbose:
                print("✅ Scaler carregado.")
        else:
            if verbose:
                print("⚠️ Nenhum scaler encontrado (necessário para modo híbrido).")

        # 3️. LSTM (opcional)
        if os.path.exists(lstm_path) and model_class is not None and numerical_cols:
            try:
                if categorical_cols and getattr(self, "encoders_", None):
                    # Reconstrói cardinalidades normalmente
                    cat_cardinalities = [len(self.encoders_[col].classes_) for col in categorical_cols]
                    embed_dims = [min(50, int(card ** 0.25) * 4) for card in cat_cardinalities]
                else:
                    # Nenhuma variável categórica
                    cat_cardinalities, embed_dims = [], []

                lstm_model = model_class(
                    num_features=len(numerical_cols),
                    cat_cardinalities=cat_cardinalities,
                    embed_dims=embed_dims,
                    hidden_size=64,
                    num_layers=2
                )
                lstm_model.load_state_dict(torch.load(lstm_path, map_location=torch.device('cpu')))
                lstm_model.eval()

                self.lstm_model_ = lstm_model
                self.pipeline_mode_ = "hybrid"
                if verbose:
                    print("✅ Modelo LSTM carregado (modo híbrido ativado).")

            except Exception as e:
                if verbose:
                    print(f"⚠️ Erro ao carregar LSTM: {e}")
        else:
            self.pipeline_mode_ = "prophet"
            if verbose:
                print("ℹ️ Nenhum modelo LSTM encontrado — modo Prophet puro ativado.")

        if verbose:
            print(f"\nModo carregado: {self.pipeline_mode_.upper()}")

############################################################################################################
### 7. EXEMPLO DE USO
############################################################################################################

if __name__ == '__main__':
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