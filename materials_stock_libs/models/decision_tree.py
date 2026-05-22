"""
Módulo de modelos de regressão baseados em árvores de decisão e regressão linear:
- Linear Regression
- Random Forest
- LightGBM
"""

############################################################################################################
### 1. IMPORTS
############################################################################################################

import pandas as pd
import numpy as np
import optuna
import lightgbm as lgb

from typing import Generator, Tuple
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from ..utils import make_synthetic_data

############################################################################################################
### 2. FUNÇÕES AUXILIARES
############################################################################################################

def normalize_horizon(horizon: str) -> str:
    """Normaliza o formato do horizonte (ex: '90 days' → '90D')."""
    import re
    match = re.match(r'(\d+)\s*days?', horizon)
    if match:
        return f"{match.group(1)}D"
    return horizon

############################################################################################################
### 3. CLASSE PRINCIPAL
############################################################################################################

class DecisionTreeModels:
    """
    Classe unificada para modelagem de séries temporais e regressões baseadas em:
      - Linear Regression
      - Random Forest
      - LightGBM

    Permite:
      - Split temporal (fixo ou walk-forward)
      - Treinamento e predição
      - Otimização de hiperparâmetros via Optuna
    """

    def __init__(self, df: pd.DataFrame, date_col: str = 'DT_EMISSAO',
                 target_col: str = 'UNIDADES', filter_col: str = None,
                 segment_value: str = None, seed: int = 42):
        self.df = df.copy()
        self.date_col = date_col
        self.target_col = target_col
        self.filter_col = filter_col
        self.segment_value = segment_value
        self.seed = seed
        self.models = {}
        self.encoders = {}

    # ==========================================================================================
    # 1️. SPLIT TEMPORAL
    # ==========================================================================================

    @staticmethod
    def time_train_test_split(
        df: pd.DataFrame,
        col_str: str,
        test_size: float = 0.2,
        n_splits: int = None,
        step_size: int = None,
        walk_forward: bool = False
    ):
        """
        Divide dados temporais, suportando holdout simples ou walk-forward.
        test_size é fração (ex: 0.2 = 20% finais para teste).
        """
        df_sorted = df.sort_values(col_str).reset_index(drop=True)
        n = len(df_sorted)

        # Convert test_size proporcional para número de linhas
        if 0 < test_size < 1:
            test_n = max(1, int(round(n * test_size)))
        else:
            raise ValueError("test_size deve ser um float entre 0 e 1 (ex: 0.2 para 20%)")

        if n <= test_n:
            raise ValueError(f"Dataset muito pequeno ({n} linhas) para test_size={test_n}")

        # HOLDOUT SIMPLES
        if not walk_forward:
            split_idx = n - test_n
            train = df_sorted.iloc[:split_idx].reset_index(drop=True)
            test = df_sorted.iloc[split_idx:].reset_index(drop=True)

            print(f"📊 Holdout aplicado: train={len(train)}, test={len(test)}")
            print(f"   Período treino: {train[col_str].min()} → {train[col_str].max()}")
            print(f"   Período teste:  {test[col_str].min()} → {test[col_str].max()}")
            return train, test

        # WALK-FORWARD
        if n_splits is None and step_size is None:
            raise ValueError("Para walk_forward=True, defina 'n_splits' ou 'step_size'.")

        if n_splits:
            step_size = max(1, (n - test_n) // n_splits)

        splits = []
        for start in range(0, n - test_n, step_size):
            end_train = start + (n - start - test_n)

            if end_train <= start:
                continue
            if end_train + test_n > n:
                break

            train = df_sorted.iloc[:end_train].reset_index(drop=True)
            test = df_sorted.iloc[end_train:end_train + test_n].reset_index(drop=True)

            splits.append((train, test))

        if not splits:
            raise ValueError(
                f"Nenhum split possível com n={n}, test_size={test_size}, "
                f"n_splits={n_splits}, step_size={step_size}"
            )

        print(f"📈 Walk-forward: {len(splits)} splits gerados")
        return splits

    # ==========================================================================================
    # 2️. PREPARAÇÃO DE DADOS
    # ==========================================================================================

    def _validate_str(self, value):
        if not isinstance(value, str):
            raise ValueError("Parâmetro esperado como string.")
        return value.strip()

    def filter_segment(self, filter_col: str = None, segment_value: str = None) -> pd.DataFrame:
        """
        Filtra o DataFrame para um segmento específico (ex: 'UF' == 'CE' ou 'material' == 'Aço').
        """
        filter_col = filter_col or self.filter_col
        segment_value = segment_value or self.segment_value

        if not filter_col or not segment_value:
            raise ValueError("Parâmetros 'filter_col' e 'segment_value' devem ser informados.")

        filter_col = self._validate_str(filter_col)
        segment_value = self._validate_str(segment_value)

        df_filtered = self.df[self.df[filter_col] == segment_value].copy()
        if df_filtered.empty:
            raise ValueError(f"Nenhum dado encontrado para {filter_col} = '{segment_value}'.")
        return df_filtered

    # ==========================================================================================
    # 3️. TREINAMENTO E PREVISÃO
    # ==========================================================================================
    def preprocess_features(self, df, categorical_cols=None, numeric_cols=None, fit_encoders=True):
        """
        Pré-processa features categóricas e numéricas.
        - fit_encoders=True: treina e salva encoders na instância.
        - fit_encoders=False: aplica encoders já treinados.
        - Preenche valores nulos nas numéricas com média histórica.

        Retorna:
            X, y, feature_cols, encoders_atualizados
        """
        df = df.copy()

        if not hasattr(self, "encoders"):
            self.encoders = {}
        if not hasattr(self, "train_stats"):
            self.train_stats = {}

        # --- Categóricas ---
        for col in (categorical_cols or []):
            if fit_encoders:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.encoders[col] = le
            else:
                le = self.encoders.get(col)
                if le is None:
                    raise ValueError(f"Encoder para {col} não foi treinado ainda.")
                # Substitui valores desconhecidos pela primeira classe
                known_classes = set(le.classes_)
                df[col] = df[col].apply(lambda x: le.transform([x])[0] if str(x) in known_classes else 0)

        # --- Numéricas ---
        for col in (numeric_cols or []):
            if col not in df.columns:
                df[col] = 0
            # Salva média histórica no treino
            if fit_encoders:
                self.train_stats[col] = {'mean': df[col].mean()}
            df[col] = df[col].fillna(self.train_stats[col]['mean'])

        # --- Timestamp ---
        if self.date_col in df.columns:
            df['timestamp'] = pd.to_datetime(df[self.date_col]).map(pd.Timestamp.toordinal)

        feature_cols = ['timestamp'] + (categorical_cols or []) + (numeric_cols or [])
        X = df[feature_cols]
        y = df[self.target_col] if self.target_col in df.columns else None

        return X, y, feature_cols, self.encoders

    # ==========================================================
    # Método de treino aprimorado
    # ==========================================================
    def train_model(
        self,
        df_segment: pd.DataFrame,
        date_col: str,
        model_type: str = 'rf',
        categorical_cols: list = None,
        numeric_cols: list = None,
        test_size: int = 90,
        n_splits: int = None,
        step_size: int = None,
        walk_forward: bool = False,
        **kwargs
    ):
        """
        Treina um modelo e salva os encoders no objeto.
        """
        self.date_col = date_col
        if not hasattr(self, "target_col"):
            self.target_col = [c for c in df_segment.columns if c not in (categorical_cols or []) + (numeric_cols or []) + [date_col]][0]

        
        split_result = self.time_train_test_split(
            df_segment,
            col_str=date_col,
            test_size=test_size,
            n_splits=n_splits,
            step_size=step_size,
            walk_forward=walk_forward
        )

        if walk_forward:
            splits = split_result  # lista de vários splits
        else:
            splits = [split_result]  # apenas um split final

        best_model = None
        best_rmse = float('inf')
        results = []
        self.pred_history = []

        for i, (train_df, val_df) in enumerate(splits):
            # Pré-processamento
            if i == 0:
                X_train, y_train, feature_cols, _ = self.preprocess_features(
                    train_df, categorical_cols=categorical_cols, numeric_cols=numeric_cols, fit_encoders=True
                )
            else:
                X_train, y_train, feature_cols, _ = self.preprocess_features(
                    train_df, categorical_cols=categorical_cols, numeric_cols=numeric_cols, fit_encoders=False
                )

            X_val, y_val, _, _ = self.preprocess_features(
                val_df, categorical_cols=categorical_cols, numeric_cols=numeric_cols, fit_encoders=False
            )

            # Instancia modelo
            if model_type == 'linear':
                model = LinearRegression(**kwargs)
            elif model_type == 'rf':
                model = RandomForestRegressor(random_state=self.seed, **kwargs)
            elif model_type == 'lgbm':
                model = lgb.LGBMRegressor(random_state=self.seed, **kwargs)
            else:
                raise ValueError("Escolha 'linear', 'rf' ou 'lgbm'.")

            # Treina
            model.fit(X_train, y_train)

            preds = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            results.append(rmse)
            print(f"📈 Split {i+1}: RMSE = {rmse:.4f}")

            # Salva previsões com datas
            self.pred_history.append(pd.DataFrame({
                "ds": val_df[self.date_col].values,
                "y_true": y_val.values,
                "y_pred": preds
            }))

            # # Avalia
            # preds = model.predict(X_val)
            # rmse = np.sqrt(mean_squared_error(y_val, preds))
            # results.append(rmse)
            # print(f"📈 Split {i+1}: RMSE = {rmse:.4f}")

            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model

            if not walk_forward:
                break

        if walk_forward:
            self.pred_history = pd.concat(self.pred_history, ignore_index=True)
        else:
            self.pred_history = pd.DataFrame()

        # Salva modelo e encoders
        self.models[model_type] = {
            "model": best_model,
            "features": feature_cols,
            "categorical_cols": categorical_cols,
            "numeric_cols": numeric_cols,
            "rmse": best_rmse,
            "encoders": self.encoders  # agora contém todos os encoders treinados
        }

        print("\n✅ Treinamento finalizado!")
        print(f"🏆 Melhor modelo: {model_type.upper()} | RMSE: {best_rmse:.4f}")
        return {
            "model": best_model,
            "rmse_splits": results,
            "best_rmse": best_rmse,
            "features": feature_cols,
            "encoders": self.encoders
        }

    def predict_future(self, model_key: str, future_dates: pd.DatetimeIndex, categorical_defaults: dict = None):
        """
        Gera previsões futuras usando encoders do treino e médias históricas para numéricas.
        """

        if model_key not in self.models:
            raise ValueError(f"Modelo '{model_key}' não encontrado.")

        model_info = self.models[model_key]
        model = model_info["model"]
        categorical_cols = model_info.get("categorical_cols", [])
        numeric_cols = model_info.get("numeric_cols", [])
        feature_cols = model_info.get("features", [])
        encoders = model_info.get("encoders", {})

        # --- Cria DataFrame futuro ---
        future_df = pd.DataFrame({self.date_col: future_dates})
        future_df["timestamp"] = pd.to_datetime(future_df[self.date_col]).map(pd.Timestamp.toordinal)

        # --- Preenche categóricas ---
        for col in categorical_cols:
            if categorical_defaults and col in categorical_defaults:
                val = categorical_defaults[col]
                # Substitui se não existir no encoder
                le = encoders.get(col)
                if le and val not in le.classes_:
                    print(f"⚠️ Valor '{val}' para '{col}' não existia no treino. Usando primeira classe.")
                    val = le.classes_[0]
            else:
                le = encoders.get(col)
                if le is None:
                    raise ValueError(f"Encoder para '{col}' não encontrado.")
                val = le.classes_[0]  # fallback seguro
            future_df[col] = val

        # --- Preenche numéricas com médias históricas ---
        for col in numeric_cols:
            if col not in future_df.columns or future_df[col].isnull().all():
                mean_val = self.train_stats.get(col, {}).get('mean', 0)
                future_df[col] = mean_val

        # --- Pré-processamento ---
        X_future, _, _, _ = self.preprocess_features(
            future_df,
            categorical_cols=categorical_cols,
            numeric_cols=numeric_cols,
            fit_encoders=False
        )

        # --- Garante ordem das features ---
        X_future = X_future[feature_cols]

        # --- Predição ---
        preds = model.predict(X_future)
        return pd.DataFrame({"ds": future_dates, "yhat": preds})

    # ==========================================================================================
    # 4️. OTIMIZAÇÃO DE HIPERPARÂMETROS COM OPTUNA
    # ==========================================================================================

    def optimize_model(
            self,
            df_segment: pd.DataFrame,
            date_col: str,
            target_col: str,
            model_type: str = 'rf',
            categorical_cols: list = None,
            numeric_cols: list = None,
            n_trials: int = 25,
            n_splits: int = 5
        ):
        """
        Otimiza hiperparâmetros de modelos regressivos via Optuna
        (Linear, Random Forest, LightGBM), com suporte a features categóricas.
        Treina encoders dentro de cada fold para evitar erros de categorias desconhecidas.
        """

        df_segment = df_segment.copy()
        df_segment["timestamp"] = pd.to_datetime(df_segment[date_col]).map(pd.Timestamp.toordinal)

        X_full = df_segment[['timestamp'] + (categorical_cols or []) + (numeric_cols or [])].copy()
        y_full = df_segment[target_col].copy()

        tscv = TimeSeriesSplit(n_splits=n_splits)

        def objective(trial):
            # Escolhe o modelo e hiperparâmetros
            if model_type == 'linear':
                model = LinearRegression(
                    fit_intercept=trial.suggest_categorical('fit_intercept', [True, False])
                )

            elif model_type == 'rf':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 400),
                    'max_depth': trial.suggest_int('max_depth', 3, 30),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                    'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                    'random_state': self.seed,
                    'n_jobs': -1
                }
                model = RandomForestRegressor(**params)

            elif model_type == 'lgbm':
                params = {
                    'num_leaves': trial.suggest_int('num_leaves', 20, 80),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'n_estimators': trial.suggest_int('n_estimators', 100, 400),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
                    'random_state': self.seed
                }
                model = lgb.LGBMRegressor(**params)

            else:
                raise ValueError("Modelo inválido. Use 'linear', 'rf' ou 'lgbm'.")

            rmses = []

            for train_idx, val_idx in tscv.split(X_full):
                X_train_fold = X_full.iloc[train_idx].copy()
                X_val_fold = X_full.iloc[val_idx].copy()
                y_train_fold = y_full.iloc[train_idx].copy()
                y_val_fold = y_full.iloc[val_idx].copy()

                # --- Treina encoders apenas no fold de treino ---
                encoders = {}
                for col in (categorical_cols or []):
                    le = LabelEncoder()
                    X_train_fold[col] = le.fit_transform(X_train_fold[col].astype(str))
                    encoders[col] = le
                    # Aplica encoder no fold de validação
                    # Substitui valores desconhecidos pela primeira classe do treino
                    X_val_fold[col] = X_val_fold[col].map(
                        lambda x: le.transform([x])[0] if x in le.classes_ else 0
                    )

                # Converte para NumPy
                X_train_np = X_train_fold.values
                X_val_np = X_val_fold.values
                y_train_np = y_train_fold.values
                y_val_np = y_val_fold.values

                # Treina e avalia
                model.fit(X_train_np, y_train_np)
                preds = model.predict(X_val_np)
                rmses.append(mean_squared_error(y_val_np, preds, squared=False))

            return np.mean(rmses)

        # --- Estudo Optuna ---
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)

        print(f"✅ Melhor RMSE: {study.best_value:.4f}")
        print("🔧 Melhores Hiperparâmetros:")
        for k, v in study.best_params.items():
            print(f"  - {k}: {v}")

        return study.best_params

    def plot_dt_model(self, df, model_type, forecast_future=None):
        plt.figure(figsize=(12,5))

        # Histórico real
        plt.plot(df['DT_EMISSAO'], df['QUANTIDADE'], label='Histórico', color='black', alpha=0.7)

        # Previsões no período de teste
        if hasattr(self, "pred_history") and isinstance(self.pred_history, pd.DataFrame) and not self.pred_history.empty:
            plt.plot(
                self.pred_history['ds'], 
                self.pred_history['y_pred'], 
                'o-', label='Previsão (Validação)', alpha=0.8
            )
            # Conectar com linha tracejada
            plt.axvline(self.pred_history['ds'].min(), color='gray', linestyle='--', alpha=0.6)

        # Predição futura
        if forecast_future is not None:
            plt.plot(forecast_future['ds'], forecast_future['yhat'], 'o--',
                    label=f'Previsão futura ({model_type.upper()})', color='tab:blue')

        plt.title(f"Forecast com {model_type.upper()} — Histórico, Validação e Futuro")
        plt.xlabel("Data")
        plt.ylabel("Unidades")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

### DEPRECATED
# def train_lightgbm(df, target_col='status', test_size=0.2, random_state=42, params=None):
#     X = df.drop(columns=[target_col])
#     y = df[target_col]

#     # Identificar colunas categóricas do tipo object
#     categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

#     # Converter colunas categóricas para category
#     for col in categorical_cols:
#         X[col] = X[col].astype('category')

#     # Dividir treino e teste
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=test_size, random_state=random_state, stratify=y
#     )

#     # Criar datasets LightGBM com categorical_feature
#     dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_cols)
#     dvalid = lgb.Dataset(X_test, label=y_test, reference=dtrain, categorical_feature=categorical_cols)

#     if params is None:
#         params = {
#             'objective': 'binary',
#             'metric': 'binary_logloss',
#             'learning_rate': 0.1,
#             'shrinkage_rate': 0.12,
#             'num_leaves': 25,
#             'max_depth': 4,
#             'min_data_in_leaf': 20,
#             'verbosity': -1,
#             'random_state': random_state
#         }
#     else:
#         params = {
#             'objective': 'binary',
#             'metric': 'binary_logloss',
#             'verbosity': -1,
#             'random_state': random_state,
#             **params
#         }

#     model = lgb.train(
#         params,
#         dtrain,
#         num_boost_round=100,
#         valid_sets=[dvalid]
#     )

#     y_pred = model.predict(X_test)
#     y_pred_binary = (y_pred > 0.5).astype(int)

#     acc = accuracy_score(y_test, y_pred_binary)
#     report = classification_report(y_test, y_pred_binary)

#     print(f"Accuracy: {acc:.4f}")
#     print("Classification report:\n", report)

#     importances = model.feature_importance(importance_type='gain')
#     feature_names = model.feature_name()

#     feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:20]

#     print("Top 20 features (gain):")
#     for feat, imp in feat_imp:
#         print(f"{feat}: {imp}")

#     top_features = [f[0] for f in feat_imp]
#     top_importances = [f[1] for f in feat_imp]

#     plt.figure(figsize=(10,6))
#     plt.barh(top_features[::-1], top_importances[::-1], color='skyblue')
#     plt.xlabel('Importance (gain)')
#     plt.title('Top 20 Features by Gain')
#     plt.show()

#     ax = lgb.plot_tree(model, tree_index=0, figsize=(20, 10), show_info=['split_gain'])
#     plt.show()

#     return model

# def tune_lightgbm_optuna(df, target_col='status', test_size=0.2, random_state=42, n_trials=50):
#     X = df.drop(columns=[target_col])
#     y = df[target_col]

#     def objective(trial):
#         params = {
#             'objective': 'binary',
#             'metric': 'binary_logloss',
#             'verbosity': -1,
#             'boosting_type': 'gbdt',
#             'num_leaves': trial.suggest_int('num_leaves', 20, 50),
#             'max_depth': trial.suggest_int('max_depth', 3, 15),
#             'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
#             'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 50),
#             'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
#             'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
#             # 'shrinkage_rate': trial.suggest_float('shrinkage_rate', 0.01, 1.0),
#             # 'eta': trial.suggest_float('eta', 0.01, 1.0),
#             'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
#             'random_state': random_state
#         }

#         X_train, X_valid, y_train, y_valid = train_test_split(
#             X, y, test_size=test_size, random_state=random_state)

#         dtrain = lgb.Dataset(X_train, label=y_train)
#         dvalid = lgb.Dataset(X_valid, label=y_valid)

#         model = lgb.train(
#             params,
#             dtrain,
#             valid_sets=[dvalid]
#         )

#         preds = model.predict(X_valid)
#         preds_binary = (preds > 0.5).astype(int)
#         accuracy = accuracy_score(y_valid, preds_binary)

#         return accuracy

#     study = optuna.create_study(direction='maximize')
#     study.optimize(objective, n_trials=n_trials)

#     print("Best hyperparameters found:")
#     print(study.best_params)
#     print(f"Best Accuracy: {study.best_value:.4f}")

#     # Treinar o modelo final com os melhores hiperparâmetros no dataset inteiro
#     best_params = study.best_params
#     best_params.update({
#         'objective': 'binary',
#         'metric': 'binary_logloss',
#         'verbosity': -1,
#         'random_state': random_state
#     })

#     dtrain_full = lgb.Dataset(X, label=y)
#     final_model = lgb.train(best_params, dtrain_full, num_boost_round=100)

#     return final_model, study.best_params


# def train_random_forest(df, target_col='status', test_size=0.2, random_state=42, params=None):
#     X = df.drop(columns=[target_col])
#     y = df[target_col]

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=test_size, random_state=random_state
#     )

#     if params is None:
#         params = {
#             'n_estimators': 100,
#             'max_depth': None,
#             'random_state': random_state,
#             'n_jobs': -1
#         }
#     else:
#         params['random_state'] = random_state

#     model = RandomForestClassifier(**params)
#     model.fit(X_train, y_train)

#     y_pred = model.predict(X_test)
#     acc = accuracy_score(y_test, y_pred)
#     report = classification_report(y_test, y_pred)

#     print(f"Accuracy: {acc:.4f}")
#     print("Classification report:\n", report)

#     # Importância das features
#     importances = model.feature_importances_
#     feature_names = X.columns
#     feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:20]

#     print("Top 20 features (impurity-based):")
#     for feat, imp in feat_imp:
#         print(f"{feat}: {imp:.4f}")

#     # Plotar
#     top_features = [f[0] for f in feat_imp]
#     top_importances = [f[1] for f in feat_imp]

#     plt.figure(figsize=(10,6))
#     sns.barplot(x=top_importances[::-1], y=top_features[::-1], palette="viridis")
#     plt.xlabel('Importance')
#     plt.title('Top 20 Features - Random Forest')
#     plt.show()

#     return model

# def optimize_random_forest(df, target_col='status', n_trials=100, random_state=42):
#     X = df.drop(columns=[target_col])
#     y = df[target_col]

#     def objective(trial):
#         params = {
#             'n_estimators': trial.suggest_int('n_estimators', 100, 300),
#             'max_depth': trial.suggest_int('max_depth', 4, 30),
#             'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
#             'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
#             'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
#             'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
#             'random_state': random_state,
#             'n_jobs': -1
#         }

#         model = RandomForestClassifier(**params)
#         score = cross_val_score(model, X, y, cv=3, scoring='accuracy').mean()
#         return score

#     study = optuna.create_study(direction='maximize')
#     study.optimize(objective, n_trials=n_trials)

#     print("Best Trial:")
#     print(f"Accuracy: {study.best_value:.4f}")
#     print("Best Params:", study.best_params)

#     return study.best_params

if __name__ == '__main__':
    ############################################################################################################
    # 1️. GERAÇÃO DE DADOS TEMPORAIS SINTÉTICOS
    ############################################################################################################

    df = make_synthetic_data()

    print(df.head())

    ############################################################################################################
    # 2. INSTANCIAR E PREPARAR O MODELO
    ############################################################################################################

    # Instancia a classe principal
    trainer = DecisionTreeModels(
        df=df,
        date_col='DT_EMISSAO',
        target_col='UNIDADES',
        filter_col='UF',
        segment_value='CE',
        seed=42
    )

    ############################################################################################################
    # 3️. TREINAR UM MODELO
    ############################################################################################################

    # Escolha o tipo de modelo: 'linear', 'rf', ou 'lgbm'
    model_type = 'lgbm'   # altere para 'rf' ou 'linear' se quiser testar outros

    model = trainer.train_model(
        filter_col='UF',
        segment_value='CE',
        model_type=model_type,
        n_estimators=200,
        learning_rate=0.1
    )

    print(f"✅ Modelo '{model_type}' treinado com sucesso!")

    ############################################################################################################
    # 4️. FAZER PREVISÕES FUTURAS (3 MESES)
    ############################################################################################################

    future_dates = pd.date_range(df['DT_EMISSAO'].max() + pd.offsets.MonthBegin(1), periods=3, freq='MS')
    forecast = trainer.predict_future(segment_value='CE', future_dates=future_dates)

    print(forecast)

    ############################################################################################################
    # 5️. VISUALIZAÇÃO DOS RESULTADOS
    ############################################################################################################

    plt.figure(figsize=(12,5))
    plt.plot(df['DT_EMISSAO'], df['UNIDADES'], label='Histórico', color='black', alpha=0.7)
    plt.plot(forecast['ds'], forecast['pred'], 'o--', label=f'Previsão ({model_type.upper()})', color='tab:blue')
    plt.axvline(df['DT_EMISSAO'].iloc[-1], color='gray', linestyle='--', alpha=0.7)
    plt.title(f"Forecast com {model_type.upper()} — Exemplo de Séries Temporais")
    plt.xlabel("Data")
    plt.ylabel("Unidades")
    plt.legend()
    plt.show()

    ############################################################################################################
    # 6️. (OPCIONAL) — OTIMIZAÇÃO DE HIPERPARÂMETROS COM OPTUNA
    ############################################################################################################

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

    ############################################################################################################
    # 7. SALVANDO E CARREGANDO O MODELO
    ############################################################################################################

    # Salvar o modelo treinado
    import joblib
    joblib.dump(trainer.models['CE'], 'lightgbm_CE_model.pkl')

    # Carregar novamente
    loaded_model = joblib.load('lightgbm_CE_model.pkl')