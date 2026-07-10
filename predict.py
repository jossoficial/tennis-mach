import os
import sys
import asyncio
import aiohttp
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime

# ML & Hyperparameter Optimization
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

import lightgbm as lgb
import xgboost as xgb

# Bayesian Optimization (lightweight for mobile)
from skopt import gp_minimize, space
from skopt.utils import use_named_args
from sklearn.model_selection import cross_val_score

# ==================== LOGGING SETUP ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tennis_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==================== CONFIGURATION ====================
@dataclass
class Config:
    """Configuration dataclass for pipeline parameters."""
    api_key: Optional[str] = os.getenv("SCRAPER_API_KEY")
    capital_total: float = 1000.0
    kelly_fraction: float = 0.25
    timeout_seconds: int = 15
    max_retries: int = 3
    chunk_size: int = 5
    
    def validate(self) -> bool:
        """Validate configuration parameters."""
        try:
            assert self.api_key is not None, "SCRAPER_API_KEY not set"
            assert self.capital_total > 0, "Capital must be positive"
            assert 0 < self.kelly_fraction <= 1, "Kelly fraction must be between 0 and 1"
            assert self.timeout_seconds > 0, "Timeout must be positive"
            return True
        except AssertionError as e:
            logger.error(f"❌ Configuration Error: {e}")
            return False

config = Config()

# File paths
FILE_HISTORICO = "historico_entrenamiento.csv"
FILE_PREDICCIONES = "registro_predicciones.csv"

# Professional features columns
COLUMNAS_PROFESIONALES = [
    'j1_win_rate_recent', 'j1_surface_efficiency', 'j1_ace_percentage', 'j1_bp_saved_pct', 
    'j1_bp_converted_pct', 'j1_first_serve_won_pct', 'j1_fatiga_sets_7d', 'j1_elo_rating_norm',
    'j2_win_rate_recent', 'j2_surface_efficiency', 'j2_ace_percentage', 'j2_bp_saved_pct', 
    'j2_bp_converted_pct', 'j2_first_serve_won_pct', 'j2_fatiga_sets_7d', 'j2_elo_rating_norm',
    'ganador_j1'
]

# ==================== DATA VALIDATION ====================
class DataValidator:
    """Robust data validation and type checking for pandas DataFrames."""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, expected_columns: List[str]) -> bool:
        """Validate DataFrame structure and types."""
        try:
            if not isinstance(df, pd.DataFrame):
                logger.error(f"Expected pd.DataFrame, got {type(df)}")
                return False
            
            missing_cols = set(expected_columns) - set(df.columns)
            if missing_cols:
                logger.warning(f"Missing columns: {missing_cols}")
                return False
            
            # Check for NaN values
            if df.isnull().any().any():
                logger.warning(f"NaN values detected. Filling with median...")
                df = df.fillna(df.median(numeric_only=True))
            
            # Validate numeric types
            for col in expected_columns:
                if df[col].dtype not in [np.float64, np.float32, np.int64, np.int32]:
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    except Exception as e:
                        logger.error(f"Cannot convert column {col} to numeric: {e}")
                        return False
            
            logger.info(f"✅ DataFrame validation passed. Shape: {df.shape}")
            return True
        except Exception as e:
            logger.error(f"❌ DataFrame validation error: {e}")
            return False
    
    @staticmethod
    def validate_features(X: np.ndarray, feature_count: int) -> bool:
        """Validate feature array."""
        try:
            if not isinstance(X, (np.ndarray, pd.DataFrame)):
                raise TypeError(f"Expected numpy array or DataFrame, got {type(X)}")
            
            if X.shape[1] != feature_count:
                raise ValueError(f"Expected {feature_count} features, got {X.shape[1]}")
            
            if np.any(np.isnan(X)) or np.any(np.isinf(X)):
                raise ValueError("Features contain NaN or Inf values")
            
            return True
        except Exception as e:
            logger.error(f"❌ Feature validation error: {e}")
            return False

# ==================== ENSEMBLE MODEL BUILDER ====================
class EnsembleModelBuilder:
    """Build LightGBM + XGBoost ensemble with Voting/Stacking."""
    
    def __init__(self, random_state: int = 42, n_jobs: int = -1):
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.scaler = StandardScaler()
        self.model = None
        self.best_params = None
        
    def _create_base_models(self, **kwargs) -> List[Tuple[str, Any]]:
        """Create LightGBM and XGBoost base models."""
        lgb_params = kwargs.get('lgb_params', {})
        xgb_params = kwargs.get('xgb_params', {})
        
        lgb_model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.1,
            num_leaves=31,
            random_state=self.random_state,
            verbosity=-1,
            **lgb_params
        )
        
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.1,
            random_state=self.random_state,
            verbosity=0,
            use_label_encoder=False,
            eval_metric='logloss',
            **xgb_params
        )
        
        return [('lgb', lgb_model), ('xgb', xgb_model)]
    
    def create_voting_ensemble(self, **kwargs) -> VotingClassifier:
        """Create Voting Classifier ensemble."""
        base_models = self._create_base_models(**kwargs)
        return VotingClassifier(
            estimators=base_models,
            voting='soft',
            n_jobs=self.n_jobs
        )
    
    def create_stacking_ensemble(self, **kwargs) -> StackingClassifier:
        """Create Stacking Classifier ensemble."""
        base_models = self._create_base_models(**kwargs)
        meta_learner = LogisticRegression(random_state=self.random_state, max_iter=1000)
        
        return StackingClassifier(
            estimators=base_models,
            final_estimator=meta_learner,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray, ensemble_type: str = 'voting', **kwargs):
        """Fit the ensemble model."""
        try:
            if not DataValidator.validate_features(X, X.shape[1]):
                raise ValueError("Invalid feature array")
            
            X_scaled = self.scaler.fit_transform(X)
            
            if ensemble_type == 'voting':
                self.model = self.create_voting_ensemble(**kwargs)
            elif ensemble_type == 'stacking':
                self.model = self.create_stacking_ensemble(**kwargs)
            else:
                raise ValueError(f"Unknown ensemble type: {ensemble_type}")
            
            self.model.fit(X_scaled, y)
            logger.info(f"✅ {ensemble_type.upper()} ensemble model fitted successfully")
            return self
        except Exception as e:
            logger.error(f"❌ Error fitting ensemble model: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        try:
            if self.model is None:
                raise RuntimeError("Model not fitted yet")
            
            if not DataValidator.validate_features(X, X.shape[1]):
                raise ValueError("Invalid feature array")
            
            X_scaled = self.scaler.transform(X)
            return self.model.predict(X_scaled)
        except Exception as e:
            logger.error(f"❌ Prediction error: {e}")
            raise
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        try:
            if self.model is None:
                raise RuntimeError("Model not fitted yet")
            
            X_scaled = self.scaler.transform(X)
            return self.model.predict_proba(X_scaled)
        except Exception as e:
            logger.error(f"❌ Probability prediction error: {e}")
            raise

# ==================== BAYESIAN HYPERPARAMETER OPTIMIZATION ====================
class BayesianOptimizer:
    """Lightweight Bayesian optimization for mobile devices."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray, cv_splits: int = 3, max_evals: int = 10):
        self.X = X
        self.y = y
        self.cv_splits = cv_splits
        self.max_evals = max_evals  # Lightweight: reduced evaluations
        self.best_params = None
        self.best_score = None
    
    def optimize(self) -> Dict[str, Any]:
        """Run Bayesian optimization."""
        try:
            logger.info("🔍 Starting Bayesian Hyperparameter Optimization...")
            
            # Define search space (lightweight for mobile)
            space_params = [
                space.Integer(50, 300, name='n_estimators'),
                space.Integer(3, 10, name='max_depth'),
                space.Real(0.01, 0.3, name='learning_rate'),
                space.Integer(15, 50, name='num_leaves'),
            ]
            
            @use_named_args(space_params)
            def objective(**params):
                """Objective function for optimization."""
                try:
                    builder = EnsembleModelBuilder()
                    X_scaled = builder.scaler.fit_transform(self.X)
                    
                    kf = StratifiedKFold(n_splits=self.cv_splits, shuffle=True, random_state=42)
                    scores = []
                    
                    for train_idx, val_idx in kf.split(self.X, self.y):
                        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                        y_train, y_val = self.y.iloc[train_idx], self.y.iloc[val_idx]
                        
                        lgb_model = lgb.LGBMClassifier(
                            n_estimators=params['n_estimators'],
                            max_depth=params['max_depth'],
                            learning_rate=params['learning_rate'],
                            num_leaves=params['num_leaves'],
                            random_state=42,
                            verbosity=-1
                        )
                        
                        lgb_model.fit(X_train, y_train)
                        y_pred = lgb_model.predict(X_val)
                        score = accuracy_score(y_val, y_pred)
                        scores.append(score)
                    
                    return -np.mean(scores)  # Minimize negative accuracy
                except Exception as e:
                    logger.warning(f"⚠️ Optimization iteration failed: {e}")
                    return 1.0  # Return worst score on error
            
            # Run optimization
            result = gp_minimize(
                objective,
                space_params,
                n_calls=self.max_evals,
                random_state=42,
                n_initial_points=3
            )
            
            self.best_params = {
                'n_estimators': result.x[0],
                'max_depth': result.x[1],
                'learning_rate': result.x[2],
                'num_leaves': result.x[3]
            }
            self.best_score = -result.fun
            
            logger.info(f"✅ Optimization complete. Best Score: {self.best_score:.4f}")
            logger.info(f"   Best Parameters: {self.best_params}")
            
            return self.best_params
        except Exception as e:
            logger.error(f"❌ Bayesian optimization error: {e}")
            return {}

# ==================== ASYNC API OPERATIONS ====================
class AsyncAPIClient:
    """Async HTTP client for ScraperAPI with robust error handling."""
    
    def __init__(self, api_key: str, timeout: int = 15, max_retries: int = 3):
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.url_odds_origin = "https://the-odds-api.com"
    
    async def fetch_odds(self, session: aiohttp.ClientSession, chunk_index: int = 0) -> Dict[str, Any]:
        """Async fetch odds data from ScraperAPI."""
        url_proxy = f"http://scraperapi.com?api_key={self.api_key}&url={self.url_odds_origin}"
        
        for attempt in range(self.max_retries):
            try:
                async with session.get(url_proxy, timeout=self.timeout) as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"✅ Odds fetched successfully (attempt {attempt + 1})")
                        return data if data else {}
                    else:
                        logger.warning(f"⚠️ API returned status {response.status} (attempt {attempt + 1})")
            except asyncio.TimeoutError:
                logger.warning(f"⏱️ Timeout on attempt {attempt + 1}/{self.max_retries}")
            except aiohttp.ClientError as e:
                logger.warning(f"🌐 Network error on attempt {attempt + 1}/{self.max_retries}: {e}")
            except Exception as e:
                logger.error(f"❌ Unexpected error on attempt {attempt + 1}: {e}")
            
            if attempt < self.max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        logger.warning("⚠️ All API attempts failed. Using fallback data.")
        return {}
    
    async def fetch_multiple_matches(self, num_chunks: int = 1) -> List[Dict[str, Any]]:
        """Fetch multiple match data in parallel."""
        async with aiohttp.ClientSession() as session:
            tasks = [self.fetch_odds(session, i) for i in range(num_chunks)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions
            valid_results = [r for r in results if isinstance(r, dict) and r]
            return valid_results

# ==================== DATA GENERATION & PERSISTENCE ====================
def generar_matriz_profesional() -> pd.DataFrame:
    """Generate professional matrix seed."""
    try:
        datos = np.array([
            [0.78, 0.82, 0.12, 0.68, 0.45, 0.76, 3,  0.85,  0.45, 0.40, 0.05, 0.50, 0.35, 0.62, 9,  0.45,  1],
            [0.52, 0.48, 0.06, 0.55, 0.38, 0.64, 7,  0.55,  0.74, 0.79, 0.14, 0.72, 0.48, 0.79, 2,  0.78,  0],
            [0.61, 0.65, 0.08, 0.60, 0.42, 0.69, 5,  0.62,  0.59, 0.55, 0.07, 0.58, 0.40, 0.66, 6,  0.58,  1],
            [0.38, 0.30, 0.03, 0.48, 0.30, 0.58, 12, 0.40,  0.81, 0.85, 0.18, 0.75, 0.52, 0.82, 3,  0.82,  0],
            [0.69, 0.72, 0.11, 0.64, 0.44, 0.71, 4,  0.70,  0.63, 0.60, 0.09, 0.61, 0.41, 0.68, 5,  0.64,  1],
            [0.44, 0.50, 0.05, 0.52, 0.36, 0.61, 6,  0.48,  0.55, 0.52, 0.06, 0.54, 0.39, 0.63, 4,  0.52,  0],
            [0.72, 0.70, 0.10, 0.66, 0.46, 0.73, 2,  0.76,  0.41, 0.35, 0.04, 0.49, 0.32, 0.59, 10, 0.38,  1],
            [0.50, 0.55, 0.07, 0.57, 0.39, 0.65, 8,  0.53,  0.68, 0.71, 0.11, 0.65, 0.47, 0.74, 3,  0.71,  0],
            [0.85, 0.89, 0.15, 0.75, 0.52, 0.81, 2,  0.92,  0.33, 0.28, 0.02, 0.42, 0.28, 0.54, 8,  0.32,  1],
            [0.58, 0.60, 0.08, 0.59, 0.41, 0.67, 5,  0.59,  0.62, 0.64, 0.09, 0.63, 0.43, 0.70, 4,  0.63,  0]
        ])
        df = pd.DataFrame(datos, columns=COLUMNAS_PROFESIONALES)
        logger.info(f"✅ Professional matrix generated. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"❌ Error generating professional matrix: {e}")
        raise

def extraer_y_actualizar_historico() -> pd.DataFrame:
    """Extract and update historical data with validation."""
    try:
        if os.path.exists(FILE_HISTORICO):
            df_historico = pd.read_csv(FILE_HISTORICO)
        else:
            df_historico = generar_matriz_profesional()
        
        # Validate and reindex
        if not DataValidator.validate_dataframe(df_historico, COLUMNAS_PROFESIONALES):
            logger.warning("⚠️ Validation failed, regenerating matrix...")
            df_historico = generar_matriz_profesional()
        
        df_historico = df_historico.reindex(columns=COLUMNAS_PROFESIONALES, fill_value=0.50)
        df_historico.to_csv(FILE_HISTORICO, index=False)
        logger.info(f"✅ Historical data updated. Shape: {df_historico.shape}")
        return df_historico
    except Exception as e:
        logger.error(f"❌ Error updating historical data: {e}")
        raise

def obtener_cartelera_respaldo() -> Dict[str, Dict[str, Any]]:
    """Fallback market data."""
    return {
        "Carlos Alcaraz vs Jannik Sinner": {
            "metricas": [0.82, 0.85, 0.11, 0.70, 0.48, 0.78, 2, 0.90,  0.80, 0.78, 0.13, 0.69, 0.46, 0.77, 3, 0.89],
            "cuota": 1.68,
            "prediccion_esperada": 1
        },
        "Daniil Medvedev vs Alexander Zverev": {
            "metricas": [0.76, 0.80, 0.14, 0.72, 0.45, 0.81, 3, 0.78,  0.44, 0.38, 0.04, 0.51, 0.33, 0.60, 8, 0.42],
            "cuota": 1.55,
            "prediccion_esperada": 1
        },
        "Novak Djokovic vs Taylor Fritz": {
            "metricas": [0.85, 0.88, 0.12, 0.74, 0.50, 0.79, 1, 0.93,  0.50, 0.48, 0.06, 0.54, 0.37, 0.63, 11, 0.51],
            "cuota": 1.82,
            "prediccion_esperada": 1
        },
        "Casper Ruud vs Holger Rune": {
            "metricas": [0.61, 0.65, 0.08, 0.60, 0.42, 0.69, 5, 0.62,  0.59, 0.55, 0.07, 0.58, 0.40, 0.66, 6, 0.58],
            "cuota": 1.22,
            "prediccion_esperada": 1
        }
    }

def extraer_cuotas_reales_api_async() -> Dict[str, Dict[str, Any]]:
    """Wrapper for async API calls."""
    try:
        api_client = AsyncAPIClient(config.api_key, timeout=config.timeout_seconds, max_retries=config.max_retries)
        
        # Run async fetch
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        results = loop.run_until_complete(api_client.fetch_multiple_matches(num_chunks=1))
        loop.close()
        
        if results:
            logger.info(f"📡 Retrieved {len(results)} data chunks from API")
        else:
            logger.warning("⚠️ No data retrieved. Using fallback.")
        
        return procesar_datos_api(results)
    except Exception as e:
        logger.error(f"❌ Error in async API call: {e}")
        return obtener_cartelera_respaldo()

def procesar_datos_api(data: List[Dict]) -> Dict[str, Dict[str, Any]]:
    """Process API response data."""
    try:
        cartelera = {}
        
        if not data or not data[0]:
            logger.warning("⚠️ Empty API response")
            return obtener_cartelera_respaldo()
        
        events = data[0].get('events', [])[:5]  # First 5 matches
        
        for evento in events:
            try:
                home_team = evento.get("home_team", "Team 1")
                away_team = evento.get("away_team", "Team 2")
                label = f"{home_team} vs {away_team}"
                
                cuota_home = 1.80
                cuota_away = 1.80
                
                bookmakers = evento.get("bookmakers", [])
                if bookmakers:
                    markets = bookmakers[0].get("markets", [])
                    if markets:
                        outcomes = markets[0].get("outcomes", [])
                        for out in outcomes:
                            if out.get("name") == home_team:
                                cuota_home = float(out.get("price", 1.80))
                            else:
                                cuota_away = float(out.get("price", 1.80))
                
                # Generate metrics based on odds hierarchy
                if cuota_home < cuota_away:
                    metricas = [0.80, 0.82, 0.12, 0.68, 0.46, 0.75, 2, 0.88,  0.48, 0.42, 0.05, 0.52, 0.35, 0.61, 6, 0.49]
                else:
                    metricas = [0.50, 0.48, 0.05, 0.54, 0.38, 0.63, 6, 0.51,  0.78, 0.81, 0.11, 0.67, 0.44, 0.74, 2, 0.84]
                
                cuota = cuota_home if cuota_home < cuota_away else cuota_away
                cartelera[label] = {
                    "metricas": metricas,
                    "cuota": cuota,
                    "prediccion_esperada": 1 if cuota_home < cuota_away else 2
                }
            except Exception as e:
                logger.warning(f"⚠️ Error processing event: {e}")
                continue
        
        if cartelera:
            logger.info(f"✅ Processed {len(cartelera)} matches from API")
            return cartelera
        else:
            logger.warning("⚠️ No valid matches processed. Using fallback.")
            return obtener_cartelera_respaldo()
    
    except Exception as e:
        logger.error(f"❌ Error processing API data: {e}")
        return obtener_cartelera_respaldo()

# ==================== KELLY CRITERION ====================
def calcular_criterio_kelly(probabilidad_modelo: float, cuota_casa: float) -> Tuple[float, float]:
    """Calculate Kelly Criterion for position sizing."""
    try:
        if not isinstance(probabilidad_modelo, (int, float)) or not isinstance(cuota_casa, (int, float)):
            raise TypeError("Probability and odds must be numeric")
        
        p = probabilidad_modelo / 100.0
        k = cuota_casa
        
        if k <= 1.0 or p <= 0 or p >= 1:
            return 0.0, 0.0
        
        kelly_raw = ((p * k) - 1) / (k - 1)
        kelly_final = kelly_raw * config.kelly_fraction
        
        if kelly_final < 0:
            return 0.0, 0.0
        
        return kelly_final * 100, config.capital_total * kelly_final
    except Exception as e:
        logger.error(f"❌ Kelly criterion calculation error: {e}")
        return 0.0, 0.0

# ==================== MAIN PIPELINE ====================
async def ejecutar_pipeline_completo():
    """Execute complete ML pipeline with ensemble and async operations."""
    try:
        logger.info("=" * 70)
        logger.info("🚀 INICIANDO PIPELINE DE ML INSTITUCIONAL PARA TENIS")
        logger.info("=" * 70)
        
        # 1. Configuration validation
        if not config.validate():
            sys.exit(1)
        
        # 2. Load and validate training data
        logger.info("\n📚 Cargando datos de entrenamiento...")
        df_entrenamiento = extraer_y_actualizar_historico()
        
        if df_entrenamiento is None or df_entrenamiento.empty:
            raise ValueError("No training data available")
        
        # 3. Prepare features and target
        features = [col for col in COLUMNAS_PROFESIONALES if col != 'ganador_j1']
        X = df_entrenamiento[features].values
        y = df_entrenamiento['ganador_j1'].values
        
        if not DataValidator.validate_features(X, len(features)):
            raise ValueError("Features validation failed")
        
        logger.info(f"✅ Datos validados. Shape: {X.shape}, Target balance: {np.mean(y):.2%}")
        
        # 4. Bayesian Hyperparameter Optimization
        logger.info("\n🔍 Ejecutando Búsqueda Bayesiana (Optimización Ligera para Móviles)...")
        optimizer = BayesianOptimizer(
            X=X,
            y=pd.Series(y),
            cv_splits=3,
            max_evals=10  # Lightweight for mobile
        )
        best_params = optimizer.optimize()
        
        # 5. Train ensemble model
        logger.info("\n🤖 Entrenando Ensemble (LightGBM + XGBoost)...")
        builder = EnsembleModelBuilder()
        builder.fit(X, y, ensemble_type='stacking', lgb_params=best_params)
        
        logger.info("✅ Modelo Ensemble entrenado exitosamente")
        
        # 6. Async API calls to fetch real odds
        logger.info("\n📡 Obteniendo cuotas en tiempo real (Async/ScraperAPI)...")
        cartelera_hoy = extraer_cuotas_reales_api_async()
        
        if not cartelera_hoy:
            logger.warning("⚠️ Using fallback market data")
            cartelera_hoy = obtener_cartelera_respaldo()
        
        # 7. Make predictions and calculate Kelly
        logger.info("\n" + "=" * 70)
        logger.info("💰 INFORME FINANCIERO Y ANÁLISIS DE CUOTAS EN VIVO 💰")
        logger.info("=" * 70)
        logger.info(f"Banca Operativa: ${config.capital_total} | Riesgo: {config.kelly_fraction}")
        logger.info("-" * 70)
        
        partidos_guardados = []
        
        for nombre_partido, info in cartelera_hoy.items():
            try:
                metricas = np.array(info["metricas"]).reshape(1, -1)
                cuota_casa = info["cuota"]
                
                # Make predictions
                prediccion = builder.predict(metricas)[0]
                probabilidades = builder.predict_proba(metricas)[0]
                porcentaje_final = max(probabilidades) * 100
                
                # Calculate Kelly criterion
                pct_kelly, monto_apuesta = calcular_criterio_kelly(porcentaje_final, cuota_casa)
                ganador_label = f"JUGADOR {int(prediccion) + 1}"
                
                logger.info(f"\n🎾 Partido: {nombre_partido}")
                logger.info(f"   📊 Probabilidad: {porcentaje_final:.2f}%")
                logger.info(f"   🎲 Cuota API: {cuota_casa}")
                logger.info(f"   🎯 Predicción: {ganador_label}")
                
                if porcentaje_final >= 70.0 and monto_apuesta > 0:
                    logger.info(f"   ✅ OPERACIÓN RENTABLE DETECTADA (+70% & VALOR)")
                    logger.info(f"   💵 Kelly %: {pct_kelly:.2f}%")
                    logger.info(f"   💰 INVERTIR: ${monto_apuesta:.2f}")
                elif porcentaje_final >= 70.0 and monto_apuesta == 0:
                    logger.info(f"   ⚠️ ALERTA: Confianza alta pero cuota castigada. OMITIR.")
                else:
                    logger.info(f"   ❌ RECHAZADO: Confianza insuficiente.")
                
                # Store prediction
                partidos_guardados.append(metricas[0].tolist() + [int(prediccion), round(porcentaje_final, 2)])
            
            except Exception as e:
                logger.error(f"❌ Error predicting match {nombre_partido}: {e}")
                continue
        
        logger.info("\n" + "=" * 70 + "\n")
        
        # 8. Save predictions to CSV
        if partidos_guardados:
            logger.info("💾 Guardando predicciones en histórico...")
            
            try:
                if os.path.exists(FILE_PREDICCIONES):
                    df_pred_antiguo = pd.read_csv(FILE_PREDICCIONES)
                else:
                    df_pred_antiguo = pd.DataFrame()
                
                df_nuevas_pred = pd.DataFrame(
                    partidos_guardados,
                    columns=features + ['ganador_predicho', 'probabilidad_calculada']
                )
                
                df_final_pred = pd.concat([df_pred_antiguo, df_nuevas_pred], ignore_index=True)
                df_final_pred.to_csv(FILE_PREDICCIONES, index=False)
                
                logger.info(f"✅ Predicciones guardadas. Total registros: {len(df_final_pred)}")
            except Exception as e:
                logger.error(f"❌ Error saving predictions: {e}")
        
        logger.info("\n✅ PIPELINE COMPLETADO EXITOSAMENTE")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"❌ Pipeline error: {e}", exc_info=True)
        sys.exit(1)

# ==================== ENTRY POINT ====================
if __name__ == "__main__":
    try:
        # Run async pipeline
        asyncio.run(ejecutar_pipeline_completo())
    except KeyboardInterrupt:
        logger.warning("⚠️ Pipeline interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"❌ Critical error: {e}", exc_info=True)
        sys.exit(1)
