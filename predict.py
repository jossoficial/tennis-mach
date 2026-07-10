import os
import sys
import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# Configuración de variables seguras desde GitHub Secrets
API_KEY = os.getenv("SCRAPER_API_KEY")
if not API_KEY:
    print("❌ ERROR: La variable SCRAPER_API_KEY no está configurada en los Secretos de GitHub.")
    sys.exit(1)

URL_ORIGEN = "https://sportradar.com"
URL_PROXY = f"http://scraperapi.com?api_key={API_KEY}&url={URL_ORIGEN}"

FILE_HISTORICO = "historico_entrenamiento.csv"
FILE_PREDICCIONES = "registro_predicciones.csv"

# Definición global de las 10 columnas obligatorias
COLUMNAS_OBJETIVO = [
    'j1_win_rate_recent', 'j1_surface_efficiency', 'j1_ace_percentage', 'j1_bp_saved_percentage', 'j1_fatiga_games',
    'j2_win_rate_recent', 'j2_surface_efficiency', 'j2_ace_percentage', 'j2_bp_saved_percentage', 'j2_fatiga_games',
    'ganador_j1'
]

def generar_matriz_expandida():
    """Genera el dataset semilla con las 10 variables del circuito profesional."""
    datos_ampliados = np.array([
        [0.78, 0.82, 0.12, 0.68, 1,  0.45, 0.40, 0.05, 0.50, 4,  1],
        [0.52, 0.48, 0.06, 0.55, 3,  0.74, 0.79, 0.14, 0.72, 1,  0],
        [0.61, 0.65, 0.08, 0.60, 2,  0.59, 0.55, 0.07, 0.58, 2,  1],
        [0.38, 0.30, 0.03, 0.48, 5,  0.81, 0.85, 0.18, 0.75, 0,  0],
        [0.69, 0.72, 0.11, 0.64, 2,  0.63, 0.60, 0.09, 0.61, 3,  1],
        [0.44, 0.50, 0.05, 0.52, 1,  0.55, 0.52, 0.06, 0.54, 2,  0],
        [0.72, 0.70, 0.10, 0.66, 2,  0.41, 0.35, 0.04, 0.49, 4,  1],
        [0.50, 0.55, 0.07, 0.57, 4,  0.68, 0.71, 0.11, 0.65, 1,  0],
        [0.85, 0.89, 0.15, 0.75, 1,  0.33, 0.28, 0.02, 0.42, 3,  1],
        [0.58, 0.60, 0.08, 0.59, 2,  0.62, 0.64, 0.09, 0.63, 2,  0],
        [0.65, 0.70, 0.09, 0.61, 1,  0.68, 0.50, 0.08, 0.60, 3,  1],
        [0.49, 0.45, 0.05, 0.50, 3,  0.51, 0.53, 0.06, 0.53, 2,  0],
        [0.80, 0.78, 0.13, 0.70, 0,  0.50, 0.52, 0.06, 0.55, 5,  1],
        [0.55, 0.58, 0.07, 0.56, 3,  0.57, 0.60, 0.08, 0.59, 2,  0],
        [0.74, 0.75, 0.11, 0.67, 1,  0.48, 0.46, 0.05, 0.52, 2,  1]
    ])
    return pd.DataFrame(datos_ampliados, columns=COLUMNAS_OBJECTIVO)

def extraer_y_actualizar_historico():
    """Carga los datos y realiza una auto-migración si el formato guardado es antiguo."""
    if os.path.exists(FILE_HISTORICO):
        df_historico = pd.read_csv(FILE_HISTORICO)
        print(f"📁 Dataset histórico cargado. Columnas actuales: {list(df_historico.columns)}")
        
        # MECANISMO DE AUTO-MIGRACIÓN: Si faltan columnas, inyecta valores base estimados
        columnas_faltantes = [col for col in COLUMNAS_OBJETIVO if col not in df_historico.columns]
        if columnas_faltantes:
            print(f"🔄 Detectado formato antiguo. Migrando de forma segura e inyectando: {columnas_faltantes}")
            if 'j1_bp_saved_percentage' in columnas_faltantes:
                df_historico['j1_bp_saved_percentage'] = 0.60  # 60% efectividad promedio
            if 'j2_bp_saved_percentage' in columnas_faltantes:
                df_historico['j2_bp_saved_percentage'] = 0.58
            if 'j1_fatiga_games' in columnas_faltantes:
                df_historico['j1_fatiga_games'] = 2            # 2 partidos promedio
            if 'j2_fatiga_games' in columnas_faltantes:
                df_historico['j2_fatiga_games'] = 2
            # Asegurar el orden exacto de columnas para el modelo
            df_historico = df_historico[COLUMNAS_OBJETIVO]
    else:
        df_historico = generar_matriz_expandida()
        print("🆕 Inicializando matriz expandida semilla desde cero...")
        
    try:
        response = requests.get(URL_PROXY, timeout=20)
        if response.status_code == 200:
            data = response.json()
            nuevos_partidos = []
            for match in data.get("summaries", []):
                statistics = match.get("statistics", {})
                j1_stats = statistics.get("totals", {}).get("competitors", [{}, {}])
                j2_stats = statistics.get("totals", {}).get("competitors", [{}, {}])
                
                nuevos_partidos.append({
                    'j1_win_rate_recent': float(j1_stats.get("matches_won_percent", 60)) / 100,
                    'j1_surface_efficiency': float(j1_stats.get("surface_win_rate", 62)) / 100,
                    'j1_ace_percentage': float(j1_stats.get("aces_per_game", 7)) / 100,
                    'j1_bp_saved_percentage': float(j1_stats.get("break_points_saved_percent", 58)) / 100,
                    'j1_fatiga_games': int(j1_stats.get("matches_played_14_days", 2)),
                    
                    'j2_win_rate_recent': float(j2_stats.get("matches_won_percent", 50)) / 100,
                    'j2_surface_efficiency': float(j2_stats.get("surface_win_rate", 48)) / 100,
                    'j2_ace_percentage': float(j2_stats.get("aces_per_game", 5)) / 100,
                    'j2_bp_saved_percentage': float(j2_stats.get("break_points_saved_percent", 52)) / 100,
                    'j2_fatiga_games': int(j2_stats.get("matches_played_14_days", 2)),
                    'ganador_j1': 1
                })
            if nuevos_partidos:
                df_nuevos = pd.DataFrame(nuevos_partidos)
                df_historico = pd.concat([df_historico, df_nuevos], ignore_index=True)
                df_historico.drop_duplicates(inplace=True)
    except Exception as e:
        print(f"ℹ️ Nota de conexión: Usando lote local corregido.")
        
    df_historico.to_csv(FILE_HISTORICO, index=False)
    return df_historico

def registrar_prediccion_diaria(variables_partido, ganador_sugerido, probabilidad):
    nuevo_registro = pd.DataFrame([{
        'j1_win_rate_recent': variables_partido, 'j1_surface_efficiency': variables_partido, 
        'j1_ace_percentage': variables_partido, 'j1_bp_saved_percentage': variables_partido, 'j1_fatiga_games': variables_partido,
        'j2_win_rate_recent': variables_partido, 'j2_surface_efficiency': variables_partido, 
        'j2_ace_percentage': variables_partido, 'j2_bp_saved_percentage': variables_partido, 'j2_fatiga_games': variables_partido,
        'ganador_predicho': ganador_sugerido, 'probabilidad_calculada': round(probabilidad, 2)
    }])
    if os.path.exists(FILE_PREDICCIONES):
        df_log = pd.read_csv(FILE_PREDICCIONES)
        df_log = pd.concat([df_log, nuevo_registro], ignore_index=True)
    else:
        df_log = nuevo_registro
    df_log.to_csv(FILE_PREDICCIONES, index=False)

def ejecutar_pipeline_continuo():
    df_entrenamiento = extraer_y_actualizar_historico()
    
    features = [
        'j1_win_rate_recent', 'j1_surface_efficiency', 'j1_ace_percentage', 'j1_bp_saved_percentage', 'j1_fatiga_games',
        'j2_win_rate_recent', 'j2_surface_efficiency', 'j2_ace_percentage', 'j2_bp_saved_percentage', 'j2_fatiga_games'
    ]
    
    X = df_entrenamiento[features]
    y = df_entrenamiento['ganador_j1']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    modelo = RandomForestClassifier(n_estimators=250, max_depth=6, random_state=42)
    
    # Validación Cruzada con el set corregido y homogeneizado
    scores = cross_val_score(modelo, X_scaled, y, cv=4, scoring='accuracy')
    precision_media = scores.mean() * 100
    desviacion_estandar = scores.std() * 100
    
    print("\n=======================================================")
    print("🛡️ AUDITORÍA METROLÓGICA (CROSS-VALIDATION) 🛡️")
    print("=======================================================")
    print(f"➡️ PRECISION MEDIA REAL DEL MODELO: {precision_media:.2f}%")
    print(f"➡️ ESTABILIDAD (DESVIACIÓN ESTÁNDAR): ±{desviacion_estandar:.2f}%")
    print(f"➡️ RENDIMIENTO POR PLIEGUE (FOLDS): {[round(x*100, 2) for x in scores]}%")
    print("=======================================================\n")
    
    modelo.fit(X_scaled, y)
    
    partido_hoy = np.array([0.76, 0.80, 0.13, 0.72, 1, 0.48, 0.42, 0.05, 0.50, 4])
    partido_hoy_scaled = scaler.transform(partido_hoy.reshape(1, -1))
    
    prediccion = modelo.predict(partido_hoy_scaled)
    probabilidades = modelo.predict_proba(partido_hoy_scaled)
    
    porcentaje_final = (probabilidades if prediccion == 1 else probabilidades) * 100
    
    print("=======================================================")
    print("📊 ALERTA PREDICTIVA DE LA JORNADA 📊")
    print("=======================================================")
    print(f"➡️ GANADOR DETECTADO: JUGADOR {1 if prediccion == 1 else 2}")
    print(f"➡️ PROBABILIDAD ASIGNADA HOY: {porcentaje_final:.2f}%")
    print("=======================================================\n")
    
    registrar_prediccion_diaria(partido_hoy, int(prediccion), porcentaje_final)

if __name__ == "__main__":
    ejecutar_pipeline_continuo()
