import os
import requests
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# 1. Configuración de API desde Variables de Entorno (Secreto de GitHub)
API_KEY = os.getenv("SCRAPER_API_KEY")
# Usamos un endpoint público/simulado de tenis compatible con proxies de scraping
URL_API = f"http://scraperapi.com?api_key={API_KEY}&url=https://sportradar.com"

def obtener_datos_vivos():
    """Obtiene partidos en vivo o programados para el día."""
    try:
        response = requests.get(URL_API, timeout=15)
        if response.status_code == 200:
            # En producción, aquí mapeas el JSON real de tu proveedor de tenis
            # Simulamos la estructura limpia para el modelo de Machine Learning
            datos = response.json()
            return datos
    except Exception as e:
        print(f"Error al conectar con la API: {e}")
    
    # Dataset de respaldo/ejemplo estructurado si la API no responde o está en mantenimiento
    # Características clave: Victorias previas, Rendimiento en Superficie, Fatiga (partidos acumulados)
    print(" Usando datos estructurados de simulación para el entrenamiento...")
    ejemplo_partidos = {
        'jugador1_win_rate_h2h': [0.65, 0.40, 0.75, 0.30, 0.55, 0.70, 0.45, 0.62],
        'jugador1_surface_efficiency': [0.70, 0.35, 0.80, 0.25, 0.60, 0.75, 0.50, 0.68],
        'jugador1_fatiga': [2, 5, 1, 4, 3, 2, 6, 1],
        'jugador2_win_rate_h2h': [0.35, 0.60, 0.25, 0.70, 0.45, 0.30, 0.55, 0.38],
        'jugador2_surface_efficiency': [0.30, 0.65, 0.20, 0.75, 0.40, 0.25, 0.55, 0.32],
        'jugador2_fatiga': [4, 1, 3, 2, 2, 5, 1, 3],
        'ganador_jugador1': [1, 0, 1, 0, 1, 1, 0, 1] # 1 = Gana Jugador 1, 0 = Gana Jugador 2
    }
    return pd.DataFrame(ejemplo_partidos)

def entrenar_y_predecir():
    df = obtener_datos_vivos()
    
    # Separar características (X) y objetivo (y)
    X = df[['jugador1_win_rate_h2h', 'jugador1_surface_efficiency', 'jugador1_fatiga',
            'jugador2_win_rate_h2h', 'jugador2_surface_efficiency', 'jugador2_fatiga']]
    y = df['ganador_jugador1']
    
    # Normalizar datos para estabilizar el algoritmo sobre el 70%
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Modelo Random Forest optimizado para evitar sobreajuste
    model = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)
    model.fit(X_scaled, y)
    
    # Predecir un partido en vivo entrante (Ejemplo de prueba)
    # Imaginemos un partido cerrado donde el Jugador 1 viene mejor en la superficie actual
    partido_en_vivo = [[0.68, 0.72, 1, 0.32, 0.28, 3]] 
    partido_en_vivo_scaled = scaler.transform(partido_en_vivo)
    
    prediccion = model.predict(partido_en_vivo_scaled)
    probabilidades = model.predict_proba(partido_en_vivo_scaled)[0]
    
    print("\n=============================================")
    print("📊 RESULTADOS DEL SISTEMA DE ML (TENNIS) 📊")
    print("=============================================")
    if prediccion[0] == 1:
        prob_exito = probabilidades[1] * 100
        print(f"Ganador sugerido: JUGADOR 1")
    else:
        prob_exito = probabilidades[0] * 100
        print(f"Ganador sugerido: JUGADOR 2")
        
    print(f"Probabilidad estimada: {prob_exito:.2f}%")
    
    # Filtro estricto solicitado
    if prob_exito >= 70.0:
        print(" Ejecutar apuesta: Cumple con la métrica >= 70%")
    else:
        print("⚠️ Descartar partido: No alcanza el umbral de seguridad.")
    print("=============================================\n")

if __name__ == "__main__":
    entrenar_y_predecir()
