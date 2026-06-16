"""
GlicemIA — Avaliação do Modelo Treinado
========================================
Carrega o modelo salvo e gera relatório completo de métricas.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

try:
    import tensorflow as tf
    TENSORFLOW_DISPONIVEL = True
except ImportError:
    TENSORFLOW_DISPONIVEL = False

from src.preprocessing.preprocessamento import preparar_dados, gerar_dados_demo
from sklearn.preprocessing import MinMaxScaler


def avaliar_modelo(
    caminho_modelo: str = "models/modelo_glicemia.keras",
    caminho_dados: str = "data/dataset/Shanghai_T1DM/1001_0_20210730.xlsx"
):
    """
    Carrega modelo salvo e avalia sobre o conjunto de validação.
    Gera relatório de métricas e gráficos comparativos.
    """
    if not TENSORFLOW_DISPONIVEL:
        print("❌ TensorFlow não disponível.")
        return

    if not Path(caminho_modelo).exists():
        print(f"❌ Modelo não encontrado em: {caminho_modelo}")
        print("   Execute primeiro: python src/model/treinar.py")
        return

    print("=" * 50)
    print("GlicemIA — Relatório de Avaliação do Modelo")
    print("=" * 50)

    # Carregar modelo
    model = tf.keras.models.load_model(caminho_modelo)
    print(f"\n✅ Modelo carregado: {caminho_modelo}")

    # Carregar dados
    X, y, scaler = preparar_dados(caminho_dados)

    # Split (mesmo do treino)
    split = int(len(X) * 0.8)
    X_val = X[split:].reshape(-1, X.shape[1], 1)
    y_val = y[split:]

    # Predição
    y_pred_norm = model.predict(X_val, verbose=0).flatten()

    # Desnormalizar
    y_pred_mgdl = scaler.inverse_transform(y_pred_norm.reshape(-1, 1)).flatten()
    y_real_mgdl = scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()

    # Métricas
    mae = np.mean(np.abs(y_pred_mgdl - y_real_mgdl))
    rmse = np.sqrt(np.mean((y_pred_mgdl - y_real_mgdl) ** 2))
    mape = np.mean(np.abs((y_real_mgdl - y_pred_mgdl) / y_real_mgdl)) * 100

    print(f"\n{'─'*40}")
    print(f"  MÉTRICAS (escala real em mg/dL)")
    print(f"{'─'*40}")
    print(f"  MAE  (Erro Médio Absoluto)    : {mae:.2f} mg/dL")
    print(f"  RMSE (Raiz do Erro Quadrático): {rmse:.2f} mg/dL")
    print(f"  MAPE (Erro Percentual Médio)  : {mape:.2f}%")
    print(f"{'─'*40}")

    # Avaliação por faixa clínica
    _avaliar_por_faixa(y_real_mgdl, y_pred_mgdl)

    # Carregar métricas do treino se existir
    metricas_path = "models/metricas.json"
    if Path(metricas_path).exists():
        with open(metricas_path) as f:
            metricas_treino = json.load(f)
        print(f"\n  Épocas executadas  : {metricas_treino.get('epochs_executadas', 'N/A')}")
        print(f"  Loss final (treino): {metricas_treino.get('loss_final_treino', 0):.6f}")
        print(f"  Loss final (val)   : {metricas_treino.get('loss_final_val', 0):.6f}")

    print(f"{'='*50}")


def _avaliar_por_faixa(y_real: np.ndarray, y_pred: np.ndarray):
    """Avalia a acurácia de classificação por faixa clínica."""
    print(f"\n  CLASSIFICAÇÃO POR FAIXA CLÍNICA")
    print(f"{'─'*40}")

    def faixa(v):
        if v < 70:
            return "HIPOGLICEMIA"
        elif v < 90:
            return "ALERTA"
        elif v <= 180:
            return "NORMAL"
        else:
            return "HIPERGLICEMIA"

    faixas_real = np.array([faixa(v) for v in y_real])
    faixas_pred = np.array([faixa(v) for v in y_pred])

    acertos = np.sum(faixas_real == faixas_pred)
    total = len(faixas_real)
    acuracia = acertos / total * 100

    print(f"  Acurácia de Classificação de Risco: {acuracia:.1f}%")
    print(f"  ({acertos}/{total} amostras classificadas corretamente)")

    for cat in ["HIPOGLICEMIA", "ALERTA", "NORMAL", "HIPERGLICEMIA"]:
        n = np.sum(faixas_real == cat)
        if n > 0:
            corretos = np.sum((faixas_real == cat) & (faixas_pred == cat))
            print(f"    {cat:<15}: {corretos}/{n} ({corretos/n*100:.0f}%)")


if __name__ == "__main__":
    avaliar_modelo()
