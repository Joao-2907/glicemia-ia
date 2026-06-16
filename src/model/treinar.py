"""
GlicemIA — Treinamento do Modelo LSTM
======================================
Define, treina e salva o modelo LSTM de predição glicêmica.

Spec de referência: docs/specs/spec-modelo-lstm.md

Arquitetura:
  Input     → (batch, 60, 1)
  LSTM(64)  → return_sequences=True
  Dropout   → 0.2
  LSTM(32)  → return_sequences=False
  Dropout   → 0.2
  Dense(1)  → valor predito normalizado
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Importação condicional do TensorFlow (pode não estar instalado em ambiente de demo)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    TENSORFLOW_DISPONIVEL = True
except ImportError:
    TENSORFLOW_DISPONIVEL = False
    print("⚠️  TensorFlow não encontrado. Execute: pip install tensorflow")

from src.preprocessing.preprocessamento import preparar_dados


def construir_modelo(tamanho_janela: int = 60) -> "tf.keras.Model":
    """
    Constrói o modelo LSTM conforme arquitetura especificada.

    Parâmetros
    ----------
    tamanho_janela : int
        Número de leituras de entrada (deve ser o mesmo do pré-processamento)

    Retorna
    -------
    model : tf.keras.Model compilado
    """
    model = Sequential([
        LSTM(64, input_shape=(tamanho_janela, 1), return_sequences=True,
             name="lstm_1"),
        Dropout(0.2, name="dropout_1"),
        LSTM(32, return_sequences=False, name="lstm_2"),
        Dropout(0.2, name="dropout_2"),
        Dense(1, name="saida_predicao")
    ])

    model.compile(
        optimizer="adam",
        loss="mean_squared_error",
        metrics=["mae"]
    )

    return model


def treinar(
    caminho_dados: str,
    tamanho_janela: int = 60,
    horizonte: int = 60,
    epochs: int = 50,
    batch_size: int = 32,
    diretorio_saida: str = "models/"
) -> dict:
    """
    Pipeline completo de treinamento:
    1. Pré-processamento dos dados
    2. Split temporal 80/20
    3. Reshape para LSTM (n, janela, 1)
    4. Treino com EarlyStopping
    5. Avaliação em mg/dL
    6. Salvamento do modelo e métricas

    Retorna
    -------
    dict com métricas: mae_mgdl, rmse_mgdl, loss_history
    """
    if not TENSORFLOW_DISPONIVEL:
        raise ImportError("TensorFlow é necessário para treinar o modelo.")

    # 1. Carregar e pré-processar dados
    print("\n[TREINO] Etapa 1/5: Pré-processando dados...")
    X, y, scaler = preparar_dados(caminho_dados, tamanho_janela, horizonte)

    # 2. Split temporal 80/20 (sem shuffle — seria data leakage em séries temporais)
    print("[TREINO] Etapa 2/5: Dividindo dados (split temporal 80/20)...")
    split = int(len(X) * 0.8)
    X_treino, X_val = X[:split], X[split:]
    y_treino, y_val = y[:split], y[split:]

    print(f"  Amostras de treino    : {len(X_treino)}")
    print(f"  Amostras de validação : {len(X_val)}")

    # 3. Reshape para LSTM: (n, janela) → (n, janela, 1)
    X_treino = X_treino.reshape((X_treino.shape[0], X_treino.shape[1], 1))
    X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))

    # 4. Construir modelo
    print("[TREINO] Etapa 3/5: Construindo modelo LSTM...")
    model = construir_modelo(tamanho_janela)
    model.summary()

    # 5. Callbacks
    os.makedirs(diretorio_saida, exist_ok=True)
    caminho_modelo = os.path.join(diretorio_saida, "modelo_glicemia.keras")

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=caminho_modelo,
            monitor="val_loss",
            save_best_only=True,
            verbose=0
        )
    ]

    # 6. Treinamento
    print("[TREINO] Etapa 4/5: Treinando modelo...")
    historico = model.fit(
        X_treino, y_treino,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    # 7. Avaliação em escala real (mg/dL)
    print("[TREINO] Etapa 5/5: Avaliando métricas em mg/dL...")
    y_pred_norm = model.predict(X_val, verbose=0).flatten()

    # Desnormalizar usando o scaler treinado
    y_pred_mgdl = scaler.inverse_transform(y_pred_norm.reshape(-1, 1)).flatten()
    y_real_mgdl = scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()

    mae = float(np.mean(np.abs(y_pred_mgdl - y_real_mgdl)))
    rmse = float(np.sqrt(np.mean((y_pred_mgdl - y_real_mgdl) ** 2)))

    print(f"\n{'='*40}")
    print(f"  MÉTRICAS DE AVALIAÇÃO")
    print(f"  MAE  (Erro Médio Absoluto) : {mae:.2f} mg/dL")
    print(f"  RMSE (Raiz do Erro Quad.)  : {rmse:.2f} mg/dL")
    print(f"{'='*40}\n")

    # 8. Salvar métricas e histórico
    metricas = {
        "mae_mgdl": mae,
        "rmse_mgdl": rmse,
        "epochs_executadas": len(historico.history["loss"]),
        "loss_final_treino": float(historico.history["loss"][-1]),
        "loss_final_val": float(historico.history["val_loss"][-1])
    }

    with open(os.path.join(diretorio_saida, "metricas.json"), "w") as f:
        json.dump(metricas, f, indent=2)

    # 9. Gerar gráfico predição vs real
    _salvar_grafico_predicao(y_real_mgdl, y_pred_mgdl, diretorio_saida)
    _salvar_grafico_loss(historico, diretorio_saida)

    print(f"[TREINO] ✅ Modelo salvo em: {caminho_modelo}")

    return metricas


def _salvar_grafico_predicao(
    y_real: np.ndarray,
    y_pred: np.ndarray,
    diretorio: str,
    n_amostras: int = 200
):
    """Gera e salva gráfico de predição vs valor real."""
    os.makedirs("outputs", exist_ok=True)
    plt.figure(figsize=(14, 5))
    plt.plot(y_real[:n_amostras], label="Glicemia Real (mg/dL)", color="#065A82")
    plt.plot(y_pred[:n_amostras], label="Predição LSTM (mg/dL)", color="#F96167",
             linestyle="--")
    plt.axhline(y=70, color="red", linestyle=":", alpha=0.5, label="Limiar Hipoglicemia (70)")
    plt.axhline(y=180, color="orange", linestyle=":", alpha=0.5, label="Limiar Hiperglicemia (180)")
    plt.title("GlicemIA — Predição vs Real", fontsize=14, fontweight="bold")
    plt.xlabel("Amostras")
    plt.ylabel("Glicemia (mg/dL)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/predicao_vs_real.png", dpi=150)
    plt.close()
    print("[TREINO] Gráfico salvo: outputs/predicao_vs_real.png")


def _salvar_grafico_loss(historico, diretorio: str):
    """Gera e salva gráfico de perda durante o treinamento."""
    os.makedirs("outputs", exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(historico.history["loss"], label="Loss Treino", color="#065A82")
    plt.plot(historico.history["val_loss"], label="Loss Validação", color="#F96167")
    plt.title("GlicemIA — Curva de Aprendizado", fontsize=14, fontweight="bold")
    plt.xlabel("Época")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/loss_treinamento.png", dpi=150)
    plt.close()
    print("[TREINO] Gráfico salvo: outputs/loss_treinamento.png")


if __name__ == "__main__":
    CAMINHO_DADOS = "data/dataset/Shanghai_T1DM/1001_0_20210730.xlsx"

    try:
        metricas = treinar(caminho_dados=CAMINHO_DADOS)
        print("\nResultado final:")
        print(f"  MAE  : {metricas['mae_mgdl']:.2f} mg/dL")
        print(f"  RMSE : {metricas['rmse_mgdl']:.2f} mg/dL")
    except Exception as e:
        print(f"\n❌ Erro durante o treinamento: {e}")
