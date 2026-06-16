"""
GlicemIA — Módulo de Pré-processamento de Dados CGM
====================================================
Responsável por carregar os arquivos do dataset Shanghai T1DM/T2DM,
limpar e normalizar os dados de glicemia, e criar as janelas deslizantes
necessárias para o treinamento do modelo LSTM.

Spec de referência: docs/specs/spec-preprocessamento.md
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path


def criar_janelas_deslizantes(
    dados: np.ndarray,
    tamanho_janela: int,
    horizonte_predicao: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cria janelas deslizantes para séries temporais.

    Parâmetros
    ----------
    dados : np.ndarray
        Array 2D normalizado (n_amostras, 1)
    tamanho_janela : int
        Número de leituras passadas usadas como entrada (ex: 60 = ~5h de histórico)
    horizonte_predicao : int
        Número de passos à frente para prever (ex: 60 = ~30 min com leituras de 5/5min)

    Retorna
    -------
    X : np.ndarray — shape (n_janelas, tamanho_janela)
    y : np.ndarray — shape (n_janelas,)
    """
    X, y = [], []
    n = len(dados)
    minimo_necessario = tamanho_janela + horizonte_predicao

    if n < minimo_necessario:
        raise ValueError(
            f"Dados insuficientes para criar janelas. "
            f"Necessário: {minimo_necessario} amostras. Disponível: {n}."
        )

    for i in range(n - tamanho_janela - horizonte_predicao):
        janela_x = dados[i : i + tamanho_janela, 0]
        alvo_y = dados[i + tamanho_janela + horizonte_predicao - 1, 0]
        X.append(janela_x)
        y.append(alvo_y)

    return np.array(X), np.array(y)


def preparar_dados(
    caminho_arquivo: str | Path,
    tamanho_janela: int = 60,
    horizonte: int = 60
) -> tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Pipeline completo de pré-processamento de um arquivo CGM.

    Etapas:
    1. Leitura do arquivo Excel
    2. Extração da coluna CGM
    3. Remoção de valores nulos
    4. Normalização MinMaxScaler [0, 1]
    5. Criação das janelas deslizantes

    Parâmetros
    ----------
    caminho_arquivo : str | Path
        Caminho para o arquivo .xlsx ou .xls do dataset
    tamanho_janela : int
        Leituras passadas por janela (padrão: 60 ≈ 5h)
    horizonte : int
        Passos à frente para prever (padrão: 60 ≈ 30min)

    Retorna
    -------
    X : np.ndarray — entradas para o modelo
    y : np.ndarray — saídas alvo
    scaler : MinMaxScaler — para desnormalizar predições em mg/dL
    """
    caminho = Path(caminho_arquivo)
    COLUNA_CGM = "CGM (mg / dl)"

    # 1. Verificar existência do arquivo
    if not caminho.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {caminho}\n"
            f"Verifique se o dataset foi colocado em data/dataset/"
        )

    print(f"[PREPROCESSAMENTO] Lendo arquivo: {caminho.name}")
    df = pd.read_excel(caminho)

    # 2. Verificar existência da coluna CGM
    if COLUNA_CGM not in df.columns:
        colunas_disponiveis = list(df.columns)
        raise KeyError(
            f"Coluna '{COLUNA_CGM}' não encontrada.\n"
            f"Colunas disponíveis: {colunas_disponiveis}"
        )

    # 3. Extrair e limpar dados de glicemia
    dados_glicemia = df[COLUNA_CGM].dropna().values.reshape(-1, 1)

    if len(dados_glicemia) == 0:
        raise ValueError(
            f"A coluna '{COLUNA_CGM}' está completamente vazia após remoção de NaN."
        )

    print(f"[PREPROCESSAMENTO] Leituras CGM válidas: {len(dados_glicemia)}")

    # 4. Normalização para o intervalo [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    dados_normalizados = scaler.fit_transform(dados_glicemia)

    # 5. Criar janelas deslizantes
    X, y = criar_janelas_deslizantes(
        dados=dados_normalizados,
        tamanho_janela=tamanho_janela,
        horizonte_predicao=horizonte
    )

    print(f"[PREPROCESSAMENTO] ✅ Dados processados com sucesso")
    print(f"  Janela de entrada  : {tamanho_janela} leituras (~{tamanho_janela * 5 // 60}h de histórico)")
    print(f"  Horizonte          : {horizonte} passos (~{horizonte * 5} min à frente)")
    print(f"  Total de amostras  : {len(X)}")
    print(f"  Shape X (entradas) : {X.shape}")
    print(f"  Shape y (alvos)    : {y.shape}")

    return X, y, scaler


def gerar_dados_demo() -> np.ndarray:
    """
    Gera 60 leituras CGM sintéticas para demonstração da API.
    Simula uma sequência com tendência de queda em direção à hipoglicemia.

    Retorna
    -------
    np.ndarray : 60 valores de glicemia em mg/dL
    """
    np.random.seed(42)

    # Fase 1: glicemia pós-prandial elevada (leituras 0-20)
    fase1 = np.linspace(160, 130, 20) + np.random.normal(0, 3, 20)

    # Fase 2: queda gradual para faixa normal (leituras 20-40)
    fase2 = np.linspace(130, 100, 20) + np.random.normal(0, 2, 20)

    # Fase 3: queda contínua — tendência hipoglicêmica (leituras 40-60)
    fase3 = np.linspace(100, 75, 20) + np.random.normal(0, 2, 20)

    return np.concatenate([fase1, fase2, fase3])


# ==========================================
# ÁREA DE TESTE
# ==========================================
if __name__ == "__main__":
    import sys

    # Teste com arquivo real (se disponível)
    CAMINHO_TESTE = "data/dataset/Shanghai_T1DM/1001_0_20210730.xlsx"

    print("=" * 50)
    print("GlicemIA — Teste de Pré-processamento")
    print("=" * 50)

    try:
        X_treino, y_treino, meu_scaler = preparar_dados(CAMINHO_TESTE)
        print("\n✅ Pré-processamento concluído. Pronto para alimentar a LSTM.\n")
    except FileNotFoundError as e:
        print(f"\n⚠️  Arquivo de teste não encontrado: {e}")
        print("   Gerando dados de demonstração sintéticos...")
        dados_demo = gerar_dados_demo()
        print(f"   Dados demo gerados: {len(dados_demo)} leituras")
        print(f"   Faixa: {dados_demo.min():.1f} – {dados_demo.max():.1f} mg/dL")
    except Exception as e:
        print(f"\n❌ Erro inesperado: {e}")
        sys.exit(1)
