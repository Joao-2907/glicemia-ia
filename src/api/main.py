"""
GlicemIA — API REST de Predição Glicêmica
==========================================
Expõe o modelo LSTM treinado via endpoints HTTP.

Spec de referência: docs/specs/spec-api-predicao.md

Endpoints:
  GET  /health   → status da API e modelo
  POST /predict  → predição de glicemia futura
  GET  /docs     → Swagger UI (gerado automaticamente pelo FastAPI)
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import numpy as np

# Importação condicional do TensorFlow
try:
    import tensorflow as tf
    TENSORFLOW_DISPONIVEL = True
except ImportError:
    TENSORFLOW_DISPONIVEL = False

from sklearn.preprocessing import MinMaxScaler

# ─── Constantes Clínicas ───────────────────────────────────────────────
TAMANHO_JANELA = 60       # leituras de entrada (5min × 60 = 5h de histórico)
HORIZONTE_MIN = 30        # minutos à frente preditos
LIMIAR_HIPO = 70          # mg/dL — hipoglicemia
LIMIAR_ALERTA = 90        # mg/dL — zona de alerta
LIMIAR_HIPER = 180        # mg/dL — hiperglicemia
CAMINHO_MODELO = "models/modelo_glicemia.keras"

# ─── Estado global (modelo + scaler) ──────────────────────────────────
modelo = None
scaler_global = None


def _classificar_risco(valor_mgdl: float) -> tuple[str, str]:
    """
    Classifica o risco clínico baseado no valor de glicemia.

    Retorna
    -------
    (classificacao, mensagem) : tuple[str, str]
    """
    if valor_mgdl < LIMIAR_HIPO:
        return (
            "HIPOGLICEMIA",
            f"⚠️ ATENÇÃO: Glicemia prevista ({valor_mgdl:.0f} mg/dL) está abaixo de "
            f"{LIMIAR_HIPO} mg/dL. Risco de crise hipoglicêmica em ~{HORIZONTE_MIN} min. "
            "Consuma carboidratos de ação rápida agora."
        )
    elif valor_mgdl < LIMIAR_ALERTA:
        return (
            "ALERTA",
            f"⚡ Glicemia prevista ({valor_mgdl:.0f} mg/dL) está em zona de alerta. "
            f"Monitore de perto nos próximos {HORIZONTE_MIN} minutos."
        )
    elif valor_mgdl <= LIMIAR_HIPER:
        return (
            "NORMAL",
            f"✅ Glicemia prevista ({valor_mgdl:.0f} mg/dL) está dentro da faixa normal. "
            "Continue monitorando regularmente."
        )
    else:
        return (
            "HIPERGLICEMIA",
            f"🔴 Glicemia prevista ({valor_mgdl:.0f} mg/dL) está acima de "
            f"{LIMIAR_HIPER} mg/dL. Verifique com seu médico ou enfermeiro."
        )


def _criar_scaler_demo() -> MinMaxScaler:
    """Cria um scaler de demonstração calibrado para faixas reais de glicemia."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    # Calibra com faixa real típica: 40–400 mg/dL
    scaler.fit(np.array([[40.0], [400.0]]))
    return scaler


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerencia o ciclo de vida da aplicação (startup/shutdown)."""
    global modelo, scaler_global
    # ── Startup ──
    if TENSORFLOW_DISPONIVEL and Path(CAMINHO_MODELO).exists():
        try:
            modelo = tf.keras.models.load_model(CAMINHO_MODELO)
            print(f"[API] ✅ Modelo carregado: {CAMINHO_MODELO}")
        except Exception as e:
            print(f"[API] ⚠️  Erro ao carregar modelo: {e}")
            modelo = None
    else:
        print("[API] ℹ️  Modelo não encontrado — modo demonstração ativo.")
    scaler_global = _criar_scaler_demo()
    yield
    # ── Shutdown ──
    print("[API] Encerrando aplicação.")


# ─── Inicialização da App ──────────────────────────────────────────────
app = FastAPI(
    title="GlicemIA API",
    description=(
        "API de predição glicêmica baseada em LSTM. "
        "Prevê o nível de glicose em ~30 minutos a partir do histórico CGM."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Schemas Pydantic ──────────────────────────────────────────────────

class EntradaPredicao(BaseModel):
    leituras_cgm: list[float]
    paciente_id: Optional[str] = None

    @field_validator("leituras_cgm")
    @classmethod
    def validar_leituras(cls, v):
        if len(v) != TAMANHO_JANELA:
            raise ValueError(
                f"São necessárias exatamente {TAMANHO_JANELA} leituras CGM. "
                f"Recebidas: {len(v)}."
            )
        if any(x < 0 or x > 600 for x in v):
            raise ValueError(
                "Todos os valores CGM devem estar entre 0 e 600 mg/dL."
            )
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "leituras_cgm": [160, 158, 155, 150, 145, 140, 135, 130, 125, 120,
                                  118, 115, 112, 110, 108, 105, 103, 100, 98, 96,
                                  95, 93, 91, 90, 88, 87, 86, 85, 84, 83,
                                  82, 81, 80, 79, 78, 77, 76, 75, 74, 73,
                                  72, 71, 70, 69, 68, 67, 66, 65, 64, 63,
                                  62, 61, 60, 59, 58, 57, 56, 55, 54, 53],
                "paciente_id": "paciente-1001"
            }
        }
    }


class RespostaPredicao(BaseModel):
    predicao_mgdl: float
    classificacao_risco: str
    mensagem: str
    timestamp_predicao: str
    horizonte_minutos: int
    modo_demo: bool


class RespostaHealth(BaseModel):
    status: str
    model_loaded: bool
    tensorflow_disponivel: bool
    version: str


# ─── Endpoints ────────────────────────────────────────────────────────

@app.get("/health", response_model=RespostaHealth, tags=["Sistema"])
async def health_check():
    """Verifica o status da API e se o modelo está carregado."""
    return RespostaHealth(
        status="ok",
        model_loaded=modelo is not None,
        tensorflow_disponivel=TENSORFLOW_DISPONIVEL,
        version="1.0.0"
    )


@app.post("/predict", response_model=RespostaPredicao, tags=["Predição"])
async def prever_glicemia(entrada: EntradaPredicao):
    """
    Realiza predição da glicemia futura.

    Recebe as últimas 60 leituras CGM (a cada ~5 min = 5h de histórico)
    e retorna a glicemia prevista para ~30 minutos à frente, com classificação de risco.
    """
    global modelo, scaler_global

    leituras = np.array(entrada.leituras_cgm, dtype=np.float32)
    modo_demo = False

    if modelo is not None and TENSORFLOW_DISPONIVEL:
        # ── Predição Real com LSTM ──
        leituras_norm = scaler_global.transform(leituras.reshape(-1, 1))
        X = leituras_norm.reshape(1, TAMANHO_JANELA, 1)

        predicao_norm = modelo.predict(X, verbose=0)[0][0]
        predicao_mgdl = float(
            scaler_global.inverse_transform([[predicao_norm]])[0][0]
        )
    else:
        # ── Modo Demonstração ──
        # Simula predição como tendência linear dos últimos 10 valores
        modo_demo = True
        ultimas = leituras[-10:]
        tendencia = np.polyfit(range(len(ultimas)), ultimas, deg=1)[0]
        predicao_mgdl = float(leituras[-1] + tendencia * (HORIZONTE_MIN // 5))
        # Garante valor dentro de faixa fisiológica
        predicao_mgdl = max(40.0, min(400.0, predicao_mgdl))

    classificacao, mensagem = _classificar_risco(predicao_mgdl)

    return RespostaPredicao(
        predicao_mgdl=round(predicao_mgdl, 1),
        classificacao_risco=classificacao,
        mensagem=mensagem,
        timestamp_predicao=datetime.now(timezone.utc).isoformat(),
        horizonte_minutos=HORIZONTE_MIN,
        modo_demo=modo_demo
    )


@app.get("/", tags=["Sistema"])
async def raiz():
    """Redireciona para a documentação."""
    return {
        "mensagem": "GlicemIA API v1.0.0 — Acesse /docs para a documentação interativa.",
        "docs": "/docs",
        "health": "/health"
    }
