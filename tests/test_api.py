"""
GlicemIA — Testes Unitários: API
=================================
Testa os endpoints da API FastAPI.

Executar: python -m pytest tests/ -v
"""

import sys
import numpy as np
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.main import app, _classificar_risco, TAMANHO_JANELA

client = TestClient(app)


class TestHealthEndpoint:

    def test_health_retorna_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_retorna_status_ok(self):
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_health_tem_campos_obrigatorios(self):
        data = client.get("/health").json()
        assert "model_loaded" in data
        assert "version" in data
        assert "tensorflow_disponivel" in data


class TestPredictEndpoint:

    def _leituras_validas(self, inicio=160, fim=75):
        """Gera 60 leituras decrescentes para simular tendência hipoglicêmica."""
        return list(np.linspace(inicio, fim, TAMANHO_JANELA).round(1))

    def test_predict_retorna_200_com_dados_validos(self):
        payload = {"leituras_cgm": self._leituras_validas()}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200

    def test_predict_campos_obrigatorios_na_resposta(self):
        payload = {"leituras_cgm": self._leituras_validas()}
        data = client.post("/predict", json=payload).json()
        assert "predicao_mgdl" in data
        assert "classificacao_risco" in data
        assert "mensagem" in data
        assert "timestamp_predicao" in data
        assert "horizonte_minutos" in data

    def test_predict_horizonte_e_30_minutos(self):
        payload = {"leituras_cgm": self._leituras_validas()}
        data = client.post("/predict", json=payload).json()
        assert data["horizonte_minutos"] == 30

    def test_predict_rejeita_menos_de_60_leituras(self):
        payload = {"leituras_cgm": [100.0] * 45}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_rejeita_mais_de_60_leituras(self):
        payload = {"leituras_cgm": [100.0] * 80}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_rejeita_valores_fora_da_faixa(self):
        leituras = [100.0] * 59 + [700.0]  # 700 mg/dL é inválido
        payload = {"leituras_cgm": leituras}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_aceita_paciente_id_opcional(self):
        payload = {
            "leituras_cgm": self._leituras_validas(),
            "paciente_id": "teste-123"
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200

    def test_predict_classificacao_valida(self):
        payload = {"leituras_cgm": self._leituras_validas()}
        data = client.post("/predict", json=payload).json()
        classificacoes_validas = {"HIPOGLICEMIA", "ALERTA", "NORMAL", "HIPERGLICEMIA"}
        assert data["classificacao_risco"] in classificacoes_validas


class TestClassificarRisco:

    def test_hipoglicemia_abaixo_de_70(self):
        classificacao, _ = _classificar_risco(65.0)
        assert classificacao == "HIPOGLICEMIA"

    def test_alerta_entre_70_e_89(self):
        classificacao, _ = _classificar_risco(80.0)
        assert classificacao == "ALERTA"

    def test_normal_entre_90_e_180(self):
        classificacao, _ = _classificar_risco(120.0)
        assert classificacao == "NORMAL"

    def test_hiperglicemia_acima_de_180(self):
        classificacao, _ = _classificar_risco(220.0)
        assert classificacao == "HIPERGLICEMIA"

    def test_fronteira_hipoglicemia(self):
        classificacao, _ = _classificar_risco(70.0)
        assert classificacao == "ALERTA"  # 70 está na faixa de alerta (70-89)

    def test_fronteira_normal(self):
        classificacao, _ = _classificar_risco(180.0)
        assert classificacao == "NORMAL"

    def test_mensagem_nao_vazia(self):
        for valor in [60, 80, 120, 200]:
            _, mensagem = _classificar_risco(valor)
            assert len(mensagem) > 0


class TestRootEndpoint:

    def test_raiz_retorna_200(self):
        response = client.get("/")
        assert response.status_code == 200

    def test_raiz_menciona_docs(self):
        data = client.get("/").json()
        assert "docs" in data
