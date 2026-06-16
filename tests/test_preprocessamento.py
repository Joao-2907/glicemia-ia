"""
GlicemIA — Testes Unitários: Pré-processamento
===============================================
Testa as funções de pré-processamento para garantir
que a pipeline está funcionando corretamente.

Executar: python -m pytest tests/ -v
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Garante que o pacote src está no path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.preprocessamento import (
    criar_janelas_deslizantes,
    gerar_dados_demo,
)


class TestCriarJanelasDeslizantes:

    def test_shape_correto(self):
        """X deve ter shape (n_amostras, tamanho_janela) e y (n_amostras,)."""
        dados = np.random.rand(200, 1)
        X, y = criar_janelas_deslizantes(dados, tamanho_janela=60, horizonte_predicao=60)

        n_esperado = 200 - 60 - 60
        assert X.shape == (n_esperado, 60), f"Shape X incorreto: {X.shape}"
        assert y.shape == (n_esperado,), f"Shape y incorreto: {y.shape}"

    def test_dados_insuficientes_lanca_erro(self):
        """Deve lançar ValueError quando não há dados suficientes para a janela."""
        dados_curtos = np.random.rand(50, 1)
        with pytest.raises(ValueError, match="Dados insuficientes"):
            criar_janelas_deslizantes(dados_curtos, tamanho_janela=60, horizonte_predicao=60)

    def test_alvo_e_passo_correto_a_frente(self):
        """y[0] deve corresponder ao valor em posição tamanho_janela + horizonte - 1."""
        dados = np.arange(200).reshape(-1, 1).astype(float)
        tamanho = 10
        horizonte = 5

        X, y = criar_janelas_deslizantes(dados, tamanho_janela=tamanho, horizonte_predicao=horizonte)

        # O primeiro alvo deve ser o valor na posição 10 + 5 - 1 = 14
        assert y[0] == dados[tamanho + horizonte - 1, 0], (
            f"Alvo esperado: {dados[tamanho + horizonte - 1, 0]}, obtido: {y[0]}"
        )

    def test_valores_dentro_do_range_normalizado(self):
        """Com dados normalizados [0,1], X e y devem estar em [0,1]."""
        dados = np.random.uniform(0, 1, size=(200, 1))
        X, y = criar_janelas_deslizantes(dados, tamanho_janela=30, horizonte_predicao=30)

        assert X.min() >= 0.0 and X.max() <= 1.0, "X fora do range [0,1]"
        assert y.min() >= 0.0 and y.max() <= 1.0, "y fora do range [0,1]"

    def test_janelas_consecutivas_se_sobrepoem_corretamente(self):
        """A segunda janela deve ser a primeira deslocada por 1 posição."""
        dados = np.arange(100, dtype=float).reshape(-1, 1)
        X, y = criar_janelas_deslizantes(dados, tamanho_janela=10, horizonte_predicao=5)

        # X[0] = [0..9], X[1] = [1..10]
        np.testing.assert_array_equal(X[0], np.arange(10, dtype=float))
        np.testing.assert_array_equal(X[1], np.arange(1, 11, dtype=float))


class TestGerarDadosDemo:

    def test_retorna_60_leituras(self):
        """Deve retornar exatamente 60 valores."""
        dados = gerar_dados_demo()
        assert len(dados) == 60, f"Esperado 60, obtido: {len(dados)}"

    def test_valores_em_faixa_fisiologica(self):
        """Valores devem estar em faixa plausível de glicemia humana."""
        dados = gerar_dados_demo()
        assert dados.min() >= 40, f"Valor abaixo do mínimo fisiológico: {dados.min()}"
        assert dados.max() <= 400, f"Valor acima do máximo fisiológico: {dados.max()}"

    def test_tendencia_decrescente(self):
        """Os dados de demo devem ter tendência de queda (simulando risco hipo)."""
        dados = gerar_dados_demo()
        media_inicio = np.mean(dados[:20])
        media_fim = np.mean(dados[-20:])
        assert media_fim < media_inicio, (
            f"Esperava tendência de queda. Início: {media_inicio:.1f}, Fim: {media_fim:.1f}"
        )

    def test_determinismo_com_seed(self):
        """Com a mesma seed, deve retornar sempre os mesmos dados."""
        dados1 = gerar_dados_demo()
        dados2 = gerar_dados_demo()
        np.testing.assert_array_equal(dados1, dados2)
