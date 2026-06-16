# GlicemIA — Makefile
# Atalhos para os principais comandos do projeto

.PHONY: install test demo api preprocess train clean help

help:
	@echo ""
	@echo "GlicemIA — Comandos disponíveis:"
	@echo "──────────────────────────────────────────────────"
	@echo "  make install     Instalar dependências"
	@echo "  make test        Executar testes unitários"
	@echo "  make demo        Rodar demonstração completa"
	@echo "  make api         Iniciar API de predição"
	@echo "  make preprocess  Pré-processar dataset"
	@echo "  make train       Treinar modelo LSTM"
	@echo "  make clean       Limpar arquivos gerados"
	@echo "──────────────────────────────────────────────────"

install:
	pip install -r requirements.txt

test:
	python -m pytest tests/ -v --tb=short

demo:
	python demo.py

api:
	uvicorn src.api.main:app --reload --port 8000

preprocess:
	python src/preprocessing/preprocessamento.py

train:
	python src/model/treinar.py

evaluate:
	python src/model/avaliar.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf outputs/*.png .pytest_cache .coverage htmlcov
	@echo "Arquivos temporários removidos."
