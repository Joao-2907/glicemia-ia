# 🩺 GlicemIA — App de Predição Glicêmica com IA

> **Transformando o tratamento da diabetes: da reatividade à prevenção com Inteligência Artificial**

Desenvolvido por **João Victor Almeida Souza** e **Yasmim Elisa da Cruz Ferreira** — UNIFEG  
Disciplina: Empreendedorismo | Curso: [seu curso]

---

## 📌 Sobre o Projeto

O **GlicemIA** é um sistema de predição glicêmica baseado em Redes Neurais (LSTM) que analisa o histórico de monitoramento contínuo de glicose (CGM) do paciente para **prever crises hipoglicêmicas com 30 a 40 minutos de antecedência**.

O grande diferencial: **prever crises antes que aconteçam**, não apenas reagir a elas.

---

## 🎯 Problema Resolvido

| Problema | Impacto |
|----------|---------|
| Hipoglicemia inesperada | Risco de vida, medo constante |
| Cuidado reativo | Correções de emergência, sem prevenção |
| Falta de autonomia | Paciente dependente de medições manuais |

---

## 🤖 Como a IA Foi Utilizada

O projeto aplica os seguintes conceitos de IA vistos em sala de aula:

| Conceito | Aplicação no Projeto |
|----------|---------------------|
| **Redes Neurais (LSTM)** | Modelo principal de predição temporal de glicemia |
| **Janela Deslizante** | Técnica de criação de sequências para séries temporais |
| **Normalização (MinMaxScaler)** | Pré-processamento dos dados CGM para treino da rede |
| **Spec Driven Design** | Todas as features foram especificadas antes de implementadas |
| **Validação de Modelos** | Métricas MAE, RMSE e comparação baseline |

A IA foi utilizada também como **assistente de desenvolvimento** (via Cursor + Claude) para:
- Geração e revisão de código Python
- Refinamento das especificações de features (ver `/docs/specs/`)
- Revisão dos MDs de documentação de IA (ver `/docs/ai/`)

---

## 🏗️ Arquitetura do Sistema

```
glicemia-ia/
├── src/
│   ├── preprocessing/     # Limpeza e preparação dos dados CGM
│   ├── model/             # Definição, treino e avaliação do modelo LSTM
│   ├── api/               # API FastAPI para servir predições
│   └── utils/             # Utilitários (logging, config, métricas)
├── data/
│   ├── dataset/           # Datasets Shanghai T1DM e T2DM
│   └── processed/         # Dados normalizados prontos para treino
├── docs/
│   ├── ai/                # MDs de uso de IA (obrigatório pelo enunciado)
│   └── specs/             # Especificações Spec Driven Design
├── tests/                 # Testes unitários
├── notebooks/             # Jupyter Notebooks de exploração
└── .github/workflows/     # CI/CD (GitHub Actions)
```

---

## 🚀 Como Executar

### Pré-requisitos

- Python 3.10+
- pip

### Instalação

```bash
git clone https://github.com/SEU_USUARIO/glicemia-ia.git
cd glicemia-ia
pip install -r requirements.txt
```

### 1. Pré-processamento dos Dados

```bash
python src/preprocessing/preprocessamento.py
```

### 2. Treinar o Modelo

```bash
python src/model/treinar.py
```

### 3. Avaliar o Modelo

```bash
python src/model/avaliar.py
```

### 4. Executar a API

```bash
uvicorn src.api.main:app --reload
```

Acesse: `http://localhost:8000/docs` para a documentação interativa.

---

## 📊 Dataset Utilizado

**Shanghai T1DM & T2DM Benchmark Dataset**  
- Pacientes com Diabetes Tipo 1 (T1DM) e Tipo 2 (T2DM)
- Monitoramento Contínuo de Glicose (CGM) em mg/dL
- Dados reais de hospital clínico de Xangai

> Os arquivos de dataset não são versionados por tamanho. Baixe conforme instruções em `data/README.md`.

---

## 📁 Documentação de IA (Obrigatório)

Todos os Markdowns de uso de IA estão em `/docs/ai/`:

| Arquivo | Conteúdo |
|---------|----------|
| [`uso-ia-desenvolvimento.md`](docs/ai/uso-ia-desenvolvimento.md) | Como a IA foi usada no desenvolvimento |
| [`prompts-utilizados.md`](docs/ai/prompts-utilizados.md) | Prompts reais utilizados com o assistente IA |
| [`aprendizados.md`](docs/ai/aprendizados.md) | Principais aprendizados com IA no projeto |

---

## 📋 Spec Driven Design

Todas as features foram especificadas antes de implementadas. Ver `/docs/specs/`:

| Arquivo | Feature |
|---------|---------|
| [`spec-preprocessamento.md`](docs/specs/spec-preprocessamento.md) | Pipeline de pré-processamento CGM |
| [`spec-modelo-lstm.md`](docs/specs/spec-modelo-lstm.md) | Modelo de predição LSTM |
| [`spec-api-predicao.md`](docs/specs/spec-api-predicao.md) | API REST de predição |

---

## 👥 Equipe

| Nome | Função |
|------|--------|
| João Victor Almeida Souza | Desenvolvimento, Modelo IA |
| Yasmim Elisa da Cruz Ferreira | Pesquisa, Documentação, Testes |

---

## 📄 Licença

MIT License — livre para uso acadêmico.
