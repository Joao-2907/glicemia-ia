# SPEC: Modelo de Predição LSTM

**Feature:** `modelo-lstm`  
**Status:** ✅ Implementado  
**Arquivo de implementação:** `src/model/treinar.py`, `src/model/avaliar.py`

---

## 1. Contexto

Com os dados pré-processados (X, y) disponíveis, o sistema precisa treinar uma Rede Neural Recorrente do tipo LSTM (Long Short-Term Memory) capaz de aprender padrões temporais na série de glicemia e prever o valor futuro com antecedência suficiente para intervenção.

---

## 2. Objetivo

Treinar, salvar e avaliar um modelo LSTM que prediz o valor de glicemia do paciente em um horizonte de ~30 minutos, com erro médio aceitável para uso clínico de suporte.

---

## 3. Arquitetura do Modelo

```
Input Layer     → (batch, 60, 1)     # 60 leituras CGM de 5 em 5 min = 5h
LSTM Layer 1    → 64 units, return_sequences=True
Dropout         → 0.2
LSTM Layer 2    → 32 units
Dropout         → 0.2
Dense Layer     → 1 unit (valor predito normalizado)
```

**Motivo da LSTM:** Captura dependências de longo prazo em séries temporais — crucial para padrões glicêmicos que dependem de refeições, insulina e atividade física horas antes.

---

## 4. Requisitos Funcionais

| ID | Requisito | Critério de Aceitação |
|----|-----------|----------------------|
| RF01 | Receber X e y pré-processados e treinar modelo | Modelo converge em ≤ 50 épocas |
| RF02 | Dividir dados em treino (80%) e validação (20%) | Split temporal, sem embaralhamento |
| RF03 | Salvar modelo treinado em arquivo `.keras` | Arquivo gerado em `models/` |
| RF04 | Salvar histórico de treino | Arquivo `.json` com loss por época |
| RF05 | Avaliar com MAE e RMSE em escala real (mg/dL) | Métricas calculadas após desnormalização |
| RF06 | Gerar gráfico de predição vs real | Imagem `.png` salva em `outputs/` |

---

## 5. Requisitos Não Funcionais

| ID | Requisito |
|----|-----------|
| RNF01 | Treino deve completar em tempo razoável (< 10 min em CPU básica) |
| RNF02 | Modelo salvo deve ser reutilizável sem re-treino |
| RNF03 | Métricas devem ser exibidas em mg/dL (escala compreensível clinicamente) |

---

## 6. Parâmetros de Treinamento

| Parâmetro | Valor | Motivo |
|-----------|-------|--------|
| `epochs` | 50 | Balanço entre qualidade e tempo de treino |
| `batch_size` | 32 | Padrão para séries temporais |
| `optimizer` | Adam | Convergência rápida e estável |
| `loss` | MSE | Penaliza erros grandes (segurança clínica) |
| `early_stopping` | patience=10 | Evita overfitting |

---

## 7. Critério de Qualidade

| Métrica | Alvo Aceitável | Justificativa |
|---------|---------------|---------------|
| MAE | < 15 mg/dL | Limiar clínico para monitoramento de suporte |
| RMSE | < 20 mg/dL | Erros maiores = alertas imprecisos |

> Estes valores são referência acadêmica; não substituem validação clínica profissional.

---

## 8. Diagrama de Fluxo de Treino

```
[X, y, scaler]
      ↓
[Split temporal 80/20]
      ↓
[Reshape: (n, 60) → (n, 60, 1)]
      ↓
[Definição do modelo LSTM]
      ↓
[Treinamento com EarlyStopping]
      ↓
[Avaliação no conjunto de validação]
      ↓
[Desnormalização das predições]
      ↓
[Cálculo MAE / RMSE em mg/dL]
      ↓
[Salvamento do modelo + gráficos]
```

---

## 9. Dependências

```
tensorflow>=2.13
keras>=2.13
numpy>=1.24
scikit-learn>=1.3
matplotlib>=3.7
```

---

## 10. Decisões de Design

- **Por que LSTM e não GRU?** LSTM tem histórico mais sólido em benchmarks de predição glicêmica na literatura. GRU pode ser avaliada em versão futura.
- **Por que split temporal sem shuffle?** Embaralhar dados temporais cria data leakage — o modelo "veria o futuro" durante treino.
- **Por que salvar em `.keras`?** Formato recomendado pelo TensorFlow 2.x, mais portátil que `.h5`.
