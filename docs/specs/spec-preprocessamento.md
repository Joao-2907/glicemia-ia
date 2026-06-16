# SPEC: Pipeline de Pré-processamento CGM

**Feature:** `preprocessamento`  
**Status:** ✅ Implementado  
**Arquivo de implementação:** `src/preprocessing/preprocessamento.py`

---

## 1. Contexto

O sistema recebe arquivos Excel do dataset Shanghai T1DM/T2DM contendo leituras de Monitoramento Contínuo de Glicose (CGM) em mg/dL. Antes de alimentar o modelo LSTM, os dados precisam ser limpos, normalizados e estruturados no formato de janela deslizante.

---

## 2. Objetivo

Transformar arquivos brutos de CGM em matrizes de entrada (X) e vetores de saída (y) prontos para o treinamento da rede neural LSTM.

---

## 3. Requisitos Funcionais

| ID | Requisito | Critério de Aceitação |
|----|-----------|----------------------|
| RF01 | Ler arquivo Excel (.xlsx ou .xls) do dataset | Arquivo lido sem erros para todos os pacientes do dataset |
| RF02 | Extrair coluna `CGM (mg / dl)` | Coluna correta extraída como array NumPy |
| RF03 | Remover valores nulos (NaN) | Nenhum NaN presente na matriz de saída |
| RF04 | Normalizar dados com MinMaxScaler no intervalo [0, 1] | Todos os valores em [0.0, 1.0] |
| RF05 | Criar janelas deslizantes de tamanho configurável | X.shape = (n_amostras, tamanho_janela) |
| RF06 | Definir horizonte de predição configurável | y representa o valor `horizonte` passos à frente |
| RF07 | Retornar o scaler treinado junto com X e y | Scaler disponível para desnormalização posterior |

---

## 4. Requisitos Não Funcionais

| ID | Requisito |
|----|-----------|
| RNF01 | Compatível com Python 3.10+ |
| RNF02 | Processar 1 arquivo em menos de 5 segundos |
| RNF03 | Logs claros no console sobre o progresso |

---

## 5. Parâmetros de Configuração

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `tamanho_janela` | `60` | Número de leituras passadas usadas como entrada (60 leituras = ~5h de dados a cada 5min) |
| `horizonte` | `60` | Número de passos à frente para prever (60 passos = ~30min) |

---

## 6. Diagrama de Fluxo

```
[Arquivo .xlsx]
      ↓
[Leitura com pandas.read_excel()]
      ↓
[Extração coluna 'CGM (mg / dl)']
      ↓
[Remoção de NaN]
      ↓
[Normalização MinMaxScaler [0,1]]
      ↓
[Criação de janelas deslizantes]
      ↓
[Saída: X, y, scaler]
```

---

## 7. Exemplo de Uso

```python
from src.preprocessing.preprocessamento import preparar_dados

X, y, scaler = preparar_dados(
    caminho_arquivo='data/dataset/Shanghai_T1DM/1001_0_20210730.xlsx',
    tamanho_janela=60,
    horizonte=60
)

print(f"X shape: {X.shape}")   # (n_amostras, 60)
print(f"y shape: {y.shape}")   # (n_amostras,)
```

---

## 8. Casos de Borda / Tratamento de Erros

| Caso | Comportamento Esperado |
|------|----------------------|
| Arquivo não encontrado | Lança `FileNotFoundError` com caminho informado |
| Coluna CGM ausente | Lança `KeyError` com nome da coluna faltante |
| Dados insuficientes para a janela | Lança `ValueError` com mensagem explicativa |
| Valores completamente nulos | Lança `ValueError` com aviso |

---

## 9. Dependências

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
openpyxl>=3.1
xlrd>=2.0
```

---

## 10. Decisões de Design

- **Motivo do MinMaxScaler:** Redes neurais convergem melhor com dados normalizados; MinMaxScaler preserva a distribuição original.
- **Motivo do horizonte de 60 passos (≈30 min):** Leituras CGM são feitas a cada ~5 minutos; 60 passos = 5 horas de histórico; horizonte de 60 = predição ~30 minutos à frente, tempo suficiente para intervenção do paciente.
- **Motivo de retornar o scaler:** Necessário para reverter a normalização e interpretar as predições em mg/dL real.
