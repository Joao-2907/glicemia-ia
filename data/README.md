# Dados — GlicemIA

## Estrutura

```
data/
├── dataset/
│   ├── Shanghai_T1DM/    ← Arquivos .xlsx de pacientes T1DM
│   └── Shanghai_T2DM/    ← Arquivos .xlsx de pacientes T2DM
└── processed/            ← Dados normalizados gerados pelo pré-processamento
```

## Dataset Utilizado

**Shanghai T1DM & T2DM Benchmark Dataset**

- Fonte: Hospital Clínico de Xangai
- Formato: Excel (.xlsx), uma linha por leitura CGM
- Coluna principal: `CGM (mg / dl)` — leituras a cada ~5 minutos
- Pacientes T1DM: ~12 pacientes
- Pacientes T2DM: ~100 pacientes

## Como Obter o Dataset

O dataset não é versionado neste repositório por razões de tamanho e privacidade.

1. Acesse a publicação original:  
   **"Shanghai T1DM and T2DM Benchmark Dataset for Blood Glucose Prediction"**

2. Coloque os arquivos no diretório correspondente:
   ```
   data/dataset/Shanghai_T1DM/1001_0_20210730.xlsx
   data/dataset/Shanghai_T1DM/1002_0_20210730.xlsx
   ...
   ```

3. Execute o pré-processamento:
   ```bash
   python src/preprocessing/preprocessamento.py
   ```

## Estrutura do Arquivo Excel

| Coluna | Descrição |
|--------|-----------|
| `Time` | Timestamp da leitura |
| `CGM (mg / dl)` | Glicemia capilar em mg/dL |
| `Finger` | Leitura capilar de dedo (quando disponível) |
| `Basal` | Taxa basal de insulina |
| `Bolus` | Dose de bolus de insulina |
| `Meal` | Registro de refeição (g de carboidratos) |

## Modo Demonstração

Se o dataset não estiver disponível, o sistema funciona em **modo demonstração** usando dados sintéticos gerados pela função `gerar_dados_demo()` em `src/preprocessing/preprocessamento.py`.
