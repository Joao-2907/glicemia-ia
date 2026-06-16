# Prompts Utilizados com IA — GlicemIA

**Projeto:** App de Predição Glicêmica  
**Ferramentas:** Cursor (Claude Sonnet), ChatGPT

> Este documento registra os prompts reais (ou aproximados) utilizados durante o desenvolvimento do projeto, com o contexto de uso e o resultado obtido.

---

## Prompt 1 — Spec Driven Design do Pré-processamento

**Contexto:** Início do projeto; queríamos especificar a feature antes de implementar.

**Prompt:**
```
Você é um arquiteto de software sênior. Crie uma especificação técnica no formato 
Spec Driven Design para o seguinte módulo:

Módulo: Pipeline de pré-processamento de dados CGM (Continuous Glucose Monitor)
Objetivo: Ler arquivos Excel do dataset Shanghai T1DM/T2DM, extrair a coluna 
'CGM (mg / dl)', normalizar com MinMaxScaler e criar janelas deslizantes de 
tamanho 60 (leituras passadas) para prever 1 valor futuro após 60 passos.

A spec deve incluir: requisitos funcionais com critérios de aceitação, parâmetros 
de configuração com justificativas, diagrama de fluxo em texto, casos de borda 
e decisões de design.
```

**Resultado:** Spec completa gerada → base para `/docs/specs/spec-preprocessamento.md`  
**Ajuste feito:** Adicionamos a seção de dependências pip manualmente.

---

## Prompt 2 — Revisão de Código com Tratamento de Erros

**Contexto:** Tínhamos o código base de `preprocessamento.py`; queríamos melhorá-lo.

**Prompt:**
```
Revise este código Python de pré-processamento de dados CGM para diabetes e:
1. Adicione tratamento de erros adequado (FileNotFoundError, KeyError, ValueError)
2. Melhore os logs para deixar o output mais informativo
3. Sugira qualquer melhoria de robustez que você identificar

[código colado aqui]
```

**Resultado:** Código com blocos try/except, mensagens de erro descritivas e validação do shape dos dados.  
**Ajuste feito:** Mantivemos a estrutura original; apenas incorporamos as melhorias de erro.

---

## Prompt 3 — Arquitetura do Modelo LSTM

**Contexto:** Definição do modelo de IA para predição glicêmica.

**Prompt:**
```
Estou desenvolvendo um modelo de predição glicêmica para diabetes usando LSTM.
Os dados de entrada são 60 leituras consecutivas de CGM (a cada 5 minutos = 5h 
de histórico), normalizadas entre 0 e 1. O objetivo é prever 1 valor 30 minutos 
à frente.

Sugira uma arquitetura LSTM adequada para este problema, justificando cada camada,
os hiperparâmetros de treino (epochs, batch_size, optimizer, loss) e como evitar
overfitting com um dataset pequeno (~12 pacientes).
```

**Resultado:** Arquitetura com 2 LSTMs (64+32 units) + Dropout 0.2 + EarlyStopping.  
**Aprendizado:** A IA nos alertou sobre data leakage em splits sem respeitar ordem temporal.

---

## Prompt 4 — Comparação FastAPI vs Flask para ML Serving

**Contexto:** Escolha do framework para a API de predição.

**Prompt:**
```
Para um projeto acadêmico de ML serving que precisa expor um modelo LSTM de 
predição de glicemia via API REST, compare FastAPI e Flask considerando:
- Facilidade de uso para iniciantes
- Documentação automática
- Validação de dados de entrada
- Performance para inferência de modelos TensorFlow/Keras
- Facilidade de deploy
```

**Resultado:** Recomendação do FastAPI com justificativas claras; estrutura dos endpoints sugerida.

---

## Prompt 5 — Design da Resposta da API com Contexto Clínico

**Contexto:** Definição do formato de resposta do endpoint `/predict`.

**Prompt:**
```
A minha API de predição glicêmica retorna um valor numérico em mg/dL.
Preciso enriquecer essa resposta para um app mobile voltado a pacientes diabéticos.

Sugira um formato de resposta JSON que inclua:
- O valor predito
- Uma classificação de risco (usando faixas clínicas padrão de glicemia)
- Uma mensagem amigável para o paciente
- Informações de contexto temporal

Use as faixas clínicas padrão: hipoglicemia < 70, alerta 70-89, normal 90-180, 
hiperglicemia > 180.
```

**Resultado:** Schema JSON com `predicao_mgdl`, `classificacao_risco`, `mensagem`, `timestamp_predicao` e `horizonte_minutos`.

---

## Prompt 6 — Estrutura do README

**Contexto:** Documentação do projeto para entrega acadêmica.

**Prompt:**
```
Crie um README.md profissional para um projeto acadêmico de IA chamado "GlicemIA",
um app de predição de glicemia para diabetes usando LSTM.

O README deve incluir: descrição do problema, como a IA foi utilizada (tabela),
instruções de instalação e execução, estrutura de arquivos e links para 
documentação. Linguagem em português brasileiro.
```

**Resultado:** Estrutura completa do README → base para `/README.md`  
**Ajuste feito:** Adaptamos os nomes dos integrantes, dataset e caminhos de arquivo.

---

## Prompt 7 — Geração de Dados de Demonstração

**Contexto:** Necessidade de dados de teste sem expor o dataset real.

**Prompt:**
```
Crie uma função Python que gere dados sintéticos de CGM (glicemia contínua) 
para fins de demonstração e teste. 

Os dados devem:
- Simular padrões realistas de glicemia (variação pós-prandial, tendência noturna)
- Incluir uma sequência com risco de hipoglicemia para demonstrar o alerta
- Retornar array numpy com 60 valores em mg/dL
- Incluir comentários explicando cada fase simulada
```

**Resultado:** Função `gerar_dados_demo()` com padrões fisiologicamente plausíveis.

---

## Reflexão sobre os Prompts

**O que funcionou bem:**
- Fornecer contexto médico específico (mg/dL, faixas clínicas, tipo de dataset)
- Pedir justificativas além do código ("justificando cada camada")
- Pedir múltiplos critérios em comparações

**O que precisou de ajuste:**
- Prompts muito genéricos geraram código demasiado simples
- Às vezes foi necessário refinar com "e agora adicione tratamento para X"
- Informações específicas do dataset (nomes exatos das colunas) precisaram ser fornecidas manualmente
