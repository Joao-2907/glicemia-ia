# Uso de IA no Desenvolvimento — GlicemIA

**Projeto:** App de Predição Glicêmica  
**Equipe:** João Victor Almeida Souza, Yasmim Elisa da Cruz Ferreira  
**Ferramentas de IA utilizadas:** Cursor IDE (Claude Sonnet), ChatGPT (pesquisa complementar)

---

## Resumo

A IA foi utilizada em todas as etapas do desenvolvimento, desde a concepção das features até a escrita da documentação. Este documento registra onde e como a IA contribuiu, conforme exigido pelo enunciado do trabalho.

---

## 1. Etapa: Spec Driven Design

**Onde a IA atuou:** Geração e refinamento das especificações de feature antes da implementação.

**Como foi usado:**
- Pedimos ao Claude para transformar nossa ideia de cada feature em uma spec estruturada no formato Spec Driven Design (tabela de requisitos, critérios de aceitação, diagrama de fluxo, casos de borda).
- A IA sugeriu campos que havíamos esquecido: como o critério de qualidade em mg/dL (MAE < 15) e a necessidade de retornar o `scaler` junto com os dados pré-processados.

**Prompt utilizado (exemplo):**
> "Você é um arquiteto de software. Crie uma spec no formato Spec Driven Design para um pipeline de pré-processamento de dados CGM de diabetes que: lê arquivos Excel, extrai coluna de glicemia, normaliza com MinMaxScaler e cria janelas deslizantes para LSTM."

**Resultado:** Specs em `/docs/specs/` prontas antes de uma linha de código ser escrita.

---

## 2. Etapa: Implementação do Pré-processamento

**Onde a IA atuou:** Geração e revisão do código `preprocessamento.py`.

**Como foi usado:**
- O código base `criar_janelas_deslizantes()` e `preparar_dados()` foi iniciado manualmente pelo time.
- A IA foi usada para revisar o código, sugerir o tratamento de exceções (FileNotFoundError, KeyError para coluna CGM ausente) e melhorar os logs de output.
- A IA também explicou por que o split temporal não deve embaralhar os dados — conceito fundamental para séries temporais.

**Contribuição humana vs IA:**

| Tarefa | Humano | IA |
|--------|--------|-----|
| Lógica da janela deslizante | ✅ | Revisão |
| Leitura do Excel + extração coluna | ✅ | Revisão |
| Tratamento de erros (try/except) | Parcial | ✅ Complementou |
| Logging informativo | ✅ | Melhorou mensagens |

---

## 3. Etapa: Modelo LSTM

**Onde a IA atuou:** Definição da arquitetura e parâmetros de treino.

**Como foi usado:**
- Perguntamos ao Claude qual arquitetura LSTM seria adequada para predição de séries temporais médicas de curto prazo.
- A IA sugeriu a combinação de 2 camadas LSTM (64 + 32 units) com Dropout para evitar overfitting, e justificou o uso de Early Stopping com patience=10.
- A IA também nos alertou sobre data leakage em séries temporais quando há shuffle no split de dados — erro que provavelmente cometeríamos sem esse aviso.

**Aprendizado chave:** A IA não apenas gerou o código, mas explicou o "porquê" de cada escolha arquitetural.

---

## 4. Etapa: API FastAPI

**Onde a IA atuou:** Estrutura da API e design dos endpoints.

**Como foi usado:**
- Pedimos para a IA comparar Flask vs FastAPI para ML serving; ela recomendou FastAPI pela geração automática de documentação Swagger e validação nativa via Pydantic.
- A IA sugeriu o design da resposta com `classificacao_risco` e `mensagem` amigável — pensando no usuário final do app.
- Gerou o esqueleto dos modelos Pydantic e o handler de erro 422.

---

## 5. Etapa: Documentação

**Onde a IA atuou:** Estrutura do README, MDs de IA, refinamento de linguagem.

**Como foi usado:**
- O README foi co-criado com a IA: fornecemos os dados do projeto e a IA sugeriu a estrutura de seções, as tabelas comparativas e o formato de "como executar".
- Os MDs de IA (este arquivo e `prompts-utilizados.md`) foram estruturados com auxílio da IA para garantir clareza e completude.

---

## 6. Limitações e Cuidados

| Situação | Como Lidamos |
|----------|-------------|
| IA gerou código com bug (índice errado na janela deslizante) | Identificamos testando manualmente e corrigimos |
| IA sugeriu biblioteca incompatível com Python 3.10 | Verificamos a versão antes de instalar |
| IA não conhecia o dataset Shanghai especificamente | Fornecemos a estrutura do arquivo Excel manualmente |
| IA às vezes era genérica demais | Refinamos os prompts com contexto médico específico |

---

## 7. Reflexão Final

A IA funcionou como um **par de programação** durante todo o projeto. Ela acelerou significativamente as etapas de documentação e boilerplate, nos alertou sobre boas práticas que não havíamos considerado, e nos ajudou a estruturar o raciocínio antes de escrever código.

O que a IA **não fez**: tomar decisões sobre o domínio médico (os limiares de risco glicêmico foram validados por nós com base na literatura), nem substituiu o entendimento que precisamos ter do problema para usar a ferramenta com responsabilidade.
