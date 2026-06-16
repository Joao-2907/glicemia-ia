# Aprendizados com IA — GlicemIA

**Projeto:** App de Predição Glicêmica  
**Equipe:** João Victor Almeida Souza, Yasmim Elisa da Cruz Ferreira

---

## Aprendizados Técnicos

### 1. Spec Driven Design com IA acelera o desenvolvimento

Usar a IA para gerar specs **antes** de escrever código mudou nossa abordagem. Em vez de sair programando e descobrir os problemas no meio, a spec nos forçou a pensar nos casos de borda, nos parâmetros configuráveis e nos critérios de aceitação desde o início.

**Impacto prático:** Quando fomos implementar `preparar_dados()`, já sabíamos que precisávamos retornar o `scaler` (não apenas X e y) — isso estava na spec. Sem a spec, provavelmente descobriríamos essa necessidade apenas na hora de desnormalizar as predições da API.

---

### 2. Data Leakage em séries temporais — o erro que a IA nos evitou

Ao pedir para a IA revisar nossa estratégia de treino/validação, ela nos alertou:

> "Em séries temporais, embaralhar os dados antes do split cria data leakage — o modelo aprende com dados 'do futuro' durante o treino e apresenta performance artificialmente alta. Sempre faça split temporal, mantendo a ordem cronológica."

Esse conceito era novo para nós e provavelmente teríamos criado um modelo com métricas mentirosas sem esse aviso.

---

### 3. A IA não conhece o seu domínio específico — você precisa ensinar

O dataset Shanghai T1DM tem uma coluna chamada `'CGM (mg / dl)'` (com espaços ao redor do `/`). Quando pedimos à IA para gerar código de leitura, ela usou nomes genéricos como `'glucose'` ou `'cgm_value'`.

**Lição:** A IA é um parceiro técnico excelente, mas o conhecimento do domínio — os nomes das colunas, as faixas clínicas, o período pós-prandial — precisa vir de você. A IA amplifica o seu conhecimento; não o substitui.

---

### 4. Prompts com justificativas ensinam mais do que prompts que só pedem código

Comparamos dois tipos de prompt:

| Prompt | Resultado |
|--------|-----------|
| "Me dê o código para um modelo LSTM de predição de glicemia" | Código funcional, mas sem entender por quê |
| "Me dê o código e **justifique cada decisão arquitetural** para um iniciante em ML" | Código + explicação do Dropout, do Early Stopping, do por que 2 camadas LSTM |

O segundo tipo de prompt foi muito mais valioso para o aprendizado.

---

### 5. A IA comete erros — e você precisa saber identificá-los

Durante o desenvolvimento, a IA gerou um erro sutil na função de janelas deslizantes: o índice de `alvo_y` estava deslocado por 1, o que faria o modelo prever `horizonte-1` passos à frente em vez de `horizonte`. O erro só foi identificado quando testamos manualmente com dados conhecidos.

**Lição:** Nunca cole código da IA sem entender o que ele faz. Testes unitários são essenciais.

---

## Aprendizados sobre Empreendedorismo + IA

### 6. IA reduz a barreira de entrada para construir MVPs

Um dos maiores aprendizados foi perceber que, com IA como assistente, dois estudantes universitários conseguiram estruturar em semanas um projeto que exigiria meses sem essa ferramenta. Isso tem implicação direta para empreendedorismo: a IA democratiza a capacidade de construir protótipos e validar ideias rapidamente.

---

### 7. O diferencial competitivo migrou do "saber programar" para "saber perguntar"

A habilidade mais valiosa que desenvolvemos não foi escrever código Python — foi aprender a fazer as perguntas certas para a IA. Prompts bem formulados, com contexto médico, critérios claros e pedido de justificativas, geraram resultados exponencialmente melhores do que perguntas genéricas.

---

### 8. Documentar o uso de IA é responsabilidade do desenvolvedor

No contexto acadêmico e profissional, saber comunicar claramente o que foi feito com IA e o que foi feito manualmente é uma habilidade emergente e esperada. Este projeto nos treinou a registrar sistematicamente essa fronteira — o que é uma boa prática de integridade intelectual.

---

## Resumo para a Apresentação

| Aprendizado | Uma frase |
|-------------|-----------|
| Spec antes de código | Evita retrabalho e força pensar nos casos de borda |
| Data leakage | Em séries temporais, nunca embaralhe antes do split |
| Domínio é seu | A IA amplifica seu conhecimento; não o substitui |
| Prompts com "porquê" | Ensina mais do que prompts que só pedem código |
| Verificar o output da IA | IA comete erros sutis; teste sempre |
| IA democratiza MVPs | Dois estudantes entregaram o que levaria meses sem IA |
