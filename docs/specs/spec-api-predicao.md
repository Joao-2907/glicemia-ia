# SPEC: API REST de Predição Glicêmica

**Feature:** `api-predicao`  
**Status:** ✅ Implementado  
**Arquivo de implementação:** `src/api/main.py`

---

## 1. Contexto

Com o modelo LSTM treinado e salvo, o sistema precisa expor as predições via API REST para que aplicativos móveis ou sistemas hospitalares possam consumir as previsões de glicemia em tempo real.

---

## 2. Objetivo

Disponibilizar um endpoint HTTP que recebe as últimas N leituras de CGM de um paciente e retorna a predição do nível glicêmico em ~30 minutos, com classificação de risco.

---

## 3. Endpoints

### `GET /health`

Verificação de saúde da API.

**Response 200:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "version": "1.0.0"
}
```

---

### `POST /predict`

Realiza predição de glicemia futura.

**Request Body:**
```json
{
  "leituras_cgm": [120, 118, 115, 113, 110, ...],
  "paciente_id": "opcional-string"
}
```

| Campo | Tipo | Obrigatório | Descrição |
|-------|------|-------------|-----------|
| `leituras_cgm` | `list[float]` | ✅ | Array com exatamente 60 leituras em mg/dL (sequência mais recente primeiro para a mais antiga) |
| `paciente_id` | `string` | ❌ | Identificador opcional do paciente para logging |

**Response 200:**
```json
{
  "predicao_mgdl": 87.4,
  "classificacao_risco": "ALERTA",
  "mensagem": "Glicemia prevista para os próximos 30 minutos está abaixo de 90 mg/dL. Considere ingestão de carboidratos.",
  "timestamp_predicao": "2026-06-14T10:30:00Z",
  "horizonte_minutos": 30
}
```

**Classificação de Risco:**

| Classificação | Faixa (mg/dL) | Cor |
|---------------|--------------|-----|
| `HIPOGLICEMIA` | < 70 | 🔴 Vermelho |
| `ALERTA` | 70 – 89 | 🟠 Laranja |
| `NORMAL` | 90 – 180 | 🟢 Verde |
| `HIPERGLICEMIA` | > 180 | 🟡 Amarelo |

**Response 422 (dados inválidos):**
```json
{
  "detail": "São necessárias exatamente 60 leituras CGM. Recebidas: 45"
}
```

---

### `GET /docs`

Documentação interativa Swagger UI (gerada automaticamente pelo FastAPI).

---

## 4. Requisitos Funcionais

| ID | Requisito | Critério de Aceitação |
|----|-----------|----------------------|
| RF01 | Carregar modelo LSTM na inicialização da API | Modelo carregado em < 5s na inicialização |
| RF02 | Validar que o array tem exatamente 60 leituras | Retornar 422 com mensagem clara se diferente |
| RF03 | Normalizar entrada antes de inferência | Usar o mesmo scaler do treino |
| RF04 | Desnormalizar saída para mg/dL | Valor final em unidade clínica |
| RF05 | Classificar o risco conforme tabela acima | Classificação correta por faixa |
| RF06 | Responder em < 1 segundo | Latência aceitável para uso em app mobile |

---

## 5. Requisitos Não Funcionais

| ID | Requisito |
|----|-----------|
| RNF01 | API implementada com FastAPI |
| RNF02 | Documentação automática via Swagger (/docs) |
| RNF03 | CORS habilitado para integração com front-end |

---

## 6. Dependências

```
fastapi>=0.110
uvicorn>=0.29
pydantic>=2.6
tensorflow>=2.13
numpy>=1.24
scikit-learn>=1.3
```

---

## 7. Como Executar

```bash
uvicorn src.api.main:app --reload --port 8000
```

Acessar: `http://localhost:8000/docs`

---

## 8. Decisões de Design

- **FastAPI sobre Flask:** FastAPI gera documentação automática (OpenAPI/Swagger), tem validação nativa de tipos via Pydantic e performance superior para ML serving.
- **60 leituras fixas:** Padroniza o contrato da API com o mesmo `tamanho_janela` usado no treinamento; simplifica validação.
- **Classificação de risco na API:** Centraliza a lógica de negócio no backend; front-end apenas exibe a classificação recebida.
