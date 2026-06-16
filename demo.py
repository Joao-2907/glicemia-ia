import sys
import numpy as np
import os

os.makedirs("outputs", exist_ok=True)

print("\n" + "=" * 60)
print("   🩺 GlicemIA — Demonstração Completa do Pipeline")
print("=" * 60)

# ── Etapa 1: Pré-processamento ─────────────────────────────────────────
print("\n ETAPA 1: Pré-processamento de Dados CGM")
print("─" * 50)

from src.preprocessing.preprocessamento import criar_janelas_deslizantes, gerar_dados_demo
from sklearn.preprocessing import MinMaxScaler

dados_demo = gerar_dados_demo()
print(f"  ✅ Dados sintéticos gerados: {len(dados_demo)} leituras CGM")
print(f"  Faixa: {dados_demo.min():.1f} – {dados_demo.max():.1f} mg/dL")
print(f"  Tendência: {'⬇ queda (risco hipo)' if dados_demo[-1] < dados_demo[0] else '⬆ subida'}")


np.random.seed(42)
dados_longos = np.concatenate([
    np.linspace(160, 90, 150) + np.random.normal(0, 4, 150),
    np.linspace(90, 65, 100) + np.random.normal(0, 3, 100),
    np.linspace(65, 80, 100) + np.random.normal(0, 3, 100),
    np.linspace(80, 110, 150) + np.random.normal(0, 4, 150),
])

scaler = MinMaxScaler(feature_range=(0, 1))
dados_norm = scaler.fit_transform(dados_longos.reshape(-1, 1))

X, y = criar_janelas_deslizantes(dados_norm, tamanho_janela=60, horizonte_predicao=60)

print(f"\n  Shape X (entradas): {X.shape}")
print(f"  Shape y (alvos)   : {y.shape}")
print(f"  Cada janela representa ~5h de histórico glicêmico")
print(f"  O alvo é a glicemia ~30 minutos à frente")


# ── Etapa 2: Simulação do Modelo ───────────────────────────────────────
print("\n📌 ETAPA 2: Simulação do Modelo LSTM (modo demo)")
print("─" * 50)

print("  Arquitetura do modelo LSTM:")
print("  ┌─────────────────────────────────────────┐")
print("  │  Input Layer    (None, 60, 1)           │")
print("  │  LSTM Layer 1   64 units  → 33.280 parm │")
print("  │  Dropout 0.2                            │")
print("  │  LSTM Layer 2   32 units  → 12.416 parm │")
print("  │  Dropout 0.2                            │")
print("  │  Dense Layer    1 unit    →     33 parm │")
print("  │────────────────────────────────────────  │")
print("  │  Total parâmetros treináveis: 45.729    │")
print("  └─────────────────────────────────────────┘")

# Simulação de resultados de treino (valores realistas do dataset Shanghai)
print("\n  📊 Resultados simulados de treinamento:")
mae_sim = 12.4
rmse_sim = 17.8
print(f"  MAE  (Erro Médio Absoluto) : {mae_sim:.1f} mg/dL  ✅ (< 15 mg/dL)")
print(f"  RMSE (Raiz Erro Quadrático): {rmse_sim:.1f} mg/dL  ✅ (< 20 mg/dL)")


# ── Etapa 3: Classificação de Risco ────────────────────────────────────
print("\n📌 ETAPA 3: Classificação de Risco Clínico")
print("─" * 50)

from src.api.main import _classificar_risco

cenarios = [
    ("Cenário 1 — Hipoglicemia Iminente", 62.0),
    ("Cenário 2 — Zona de Alerta",        83.0),
    ("Cenário 3 — Glicemia Normal",       118.0),
    ("Cenário 4 — Hiperglicemia",         210.0),
]

for nome, valor in cenarios:
    cls, msg = _classificar_risco(valor)
    print(f"\n  {nome}")
    print(f"  Predição: {valor} mg/dL  →  [{cls}]")
    print(f"  Mensagem: {msg[:80]}...")


# ── Etapa 4: Simulação da API ───────────────────────────────────────────
print("\n ETAPA 4: Teste da API em modo demo")
print("─" * 50)

from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

# Health check
health = client.get("/health").json()
print(f"  GET /health → status: {health['status']} | modelo: {health['model_loaded']}")

# Predição com dados demo
payload = {
    "leituras_cgm": [round(float(v), 1) for v in dados_demo],
    "paciente_id": "demo-apresentacao"
}
resp = client.post("/predict", json=payload).json()

print(f"\n  POST /predict →")
print(f"  Predição          : {resp['predicao_mgdl']} mg/dL")
print(f"  Classificação     : {resp['classificacao_risco']}")
print(f"  Horizonte         : {resp['horizonte_minutos']} minutos à frente")
print(f"  Modo demo         : {resp['modo_demo']}")
print(f"  Mensagem          : {resp['mensagem'][:80]}...")


# ── Etapa 5: Gráfico de demonstração ───────────────────────────────────
print("\n ETAPA 5: Gerando visualização de demonstração")
print("─" * 50)

try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Gráfico 1: Série temporal com zonas de risco
    x_eixo = np.arange(len(dados_longos)) * 5  # minutos
    axes[0].plot(x_eixo, dados_longos, color='#065A82', linewidth=1.0, alpha=0.8)
    axes[0].fill_between(x_eixo, 0, 70, alpha=0.12, color='red', label='Hipoglicemia (<70)')
    axes[0].fill_between(x_eixo, 70, 90, alpha=0.12, color='orange', label='Alerta (70–90)')
    axes[0].fill_between(x_eixo, 180, 300, alpha=0.10, color='goldenrod', label='Hiperglicemia (>180)')
    axes[0].axhline(70, color='red', linestyle='--', linewidth=0.8, alpha=0.7)
    axes[0].axhline(90, color='orange', linestyle='--', linewidth=0.8, alpha=0.7)
    axes[0].axhline(180, color='goldenrod', linestyle='--', linewidth=0.8, alpha=0.7)
    axes[0].set_title('Série CGM com Zonas de Risco Clínico', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Tempo (minutos)')
    axes[0].set_ylabel('Glicemia (mg/dL)')
    axes[0].set_ylim(40, 200)
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].grid(alpha=0.25)

    # Gráfico 2: Dados de demo com predição
    x_demo = np.arange(len(dados_demo)) * 5
    axes[1].plot(x_demo, dados_demo, color='#065A82', linewidth=1.5, label='Histórico CGM (entrada)')
    pred_val = resp['predicao_mgdl']
    pred_x = (len(dados_demo) + 6) * 5  # 30 min depois
    axes[1].scatter([pred_x], [pred_val], color='#F96167', s=150, zorder=5,
                    label=f'Predição +30min: {pred_val} mg/dL')
    axes[1].axhline(70, color='red', linestyle=':', alpha=0.5)
    axes[1].axhline(90, color='orange', linestyle=':', alpha=0.5)
    axes[1].annotate(f'{resp["classificacao_risco"]}\n{pred_val} mg/dL',
                     xy=(pred_x, pred_val),
                     xytext=(pred_x - 40, pred_val + 15),
                     fontsize=9, color='#F96167', fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color='#F96167', lw=1.5))
    axes[1].set_title('Demonstração: Predição de Glicemia Futura', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Tempo (minutos)')
    axes[1].set_ylabel('Glicemia (mg/dL)')
    axes[1].set_ylim(40, 200)
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.25)

    plt.suptitle('GlicemIA — App de Predição Glicêmica com LSTM',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('outputs/demo_completa.png', dpi=150, bbox_inches='tight')
    print("  ✅ Gráfico salvo em: outputs/demo_completa.png")
    plt.close()
except ImportError:
    print("  ⚠️  matplotlib não disponível — gráfico não gerado")


# ── Resumo ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("   ✅ Demonstração concluída com sucesso!")
print("=" * 60)
print("""
""")
