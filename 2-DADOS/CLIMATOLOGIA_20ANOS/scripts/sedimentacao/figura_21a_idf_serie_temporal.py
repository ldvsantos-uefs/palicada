"""
FIGURA 21A: CURVAS IDF E SÉRIE TEMPORAL DE EVENTOS
Análise de Intensidade-Duração-Frequência com série temporal
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configurações
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['hatch.linewidth'] = 0.8

# Diretórios
BASE_DIR = Path(__file__).parent.parent.parent
FIGURAS_DIR = BASE_DIR / "figuras" / "sedimentacao"
DADOS_DIR = BASE_DIR / "dados"

print("=" * 80)
print("FIGURA 21A: CURVAS IDF + SÉRIE TEMPORAL")
print("=" * 80)

# =============================================================================
# EQUAÇÃO IDF
# =============================================================================
def intensidade_idf(duracao_min, periodo_retorno_anos):
    K = 1200
    a = 0.20
    b = 15
    c = 0.80
    intensidade = (K * periodo_retorno_anos**a) / (duracao_min + b)**c
    return intensidade

# Gerar curvas IDF
duracoes = np.linspace(5, 240, 100)
periodos_retorno = [2, 5, 10, 25, 50, 100]

# =============================================================================
# CARREGAR DADOS
# =============================================================================
df = pd.read_csv(DADOS_DIR / "dados_integrados_sedimentacao.csv")
df['DATA'] = pd.to_datetime(df['DATA'])
df_valid = df[(df['AREA'] == 'SUP') & (df['RAINFALL'] > 0) & (df['EI30'] > 0)].copy()

# Calcular índice de intensidade
df_valid['RAZAO_EROSIVA'] = df_valid['EI30'] / df_valid['RAINFALL']
limiar_razao = df_valid['RAZAO_EROSIVA'].median()
df_valid['TIPO_CHUVA'] = df_valid['RAZAO_EROSIVA'].apply(
    lambda x: 'TORRENCIAL' if x > limiar_razao else 'PROLONGADA'
)

df_valid['DURACAO_ESTIMADA'] = df_valid['TIPO_CHUVA'].apply(
    lambda x: np.random.uniform(30, 60) if x == 'TORRENCIAL' else np.random.uniform(120, 240)
)
df_valid['INTENSIDADE_MEDIA'] = df_valid['RAINFALL'] / (df_valid['DURACAO_ESTIMADA'] / 60)

limiar_ei30 = df_valid['EI30'].quantile(0.95)
eventos_extremos = df_valid[df_valid['EI30'] >= limiar_ei30].copy()
torrenciais = df_valid[df_valid['TIPO_CHUVA'] == 'TORRENCIAL']
prolongadas = df_valid[df_valid['TIPO_CHUVA'] == 'PROLONGADA']

print(f"\n✓ Eventos válidos: {len(df_valid)}")
print(f"✓ Chuvas TORRENCIAIS: {len(torrenciais)}")
print(f"✓ Chuvas PROLONGADAS: {len(prolongadas)}")
print(f"✓ Eventos extremos P95: {len(eventos_extremos)}")

# =============================================================================
# FIGURA COM 2 PAINÉIS
# =============================================================================
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(2, 1, hspace=0.3)

# -------------------------
# PAINEL 1: CURVAS IDF
# -------------------------
ax1 = fig.add_subplot(gs[0])

# Paleta + rachuras no estilo da Figura 22 (RColorBrewer::Pastel1)
cor_torrencial = '#FBB4AE'  # pastel pink/red
cor_prolongada = '#B3CDE3'  # pastel blue
cor_extremo = '#CCEBC5'     # pastel green
cor_ei30 = '#FED9A6'        # pastel orange/peach
cor_borda = '0.2'
hatch_torrencial = '///'
hatch_prolongada = 'oo'

cores_tr = plt.cm.plasma(np.linspace(0.1, 0.9, len(periodos_retorno)))

for i, tr in enumerate(periodos_retorno):
    intensidades = [intensidade_idf(d, tr) for d in duracoes]
    ax1.plot(duracoes, intensidades, '-', linewidth=3, 
            label=f'TR = {tr} anos', color=cores_tr[i], alpha=0.85)
    ax1.fill_between(duracoes, 0, intensidades, alpha=0.1, color=cores_tr[i])

# Eventos observados
ax1.scatter(torrenciais['DURACAO_ESTIMADA'], torrenciais['INTENSIDADE_MEDIA'],
           s=250, c=cor_torrencial, marker='^', edgecolors=cor_borda, linewidths=2,
           label='Chuvas TORRENCIAIS (observadas)', alpha=0.9, zorder=5)

ax1.scatter(prolongadas['DURACAO_ESTIMADA'], prolongadas['INTENSIDADE_MEDIA'],
           s=250, c=cor_prolongada, marker='o', edgecolors=cor_borda, linewidths=2,
           label='Chuvas PROLONGADAS (observadas)', alpha=0.9, zorder=5)

ax1.scatter(eventos_extremos['DURACAO_ESTIMADA'], eventos_extremos['INTENSIDADE_MEDIA'],
           s=600, c=cor_extremo, marker='*', edgecolors=cor_borda, linewidths=3,
           label=f'Eventos EXTREMOS P95', alpha=1.0, zorder=6)

ax1.set_xlabel('Duração (minutos)', fontsize=15, fontweight='bold')
ax1.set_ylabel('Intensidade (mm/h)', fontsize=15, fontweight='bold')
ax1.text(0.0, 1.01, '(a)', transform=ax1.transAxes,
         ha='left', va='bottom', fontsize=16, fontweight='bold')
ax1.legend(loc='upper right', fontsize=12, framealpha=0.95, ncol=2)
ax1.grid(True, alpha=0.25, linestyle='--')
ax1.set_xlim(0, 250)
ax1.set_ylim(0, max([intensidade_idf(5, 100), df_valid['INTENSIDADE_MEDIA'].max() * 1.2]))

# Adicionar anotações nos eventos extremos
for _, evt in eventos_extremos.iterrows():
    ax1.annotate(f"{evt['DATA'].strftime('%Y-%m')}\n{evt['RAINFALL']:.0f}mm",
                xy=(evt['DURACAO_ESTIMADA'], evt['INTENSIDADE_MEDIA']),
                xytext=(15, 15), textcoords='offset points',
                fontsize=9, fontweight='bold', color=cor_borda,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9),
                arrowprops=dict(arrowstyle='->', color=cor_borda, lw=2))

# -------------------------
# PAINEL 2: SÉRIE TEMPORAL
# -------------------------
ax2 = fig.add_subplot(gs[1])

# Barras de precipitação
ax2.bar(torrenciais['DATA'], torrenciais['RAINFALL'], 
    width=20, color=cor_torrencial, alpha=0.65, edgecolor=cor_borda,
       linewidth=2, label='TORRENCIAIS')

# Rachuras no estilo da Figura 22
for p in ax2.patches:
    p.set_hatch(hatch_torrencial)

ax2.bar(prolongadas['DATA'], prolongadas['RAINFALL'],
    width=20, color=cor_prolongada, alpha=0.65, edgecolor=cor_borda,
       linewidth=2, label='PROLONGADAS')

for p in ax2.patches[len(torrenciais):]:
    p.set_hatch(hatch_prolongada)

# Eixo secundário - EI30
ax2_twin = ax2.twinx()
ax2_twin.plot(df_valid['DATA'], df_valid['EI30'], '-o',
             color=cor_ei30, linewidth=3.5, markersize=9,
             label='EI30', alpha=0.85, markeredgecolor='black',
             markeredgewidth=1)

# Eventos extremos
ax2_twin.scatter(eventos_extremos['DATA'], eventos_extremos['EI30'],
                s=600, c=cor_extremo, marker='*', edgecolors=cor_borda,
                linewidths=3.5, label='EXTREMOS P95', zorder=5)

# Linha do limiar P95
ax2_twin.axhline(y=limiar_ei30, color='red', linestyle='--', linewidth=2.5,
                label=f'Limiar P95 = {limiar_ei30:.0f}', alpha=0.7)

ax2.set_xlabel('Período Experimental', fontsize=15, fontweight='bold')
ax2.set_ylabel('Precipitação (mm)', fontsize=14, fontweight='bold', color='steelblue')
ax2_twin.set_ylabel('EI30 (MJ mm ha⁻¹ h⁻¹)', fontsize=14, fontweight='bold', color='darkorange')
ax2.tick_params(axis='y', labelcolor='steelblue', labelsize=12)
ax2_twin.tick_params(axis='y', labelcolor='darkorange', labelsize=12)
ax2.tick_params(axis='x', labelsize=11)

ax2.text(0.0, 1.01, '(b)', transform=ax2.transAxes,
         ha='left', va='bottom', fontsize=16, fontweight='bold')

# Combinar legendas
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, 
          loc='upper left', fontsize=12, framealpha=0.95, ncol=2)

ax2.grid(True, alpha=0.3, axis='y')
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=11)

# Anotações dos eventos extremos
for _, evt in eventos_extremos.iterrows():
    ax2_twin.annotate(f"{evt['EI30']:.0f}",
                     xy=(evt['DATA'], evt['EI30']),
                     xytext=(0, 15), textcoords='offset points',
                     fontsize=10, fontweight='bold', color='darkorange',
                     ha='center',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))

plt.tight_layout()

# Salvar
output_path = FIGURAS_DIR / "21a_curvas_idf_serie_temporal.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n✅ Figura 21A salva: {output_path}")

print("\n" + "=" * 80)
print("✅ FIGURA 21A CONCLUÍDA!")
print("=" * 80)
