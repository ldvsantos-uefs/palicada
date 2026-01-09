"""
FIGURA 20: EROSIVIDADE (EI30) COM EVENTOS EXTREMOS
Série temporal de erosividade da chuva com classificação P95
Método: Percentil 95 para identificação de eventos extremos
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configurações
plt.style.use('ggplot')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 300

# Diretórios
BASE_DIR = Path(__file__).parent.parent.parent
FIGURAS_DIR = BASE_DIR / "figuras" / "sedimentacao"
DADOS_DIR = BASE_DIR / "dados"

print("=" * 80)
print("SÉRIE TEMPORAL + EVENTOS EXTREMOS - EROSIVIDADE (EI30)")
print("=" * 80)

# Carregar dados
df = pd.read_csv(DADOS_DIR / "dados_integrados_sedimentacao.csv")
df['DATA'] = pd.to_datetime(df['DATA'])

# Remover NaN de EI30
df_valid = df.dropna(subset=['EI30']).copy()

# Classificar eventos extremos usando P95
limiar_precip = df_valid['RAINFALL'].quantile(0.95)
limiar_ei30 = df_valid['EI30'].quantile(0.95)
limiar_incremental = df_valid['FRACIONADO'].quantile(0.95)

# Identificar eventos
eventos_precip = df_valid[df_valid['RAINFALL'] >= limiar_precip].copy()
eventos_ei30 = df_valid[df_valid['EI30'] >= limiar_ei30].copy()
eventos_incremental = df_valid[df_valid['FRACIONADO'] >= limiar_incremental].copy()

print(f"\n✓ Dados válidos: {len(df_valid)} registros (sem NaN em EI30)")
print(f"✓ Limiar Precipitação P95: {limiar_precip:.2f} mm")
print(f"✓ Limiar EI30 P95: {limiar_ei30:.2f} MJ mm ha⁻¹ h⁻¹")
print(f"✓ Limiar Incremental P95: {limiar_incremental:.4f} cm")
print(f"\n✓ Eventos precipitação extrema: {len(eventos_precip)}")
print(f"✓ Eventos EI30 extrema: {len(eventos_ei30)}")
print(f"✓ Eventos sedimentação incremental extrema: {len(eventos_incremental)}")

# =============================================================================
# FIGURA - TRIPLO EIXO Y
# =============================================================================
fig, ax1 = plt.subplots(figsize=(18, 9))

# Cores por área
cores_areas = {'SUP': 'saddlebrown', 'MED': 'darkolivegreen', 'INF': 'indigo'}

# -------------------------
# EIXO 1: Precipitação (linha tracejada azul)
# -------------------------
df_precip = df_valid.drop_duplicates(subset=['DATA']).sort_values('DATA')
ax1.plot(df_precip['DATA'], df_precip['RAINFALL'], '--o', color='steelblue', 
         linewidth=2.5, markersize=7, label='Precipitação Mensal', 
         alpha=0.8, markeredgecolor='navy', markeredgewidth=0.8, dashes=(5, 3))

# Eventos extremos de precipitação
eventos_precip_unicos = eventos_precip.drop_duplicates(subset=['DATA'])
ax1.scatter(eventos_precip_unicos['DATA'], eventos_precip_unicos['RAINFALL'],
           s=300, c='red', marker='*', edgecolors='darkred', linewidths=2,
           label=f'Evento Extremo Precipitação (P95 ≥ {limiar_precip:.0f} mm)',
           alpha=0.95, zorder=5)

ax1.set_ylabel('Precipitação (mm)', fontsize=14, fontweight='bold', color='steelblue')
ax1.tick_params(axis='y', labelcolor='steelblue', labelsize=11)
ax1.set_ylim(bottom=0)
ax1.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)

# -------------------------
# EIXO 2: Erosividade EI30 (linha sólida laranja com área sombreada)
# -------------------------
ax2 = ax1.twinx()

# Plotar EI30 para cada área
for area, cor in cores_areas.items():
    df_area = df_valid[df_valid['AREA'] == area].sort_values('DATA')
    ax2.plot(df_area['DATA'], df_area['EI30'], '-o', color=cor,
            linewidth=2.5, markersize=6, label=f'EI30 {area}',
            alpha=0.85, markeredgecolor='black', markeredgewidth=0.6)
    
    # Preenchimento abaixo da linha
    ax2.fill_between(df_area['DATA'], 0, df_area['EI30'], 
                     color=cor, alpha=0.15)

# Eventos extremos de EI30
ax2.scatter(eventos_ei30['DATA'], eventos_ei30['EI30'],
           s=350, c='gold', marker='D', edgecolors='darkorange', linewidths=2.5,
           label=f'Evento Extremo EI30 (P95 ≥ {limiar_ei30:.0f} MJ mm ha⁻¹ h⁻¹)',
           alpha=0.95, zorder=6)

# Linha horizontal do limiar P95
ax2.axhline(y=limiar_ei30, color='darkorange', linestyle='--', linewidth=2.5,
           label=f'Limiar P95 EI30 = {limiar_ei30:.0f}', alpha=0.7)

ax2.set_ylabel('Erosividade EI30 (MJ mm ha⁻¹ h⁻¹)', fontsize=14, fontweight='bold', color='darkorange')
ax2.tick_params(axis='y', labelcolor='darkorange', labelsize=11)
ax2.set_ylim(bottom=0)

# -------------------------
# EIXO 3: Sedimentação Incremental (lado direito adicional)
# -------------------------
ax3 = ax1.twinx()
ax3.spines['right'].set_position(('outward', 80))

for area, cor in cores_areas.items():
    df_area = df_valid[df_valid['AREA'] == area].sort_values('DATA')
    ax3.plot(df_area['DATA'], df_area['FRACIONADO'], '-.', color=cor,
            linewidth=1.8, markersize=4, alpha=0.6)

# Eventos extremos de sedimentação incremental
ax3.scatter(eventos_incremental['DATA'], eventos_incremental['FRACIONADO'],
           s=250, c='purple', marker='s', edgecolors='indigo', linewidths=2,
           label=f'Evento Extremo Sedim. Incr. (P95 ≥ {limiar_incremental:.3f} cm)',
           alpha=0.85, zorder=4)

ax3.set_ylabel('Sedimentação Incremental (cm/mês)', fontsize=13, fontweight='bold', color='purple')
ax3.tick_params(axis='y', labelcolor='purple', labelsize=10)
ax3.set_ylim(bottom=0)

# -------------------------
# Formatação do eixo X (tempo)
# -------------------------
ax1.set_xlabel('Período de Monitoramento', fontsize=14, fontweight='bold')
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=10)

# -------------------------
# Título e legenda
# -------------------------
plt.title('Erosividade da Chuva (EI30) e Eventos Extremos de Erosão\n' + 
          f'Classificação por Percentil 95 (P95) - {len(df_valid)} observações',
          fontsize=16, fontweight='bold', pad=20)

# Combinar todas as legendas
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines3, labels3 = ax3.get_legend_handles_labels()

all_lines = lines1 + lines2 + lines3
all_labels = labels1 + labels2 + labels3

ax1.legend(all_lines, all_labels, loc='upper left', fontsize=10, 
          framealpha=0.95, edgecolor='black', ncol=2)

# -------------------------
# Anotações dos eventos extremos
# -------------------------
# Anotar os eventos EI30 extremos
for idx, row in eventos_ei30.iterrows():
    ax2.annotate(f"{row['EI30']:.0f}\n{row['AREA']}", 
                xy=(row['DATA'], row['EI30']),
                xytext=(10, 10), textcoords='offset points',
                fontsize=8, color='darkorange', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', color='darkorange', lw=1.5))

plt.tight_layout()

# Salvar figura
output_path = FIGURAS_DIR / "20_serie_eventos_extremos_EROSIVIDADE.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n✅ Figura salva: {output_path}")

# =============================================================================
# ESTATÍSTICAS DOS EVENTOS
# =============================================================================
print("\n" + "=" * 80)
print("ESTATÍSTICAS DOS EVENTOS EXTREMOS (P95)")
print("=" * 80)

print(f"\n📊 LIMIARES CALCULADOS:")
print(f"  • Precipitação P95:      {limiar_precip:.2f} mm")
print(f"  • EI30 P95:              {limiar_ei30:.2f} MJ mm ha⁻¹ h⁻¹")
print(f"  • Sedim. Incremental P95: {limiar_incremental:.4f} cm")

print(f"\n🎯 EVENTOS IDENTIFICADOS:")
print(f"\nPrecipitação Extrema ({len(eventos_precip)} eventos):")
for _, evento in eventos_precip.iterrows():
    print(f"  {evento['DATA'].strftime('%Y-%m')} | {evento['AREA']} | {evento['RAINFALL']:.1f} mm")

print(f"\nErosividade (EI30) Extrema ({len(eventos_ei30)} eventos):")
for _, evento in eventos_ei30.iterrows():
    print(f"  {evento['DATA'].strftime('%Y-%m')} | {evento['AREA']} | {evento['EI30']:.2f} MJ mm ha⁻¹ h⁻¹ | Precip: {evento['RAINFALL']:.1f} mm")

print(f"\nSedimentação Incremental Extrema ({len(eventos_incremental)} eventos):")
for _, evento in eventos_incremental.iterrows():
    print(f"  {evento['DATA'].strftime('%Y-%m')} | {evento['AREA']} | {evento['FRACIONADO']:.4f} cm | EI30: {evento['EI30']:.2f}")

# Correlações
print(f"\n📈 CORRELAÇÕES (dados válidos):")
from scipy import stats
corr_ei30_sedim, p_ei30_sedim = stats.pearsonr(df_valid['EI30'], df_valid['FRACIONADO'])
corr_precip_ei30, p_precip_ei30 = stats.pearsonr(df_valid['RAINFALL'], df_valid['EI30'])
corr_precip_sedim, p_precip_sedim = stats.pearsonr(df_valid['RAINFALL'], df_valid['FRACIONADO'])

print(f"  • EI30 × Sedim. Incremental:   r = {corr_ei30_sedim:.4f} (p = {p_ei30_sedim:.4f})")
print(f"  • Precipitação × EI30:         r = {corr_precip_ei30:.4f} (p = {p_precip_ei30:.4f})")
print(f"  • Precipitação × Sedim. Incr.: r = {corr_precip_sedim:.4f} (p = {p_precip_sedim:.4f})")

print("\n" + "=" * 80)
print("✅ ANÁLISE CONCLUÍDA!")
print("=" * 80)
