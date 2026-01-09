"""
FIGURA 1: SEDIMENTAÇÃO ACUMULADA
Série temporal com precipitação e eventos extremos
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
print("SÉRIE TEMPORAL + EVENTOS EXTREMOS - SEDIMENTAÇÃO ACUMULADA")
print("=" * 80)

# Carregar dados
df = pd.read_csv(DADOS_DIR / "dados_integrados_sedimentacao.csv")
df['DATA'] = pd.to_datetime(df['DATA'])

# Classificar eventos extremos (P95)
limiar_precip = df['RAINFALL'].quantile(0.95)
limiar_acumulada = df['SEDIMENT'].quantile(0.95)

eventos_precip = df[df['RAINFALL'] >= limiar_precip].copy()
eventos_acumulada = df[df['SEDIMENT'] >= limiar_acumulada].copy()

print(f"\n✓ Dados: {len(df)} registros")
print(f"✓ Limiar Precipitação P95: {limiar_precip:.2f} mm")
print(f"✓ Limiar Acumulada P95: {limiar_acumulada:.4f} cm")
print(f"✓ Eventos precipitação extrema: {len(eventos_precip)}")
print(f"✓ Eventos acumulada extrema: {len(eventos_acumulada)}")

# =============================================================================
# FIGURA
# =============================================================================
fig, ax1 = plt.subplots(figsize=(16, 8))

# Cores por área
cores_areas = {'SUP': 'saddlebrown', 'MED': 'darkolivegreen', 'INF': 'indigo'}

# Eixo 1: Precipitação (tracejada azul)
df_precip = df.drop_duplicates(subset=['DATA']).sort_values('DATA')
ax1.plot(df_precip['DATA'], df_precip['RAINFALL'], '--o', color='steelblue', 
         linewidth=2.5, markersize=7, label='Precipitação Mensal', 
         alpha=0.8, markeredgecolor='navy', markeredgewidth=0.8, dashes=(5, 3))

# Eventos extremos de precipitação
eventos_precip_unicos = eventos_precip.drop_duplicates(subset=['DATA'])
ax1.scatter(eventos_precip_unicos['DATA'], eventos_precip_unicos['RAINFALL'], 
           color='crimson', s=400, marker='*', zorder=10, 
           label=f'Precipitação Extrema (≥P95: {limiar_precip:.1f} mm)', 
           edgecolors='darkred', linewidth=2)

# Limiar de precipitação
ax1.axhline(y=limiar_precip, color='orangered', linestyle='--', linewidth=2.5, 
           alpha=0.6, label='Limiar P95 Precipitação')

ax1.set_xlabel('Data', fontweight='bold', fontsize=13)
ax1.set_ylabel('Precipitação Mensal (mm)', fontweight='bold', fontsize=13, color='steelblue')
ax1.tick_params(axis='y', labelcolor='steelblue', labelsize=11)
ax1.tick_params(axis='x', labelsize=11)
ax1.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)

# Eixo 2: Sedimentação ACUMULADA
ax2 = ax1.twinx()
for area in df['AREA'].unique():
    df_area = df[df['AREA'] == area].sort_values('DATA')
    ax2.plot(df_area['DATA'], df_area['SEDIMENT'], '-o', 
             color=cores_areas.get(area, 'gray'), 
             linewidth=2.5, markersize=7, label=f'Sedimentação Acumulada - {area}', 
             alpha=0.85, markeredgecolor='black', markeredgewidth=0.5)

# Eventos extremos ACUMULADA
ax2.scatter(eventos_acumulada['DATA'], eventos_acumulada['SEDIMENT'],
           color='gold', s=300, marker='D', zorder=9,
           label=f'Acumulada Extrema (≥P95: {limiar_acumulada:.4f} cm)',
           edgecolors='darkgoldenrod', linewidth=2, alpha=0.95)

# Limiar acumulada
ax2.axhline(y=limiar_acumulada, color='goldenrod', linestyle=':', linewidth=2.5, 
           alpha=0.7, label='Limiar P95 Acumulada')

ax2.set_ylabel('Sedimentação Acumulada (cm)', fontweight='bold', fontsize=13, color='saddlebrown')
ax2.tick_params(axis='y', labelcolor='saddlebrown', labelsize=11)

# Formatação do eixo x
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

# Título
plt.title('Série Temporal: Precipitação × Sedimentação Acumulada (Eventos Extremos Destacados)',
         fontsize=14, fontweight='bold', pad=20)

# Legendas combinadas
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2,
          loc='upper left', fontsize=10, framealpha=0.95,
          edgecolor='black', fancybox=True, shadow=True)

# Estatísticas
stats_text = f"""Estatísticas (Acumulada):
• Precipitação média: {df['RAINFALL'].mean():.1f} mm
• Sedimentação acumulada máxima: {df['SEDIMENT'].max():.3f} cm
• Eventos precipitação extrema: {len(eventos_precip)}
• Eventos acumulada extrema: {len(eventos_acumulada)}
• Período: {df['DATA'].min().strftime('%b/%Y')} - {df['DATA'].max().strftime('%b/%Y')}

Interpretação:
Sedimentação acumulada reflete perda TOTAL de solo
até cada momento (integral temporal)"""

ax1.text(0.98, 0.50, stats_text, transform=ax1.transAxes,
        fontsize=9, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85, edgecolor='black'),
        family='monospace')

plt.tight_layout()
plt.savefig(FIGURAS_DIR / "18_serie_eventos_extremos_ACUMULADA.png", 
           dpi=300, bbox_inches='tight')
print("\n✓ Figura salva: 18_serie_eventos_extremos_ACUMULADA.png")
plt.close()

print("\n" + "=" * 80)
print("CONCLUÍDO!")
print("=" * 80)
