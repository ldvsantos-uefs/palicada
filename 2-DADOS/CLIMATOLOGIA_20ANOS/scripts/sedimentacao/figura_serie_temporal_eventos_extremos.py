"""
SÉRIE TEMPORAL: PRECIPITAÇÃO E SEDIMENTAÇÃO COM EVENTOS EXTREMOS
Figura única, expandida e detalhada
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
print("SÉRIE TEMPORAL: PRECIPITAÇÃO × SEDIMENTAÇÃO × EVENTOS EXTREMOS")
print("=" * 80)

# Carregar dados
df = pd.read_csv(DADOS_DIR / "dados_integrados_sedimentacao.csv")
df['DATA'] = pd.to_datetime(df['DATA'])

# Classificar eventos extremos (P95 - mais restritivo)
limiar_precip_p95 = df['RAINFALL'].quantile(0.95)
limiar_sed_p95 = df['FRACIONADO'].quantile(0.95)

# Eventos extremos por precipitação
eventos_extremos_precip = df[df['RAINFALL'] >= limiar_precip_p95].copy()

# Eventos extremos por sedimentação
eventos_extremos_sed = df[df['FRACIONADO'] >= limiar_sed_p95].copy()

print(f"\n✓ Dados: {len(df)} registros")
print(f"✓ Limiar Precipitação P95: {limiar_precip_p95:.2f} mm")
print(f"✓ Limiar Sedimentação P95: {limiar_sed_p95:.4f} cm")
print(f"✓ Eventos extremos precipitação: {len(eventos_extremos_precip)}")
print(f"✓ Eventos extremos sedimentação: {len(eventos_extremos_sed)}")

# Criar figura
fig, ax1 = plt.subplots(figsize=(16, 8))

# Eixo 1: Precipitação (única linha para todas as áreas)
df_precip = df.drop_duplicates(subset=['DATA']).sort_values('DATA')
ax1.plot(df_precip['DATA'], df_precip['RAINFALL'], '--o', color='steelblue', 
         linewidth=2.5, markersize=7, label='Precipitação Mensal', 
         alpha=0.8, markeredgecolor='navy', markeredgewidth=0.8, dashes=(5, 3))

# Destacar eventos extremos de precipitação (estrelas vermelhas)
eventos_extremos_precip_unicos = eventos_extremos_precip.drop_duplicates(subset=['DATA'])
ax1.scatter(eventos_extremos_precip_unicos['DATA'], eventos_extremos_precip_unicos['RAINFALL'], 
           color='crimson', s=350, marker='*', zorder=10, 
           label=f'Precipitação Extrema (≥P95: {limiar_precip_p95:.1f} mm)', 
           edgecolors='darkred', linewidth=2)

# Linha do limiar de precipitação
ax1.axhline(y=limiar_precip_p95, color='orangered', linestyle='--', linewidth=2.5, 
           alpha=0.6, label=f'Limiar P95 Precipitação')

# Configurações do eixo 1
ax1.set_xlabel('Data', fontweight='bold', fontsize=13)
ax1.set_ylabel('Precipitação Mensal (mm)', fontweight='bold', fontsize=13, color='steelblue')
ax1.tick_params(axis='y', labelcolor='steelblue', labelsize=11)
ax1.tick_params(axis='x', labelsize=11, rotation=45)
ax1.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)

# Eixo 2: Sedimentação (diferenciando por área)
ax2 = ax1.twinx()
cores_sed_areas = {'SUP': 'saddlebrown', 'MED': 'darkolivegreen', 'INF': 'indigo'}

for area in df['AREA'].unique():
    df_area = df[df['AREA'] == area].sort_values('DATA')
    ax2.plot(df_area['DATA'], df_area['FRACIONADO'], '-s', 
             color=cores_sed_areas.get(area, 'gray'), 
             linewidth=2.5, markersize=7, label=f'Sedimentação - {area}', 
             alpha=0.85, markeredgecolor='black', markeredgewidth=0.5)

# Destacar sedimentação extrema (diamantes dourados)
ax2.scatter(eventos_extremos_sed['DATA'], eventos_extremos_sed['FRACIONADO'],
           color='gold', s=280, marker='D', zorder=9,
           label=f'Sedimentação Extrema (≥P95: {limiar_sed_p95:.4f} cm)',
           edgecolors='darkgoldenrod', linewidth=2, alpha=0.95)

# Linha do limiar de sedimentação
ax2.axhline(y=limiar_sed_p95, color='goldenrod', linestyle=':', linewidth=2.5, 
           alpha=0.7, label=f'Limiar P95 Sedimentação')

# Configurações do eixo 2
ax2.set_ylabel('Sedimentação Incremental (cm/mês)', fontweight='bold', fontsize=13, color='saddlebrown')
ax2.tick_params(axis='y', labelcolor='saddlebrown', labelsize=11)

# Formatação do eixo x (datas)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

# Título e legendas
plt.title('Série Temporal: Precipitação e Sedimentação com Eventos Extremos em Destaque',
         fontsize=15, fontweight='bold', pad=20)

# Combinar legendas
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, 
          loc='upper left', fontsize=10, framealpha=0.95,
          edgecolor='black', fancybox=True, shadow=True)

# Adicionar estatísticas na figura
stats_text = f"""Estatísticas:
• Precipitação média: {df['RAINFALL'].mean():.1f} mm
• Precipitação máxima: {df['RAINFALL'].max():.1f} mm
• Sedimentação média: {df['FRACIONADO'].mean():.4f} cm/mês
• Eventos extremos precipitação: {len(eventos_extremos_precip)}
• Eventos extremos sedimentação: {len(eventos_extremos_sed)}
• Período: {df['DATA'].min().strftime('%b/%Y')} - {df['DATA'].max().strftime('%b/%Y')}"""

ax1.text(0.98, 0.58, stats_text, transform=ax1.transAxes,
        fontsize=9, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black'),
        family='monospace')

# Ajustes finais
plt.tight_layout()

# Salvar
plt.savefig(FIGURAS_DIR / "15_serie_temporal_eventos_extremos_destaque.png", 
           dpi=300, bbox_inches='tight')
print("\n✓ Figura salva: 15_serie_temporal_eventos_extremos_destaque.png")
plt.close()

print("\n" + "=" * 80)
print("CONCLUÍDO!")
print("=" * 80)
