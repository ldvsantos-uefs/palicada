"""
FIGURA COMPARATIVA: EVENTOS EXTREMOS - ACUMULADA vs INCREMENTAL
Painel superior: Série temporal comparativa
Painel inferior: Eventos extremos em ambas as escalas
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
print("EVENTOS EXTREMOS: ACUMULADA vs INCREMENTAL")
print("=" * 80)

# Carregar dados
df = pd.read_csv(DADOS_DIR / "dados_integrados_sedimentacao.csv")
df['DATA'] = pd.to_datetime(df['DATA'])

# Classificar eventos extremos (P95)
limiar_precip = df['RAINFALL'].quantile(0.95)
limiar_acumulada = df['SEDIMENT'].quantile(0.95)
limiar_incremental = df['FRACIONADO'].quantile(0.95)

# Identificar eventos extremos
eventos_precip = df[df['RAINFALL'] >= limiar_precip].copy()
eventos_acumulada = df[df['SEDIMENT'] >= limiar_acumulada].copy()
eventos_incremental = df[df['FRACIONADO'] >= limiar_incremental].copy()

print(f"\n✓ Dados: {len(df)} registros")
print(f"✓ Limiar Precipitação P95: {limiar_precip:.2f} mm")
print(f"✓ Limiar Acumulada P95: {limiar_acumulada:.4f} cm")
print(f"✓ Limiar Incremental P95: {limiar_incremental:.4f} cm")
print(f"✓ Eventos precipitação extrema: {len(eventos_precip)}")
print(f"✓ Eventos acumulada extrema: {len(eventos_acumulada)}")
print(f"✓ Eventos incremental extrema: {len(eventos_incremental)}")

# =============================================================================
# FIGURA COMPARATIVA
# =============================================================================
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.35)

# ============== PAINEL SUPERIOR: SÉRIE TEMPORAL COMPARATIVA ==============
ax1 = fig.add_subplot(gs[0])

# Cores por área
cores_areas = {'SUP': 'saddlebrown', 'MED': 'darkolivegreen', 'INF': 'indigo'}

# Eixo 1: Precipitação (tracejada)
df_precip = df.drop_duplicates(subset=['DATA']).sort_values('DATA')
ax1.plot(df_precip['DATA'], df_precip['RAINFALL'], '--o', color='steelblue', 
         linewidth=2.5, markersize=7, label='Precipitação Mensal', 
         alpha=0.8, markeredgecolor='navy', markeredgewidth=0.8, dashes=(5, 3))

ax1.set_xlabel('Data', fontweight='bold', fontsize=12)
ax1.set_ylabel('Precipitação (mm)', fontweight='bold', fontsize=12, color='steelblue')
ax1.tick_params(axis='y', labelcolor='steelblue', labelsize=11)
ax1.tick_params(axis='x', labelsize=11)
ax1.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)

# Eixo 2: Sedimentação ACUMULADA
ax2 = ax1.twinx()
for area in df['AREA'].unique():
    df_area = df[df['AREA'] == area].sort_values('DATA')
    ax2.plot(df_area['DATA'], df_area['SEDIMENT'], '-o', 
             color=cores_areas.get(area, 'gray'), 
             linewidth=2, markersize=5, label=f'Acumulada - {area}', 
             alpha=0.7, markeredgecolor='black', markeredgewidth=0.5)

ax2.set_ylabel('Sedimentação Acumulada (cm)', fontweight='bold', fontsize=12, color='red')
ax2.tick_params(axis='y', labelcolor='red', labelsize=11)

# Eixo 3: Sedimentação INCREMENTAL
ax3 = ax1.twinx()
ax3.spines['right'].set_position(('outward', 70))
for area in df['AREA'].unique():
    df_area = df[df['AREA'] == area].sort_values('DATA')
    ax3.plot(df_area['DATA'], df_area['FRACIONADO'], '-s', 
             color=cores_areas.get(area, 'gray'), 
             linewidth=2, markersize=6, label=f'Incremental - {area}', 
             alpha=0.85, markeredgecolor='black', markeredgewidth=0.5,
             linestyle='--')

ax3.set_ylabel('Sedimentação Incremental (cm/mês)', fontweight='bold', fontsize=12, color='green')
ax3.tick_params(axis='y', labelcolor='green', labelsize=11)

# Formatação do eixo x
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

# Título
ax1.set_title('Série Temporal Comparativa: Precipitação × Sedimentação (Acumulada e Incremental)',
             fontsize=13, fontweight='bold', pad=15)

# Legendas combinadas
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines3, labels3 = ax3.get_legend_handles_labels()
ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3,
          loc='upper left', fontsize=9, ncol=2, framealpha=0.95,
          edgecolor='black')

# ============== PAINEL INFERIOR: EVENTOS EXTREMOS ==============
ax4 = fig.add_subplot(gs[1])

# Precipitação base
df_precip_unicos = df.drop_duplicates(subset=['DATA']).sort_values('DATA')
ax4.plot(df_precip_unicos['DATA'], df_precip_unicos['RAINFALL'], '--o', 
         color='lightsteelblue', linewidth=2, markersize=5, 
         label='Precipitação', alpha=0.5, dashes=(5, 3))

# Eventos extremos de precipitação
eventos_precip_unicos = eventos_precip.drop_duplicates(subset=['DATA'])
ax4.scatter(eventos_precip_unicos['DATA'], eventos_precip_unicos['RAINFALL'], 
           color='crimson', s=400, marker='*', zorder=10, 
           label=f'Precipitação Extrema (≥{limiar_precip:.1f} mm)', 
           edgecolors='darkred', linewidth=2)

# Limiar de precipitação
ax4.axhline(y=limiar_precip, color='orangered', linestyle='--', linewidth=2, 
           alpha=0.6, label='Limiar P95 Precipitação')

ax4.set_xlabel('Data', fontweight='bold', fontsize=12)
ax4.set_ylabel('Precipitação (mm)', fontweight='bold', fontsize=12, color='steelblue')
ax4.tick_params(axis='y', labelcolor='steelblue', labelsize=11)
ax4.tick_params(axis='x', labelsize=11)
ax4.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)

# Eixo secundário: Sedimentação ACUMULADA
ax5 = ax4.twinx()
ax5.plot(df['DATA'], df['SEDIMENT'], '-', color='lightcoral', 
         linewidth=1.5, alpha=0.4, label='Sed. Acumulada')

# Eventos extremos ACUMULADA
ax5.scatter(eventos_acumulada['DATA'], eventos_acumulada['SEDIMENT'],
           color='red', s=250, marker='o', zorder=9,
           label=f'Acumulada Extrema (≥{limiar_acumulada:.3f} cm)',
           edgecolors='darkred', linewidth=2, alpha=0.9)

# Limiar acumulada
ax5.axhline(y=limiar_acumulada, color='red', linestyle=':', linewidth=2, 
           alpha=0.6, label='Limiar P95 Acumulada')

ax5.set_ylabel('Sedimentação Acumulada (cm)', fontweight='bold', fontsize=12, color='red')
ax5.tick_params(axis='y', labelcolor='red', labelsize=11)

# Eixo terciário: Sedimentação INCREMENTAL
ax6 = ax4.twinx()
ax6.spines['right'].set_position(('outward', 70))
ax6.plot(df['DATA'], df['FRACIONADO'], '--', color='lightgreen', 
         linewidth=1.5, alpha=0.4, label='Sed. Incremental')

# Eventos extremos INCREMENTAL
ax6.scatter(eventos_incremental['DATA'], eventos_incremental['FRACIONADO'],
           color='limegreen', s=250, marker='D', zorder=8,
           label=f'Incremental Extrema (≥{limiar_incremental:.4f} cm)',
           edgecolors='darkgreen', linewidth=2, alpha=0.9)

# Limiar incremental
ax6.axhline(y=limiar_incremental, color='green', linestyle='-.', linewidth=2, 
           alpha=0.6, label='Limiar P95 Incremental')

ax6.set_ylabel('Sedimentação Incremental (cm/mês)', fontweight='bold', fontsize=12, color='green')
ax6.tick_params(axis='y', labelcolor='green', labelsize=11)

# Formatação do eixo x
ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

# Título
ax4.set_title('Eventos Extremos: Comparação entre Acumulada e Incremental',
             fontsize=13, fontweight='bold', pad=15)

# Legendas combinadas
lines4, labels4 = ax4.get_legend_handles_labels()
lines5, labels5 = ax5.get_legend_handles_labels()
lines6, labels6 = ax6.get_legend_handles_labels()
ax4.legend(lines4 + lines5 + lines6, labels4 + labels5 + labels6,
          loc='upper left', fontsize=9, ncol=2, framealpha=0.95,
          edgecolor='black')

# Adicionar estatísticas
stats_text = f"""Estatísticas de Eventos Extremos (P95):
• Precipitação: {len(eventos_precip)} eventos (≥{limiar_precip:.1f} mm)
• Acumulada: {len(eventos_acumulada)} eventos (≥{limiar_acumulada:.3f} cm)
• Incremental: {len(eventos_incremental)} eventos (≥{limiar_incremental:.4f} cm)

Interpretação:
• Acumulada: reflete tendência cumulativa (memória do sistema)
• Incremental: detecta eventos isolados (resposta imediata)"""

ax4.text(0.98, 0.45, stats_text, transform=ax4.transAxes,
        fontsize=8.5, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85, edgecolor='black'),
        family='monospace')

plt.suptitle('COMPARAÇÃO: EVENTOS EXTREMOS - ACUMULADA vs INCREMENTAL', 
            fontsize=15, fontweight='bold', y=0.995)

plt.savefig(FIGURAS_DIR / "17_eventos_extremos_acumulada_vs_incremental.png", 
           dpi=300, bbox_inches='tight')
print("\n✓ Figura salva: 17_eventos_extremos_acumulada_vs_incremental.png")
plt.close()

# =============================================================================
# RELATÓRIO
# =============================================================================
print("\n" + "=" * 80)
print("RELATÓRIO COMPARATIVO")
print("=" * 80)

relatorio = f"""
EVENTOS EXTREMOS: ACUMULADA vs INCREMENTAL
==========================================

LIMIARES P95:
  Precipitação:  {limiar_precip:.2f} mm
  Acumulada:     {limiar_acumulada:.4f} cm
  Incremental:   {limiar_incremental:.4f} cm

EVENTOS IDENTIFICADOS:
  Precipitação extrema:  {len(eventos_precip)} eventos
  Acumulada extrema:     {len(eventos_acumulada)} eventos
  Incremental extrema:   {len(eventos_incremental)} eventos

DIFERENÇAS CONCEITUAIS:

ACUMULADA:
  • Integral temporal de todos os eventos
  • Cresce monotonicamente
  • Eventos extremos = períodos de maior acúmulo total
  • Reflete perda histórica de solo
  
INCREMENTAL:
  • Taxa mensal (cm/mês)
  • Pode subir e descer
  • Eventos extremos = picos de erosão específicos
  • Reflete resposta ao evento do mês

IMPLICAÇÕES PARA INTERPRETAÇÃO:
  
  → Evento extremo ACUMULADA: maior sedimentação acumulada até aquele ponto
  → Evento extremo INCREMENTAL: maior taxa de erosão naquele mês específico
  
  ⚠️  Um mês pode ter sedimentação incremental baixa mas acumulada alta
      (acúmulo de eventos anteriores)
  
  ⚠️  Um mês pode ter sedimentação incremental alta mas acumulada baixa
      (evento isolado no início do monitoramento)

RECOMENDAÇÃO:
  Para análise de EVENTOS e correlação com chuva mensal: USE INCREMENTAL
  Para avaliação de PERDA TOTAL de solo: USE ACUMULADA
"""

print(relatorio)

with open(DADOS_DIR / "relatorio_eventos_acumulada_vs_incremental.txt", 'w', encoding='utf-8') as f:
    f.write(relatorio)

print("\n✓ Relatório salvo: relatorio_eventos_acumulada_vs_incremental.txt")

print("\n" + "=" * 80)
print("CONCLUÍDO!")
print("=" * 80)
