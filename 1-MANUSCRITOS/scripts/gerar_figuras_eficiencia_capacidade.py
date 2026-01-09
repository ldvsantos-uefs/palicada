"""
Script para gerar figuras de eficiência por segmento e capacidade residual
Estilo: Publicação científica (Nature/Science style)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns

# Configurar estilo acadêmico
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.3)
sns.set_palette("Set2")

# Configurações globais
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13

# Cores por segmento (consistentes com literatura)
COLORS = {
    'SUP': '#2E7D32',  # Verde escuro
    'MED': '#1976D2',  # Azul
    'INF': '#D32F2F'   # Vermelho
}

# ============================================================================
# CARREGAR DADOS
# ============================================================================
file_path = '2-DADOS/CLIMATOLOGIA_20ANOS/dados/dados_integrados_sedimentacao.csv'
df = pd.read_csv(file_path)
df['DATA'] = pd.to_datetime(df['DATA'])

# Dados de eficiência (da análise 3)
efficiency_data = {
    'Segmento': ['SUP', 'MED', 'INF'],
    'Sedimentação Total (cm)': [0.480667, 0.288667, 0.505667],
    'Eficiência (×10⁻⁴ cm/mm)': [1.871901, 1.124179, 1.969260],
    'CV (%)': [234.807675, 468.754687, 162.061539],
    'R²': [0.021089, 0.007176, 0.150863],
    'Contribuição (%)': [37.699346, 22.640523, 39.660131]
}
df_eff = pd.DataFrame(efficiency_data)

# Dados de capacidade (da análise 4)
capacity_data = {
    'Segmento': ['SUP', 'MED', 'INF'],
    'Capacidade Máxima (cm)': [50.0, 76.0, 36.0],
    'Acumulado Atual (cm)': [0.48, 0.29, 0.51],
    'Ocupação (%)': [1.0, 0.4, 1.4]
}
df_cap = pd.DataFrame(capacity_data)
df_cap['Residual (cm)'] = df_cap['Capacidade Máxima (cm)'] - df_cap['Acumulado Atual (cm)']

# ============================================================================
# FIGURA 1: EFICIÊNCIA POR SEGMENTO (Painel múltiplo)
# ============================================================================
print("Gerando Figura 1: Eficiência por segmento...")

fig = plt.figure(figsize=(14, 10))
gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

# (a) Sedimentação total acumulada
ax1 = fig.add_subplot(gs[0, 0])
bars1 = ax1.bar(df_eff['Segmento'], df_eff['Sedimentação Total (cm)'], 
                color=[COLORS[s] for s in df_eff['Segmento']], 
                edgecolor='black', linewidth=1.2, alpha=0.85)
ax1.set_ylabel('Sedimentação Total (cm)', fontweight='bold')
ax1.set_xlabel('Segmento', fontweight='bold')
ax1.set_title('(a) Retenção acumulada', fontweight='bold', pad=10)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.set_ylim(0, max(df_eff['Sedimentação Total (cm)']) * 1.2)
# Adicionar valores
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

# (b) Eficiência de retenção
ax2 = fig.add_subplot(gs[0, 1])
bars2 = ax2.bar(df_eff['Segmento'], df_eff['Eficiência (×10⁻⁴ cm/mm)'], 
                color=[COLORS[s] for s in df_eff['Segmento']], 
                edgecolor='black', linewidth=1.2, alpha=0.85)
ax2.set_ylabel('Eficiência (×10⁻⁴ cm/mm)', fontweight='bold')
ax2.set_xlabel('Segmento', fontweight='bold')
ax2.set_title('(b) Eficiência de retenção', fontweight='bold', pad=10)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.set_ylim(0, max(df_eff['Eficiência (×10⁻⁴ cm/mm)']) * 1.2)
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

# (c) Contribuição relativa
ax3 = fig.add_subplot(gs[0, 2])
wedges, texts, autotexts = ax3.pie(df_eff['Contribuição (%)'], 
                                     labels=df_eff['Segmento'],
                                     colors=[COLORS[s] for s in df_eff['Segmento']],
                                     autopct='%1.1f%%',
                                     startangle=90,
                                     textprops={'fontsize': 10, 'fontweight': 'bold'},
                                     wedgeprops={'edgecolor': 'black', 'linewidth': 1.2})
ax3.set_title('(c) Contribuição relativa', fontweight='bold', pad=10)

# (d) Série temporal por segmento
ax4 = fig.add_subplot(gs[1, :])
for segment in ['SUP', 'MED', 'INF']:
    df_seg = df[df['AREA'] == segment].sort_values('DATA')
    ax4.plot(df_seg['DATA'], df_seg['SEDIMENT'], 
            marker='o', markersize=5, linewidth=2, 
            color=COLORS[segment], label=segment, alpha=0.85)

ax4.set_ylabel('Sedimentação Acumulada (cm)', fontweight='bold')
ax4.set_xlabel('Data', fontweight='bold')
ax4.set_title('(d) Evolução temporal da sedimentação', fontweight='bold', pad=10)
ax4.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
ax4.grid(alpha=0.3, linestyle='--')
ax4.tick_params(axis='x', rotation=45)

plt.tight_layout()
output_path1 = '1-MANUSCRITOS/1-CONTROLE_PLITOSSOLO/media/analises_estatisticas/eficiencia_por_segmento.png'
plt.savefig(output_path1, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Figura 1 salva: {output_path1}")
plt.close()

# ============================================================================
# FIGURA 2: CAPACIDADE E PROJEÇÃO (Painel duplo)
# ============================================================================
print("Gerando Figura 2: Capacidade residual e projeção...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# (a) Capacidade residual - Barras empilhadas
ax = axes[0]
width = 0.6
x = np.arange(len(df_cap['Segmento']))

bars1 = ax.bar(x, df_cap['Acumulado Atual (cm)'], width, 
              label='Ocupado', color='#EF5350', edgecolor='black', linewidth=1.2)
bars2 = ax.bar(x, df_cap['Residual (cm)'], width, bottom=df_cap['Acumulado Atual (cm)'],
              label='Disponível', color='#66BB6A', edgecolor='black', linewidth=1.2)

ax.set_ylabel('Altura (cm)', fontweight='bold')
ax.set_xlabel('Segmento', fontweight='bold')
ax.set_title('(a) Capacidade de retenção', fontweight='bold', pad=10)
ax.set_xticks(x)
ax.set_xticklabels(df_cap['Segmento'])
ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Adicionar linha de capacidade máxima
for i, (idx, row) in enumerate(df_cap.iterrows()):
    ax.plot([i-width/2, i+width/2], [row['Capacidade Máxima (cm)'], row['Capacidade Máxima (cm)']], 
           'k--', linewidth=2)
    ax.text(i, row['Capacidade Máxima (cm)'] + 2, 
           f"{row['Capacidade Máxima (cm)']:.0f} cm", 
           ha='center', fontsize=9, fontweight='bold')
    # Percentual de ocupação
    ax.text(i, row['Acumulado Atual (cm)']/2, 
           f"{row['Ocupação (%)']:.1f}%", 
           ha='center', va='center', fontsize=9, fontweight='bold', color='white')

# (b) Projeção de tempo até saturação
ax = axes[1]

# Dados de projeção (cenários P90 e P95)
proj_data = {
    'SUP': {'P90': 1466.0, 'P95': 1422.0},
    'MED': {'P90': 3497.9, 'P95': 3383.7},
    'INF': {'P90': 743.0, 'P95': 712.7}
}

x = np.arange(len(df_cap['Segmento']))
width = 0.35

bars1 = ax.bar(x - width/2, [proj_data[s]['P90'] for s in df_cap['Segmento']], 
              width, label='Cenário P90', color='#42A5F5', edgecolor='black', linewidth=1.2)
bars2 = ax.bar(x + width/2, [proj_data[s]['P95'] for s in df_cap['Segmento']], 
              width, label='Cenário P95', color='#FF7043', edgecolor='black', linewidth=1.2)

ax.set_ylabel('Tempo até saturação (meses)', fontweight='bold')
ax.set_xlabel('Segmento', fontweight='bold')
ax.set_title('(b) Projeção de longevidade operacional', fontweight='bold', pad=10)
ax.set_xticks(x)
ax.set_xticklabels(df_cap['Segmento'])
ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_yscale('log')

# Adicionar linha de referência (10 anos = 120 meses)
ax.axhline(y=120, color='red', linestyle='--', linewidth=2, alpha=0.7, label='10 anos')
ax.text(2.5, 150, '10 anos', ha='right', fontsize=9, color='red', fontweight='bold')

plt.tight_layout()
output_path2 = '1-MANUSCRITOS/1-CONTROLE_PLITOSSOLO/media/analises_estatisticas/capacidade_projecao.png'
plt.savefig(output_path2, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Figura 2 salva: {output_path2}")
plt.close()

# ============================================================================
# FIGURA 3: DISPERSÃO PRECIPITAÇÃO vs SEDIMENTAÇÃO (por segmento)
# ============================================================================
print("Gerando Figura 3: Relação precipitação-sedimentação...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

for i, segment in enumerate(['SUP', 'MED', 'INF']):
    ax = axes[i]
    df_seg = df[df['AREA'] == segment].copy()
    
    # Scatter plot
    ax.scatter(df_seg['RAINFALL'], df_seg['FRACIONADO'], 
              s=80, color=COLORS[segment], alpha=0.6, 
              edgecolor='black', linewidth=1)
    
    # Linha de regressão
    if len(df_seg) > 0:
        z = np.polyfit(df_seg['RAINFALL'], df_seg['FRACIONADO'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df_seg['RAINFALL'].min(), df_seg['RAINFALL'].max(), 100)
        ax.plot(x_line, p(x_line), color=COLORS[segment], 
               linewidth=2.5, linestyle='--', alpha=0.8)
        
        # R² annotation
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(df_seg['RAINFALL'], df_seg['FRACIONADO'])
        ax.text(0.05, 0.95, f'R² = {r_value**2:.3f}', 
               transform=ax.transAxes, fontsize=10, fontweight='bold',
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Precipitação (mm/mês)', fontweight='bold')
    if i == 0:
        ax.set_ylabel('Sedimentação Incremental (cm/mês)', fontweight='bold')
    ax.set_title(f'({chr(97+i)}) Segmento {segment}', fontweight='bold', pad=10)
    ax.grid(alpha=0.3, linestyle='--')

plt.tight_layout()
output_path3 = '1-MANUSCRITOS/1-CONTROLE_PLITOSSOLO/media/analises_estatisticas/precipitacao_sedimentacao_segmentos.png'
plt.savefig(output_path3, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Figura 3 salva: {output_path3}")
plt.close()

# ============================================================================
# FIGURA 4: BOXPLOT COMPARATIVO (Variabilidade por segmento)
# ============================================================================
print("Gerando Figura 4: Variabilidade da sedimentação...")

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# (a) Boxplot sedimentação incremental
ax = axes[0]
data_boxplot = [df[df['AREA'] == seg]['FRACIONADO'].values for seg in ['SUP', 'MED', 'INF']]
bp = ax.boxplot(data_boxplot, labels=['SUP', 'MED', 'INF'],
               patch_artist=True, widths=0.6,
               boxprops=dict(linewidth=1.5),
               whiskerprops=dict(linewidth=1.5),
               capprops=dict(linewidth=1.5),
               medianprops=dict(linewidth=2, color='red'))

for patch, segment in zip(bp['boxes'], ['SUP', 'MED', 'INF']):
    patch.set_facecolor(COLORS[segment])
    patch.set_alpha(0.7)

ax.set_ylabel('Sedimentação Incremental (cm/mês)', fontweight='bold')
ax.set_xlabel('Segmento', fontweight='bold')
ax.set_title('(a) Distribuição da sedimentação', fontweight='bold', pad=10)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

# (b) Coeficiente de variação
ax = axes[1]
bars = ax.bar(df_eff['Segmento'], df_eff['CV (%)'], 
             color=[COLORS[s] for s in df_eff['Segmento']], 
             edgecolor='black', linewidth=1.2, alpha=0.85)
ax.set_ylabel('Coeficiente de Variação (%)', fontweight='bold')
ax.set_xlabel('Segmento', fontweight='bold')
ax.set_title('(b) Variabilidade temporal', fontweight='bold', pad=10)
ax.grid(axis='y', alpha=0.3, linestyle='--')

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{height:.0f}%',
           ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
output_path4 = '1-MANUSCRITOS/1-CONTROLE_PLITOSSOLO/media/analises_estatisticas/variabilidade_segmentos.png'
plt.savefig(output_path4, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Figura 4 salva: {output_path4}")
plt.close()

print("\n" + "="*80)
print("TODAS AS FIGURAS FORAM GERADAS COM SUCESSO!")
print("="*80)
print(f"\n1. Eficiência por segmento: {output_path1}")
print(f"2. Capacidade e projeção: {output_path2}")
print(f"3. Precipitação vs Sedimentação: {output_path3}")
print(f"4. Variabilidade: {output_path4}")
