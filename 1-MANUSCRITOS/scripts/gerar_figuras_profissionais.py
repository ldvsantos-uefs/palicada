"""
Figuras profissionais estilo Nature/Science para eficiência e capacidade
Design minimalista, cores sofisticadas, alta qualidade
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy import stats

# Configuração profissional estilo Nature
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial'],
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'axes.linewidth': 0.8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'legend.fontsize': 8,
    'legend.frameon': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

# Paleta de cores profissional (inspirada em Nature)
COLORS = {
    'SUP': '#E64B35',    # Vermelho coral
    'MED': '#4DBBD5',    # Azul turquesa
    'INF': '#00A087',    # Verde esmeralda
    'accent': '#3C5488', # Azul escuro
    'gray': '#7E7E7E'    # Cinza neutro
}

# ============================================================================
# CARREGAR DADOS
# ============================================================================
file_path = '2-DADOS/CLIMATOLOGIA_20ANOS/dados/dados_integrados_sedimentacao.csv'
df = pd.read_csv(file_path)
df['DATA'] = pd.to_datetime(df['DATA'])

efficiency_data = {
    'Segmento': ['SUP', 'MED', 'INF'],
    'Sedimentação Total (cm)': [0.480667, 0.288667, 0.505667],
    'Eficiência (×10⁻⁴ cm/mm)': [1.871901, 1.124179, 1.969260],
    'Contribuição (%)': [37.699346, 22.640523, 39.660131]
}
df_eff = pd.DataFrame(efficiency_data)

capacity_data = {
    'Segmento': ['SUP', 'MED', 'INF'],
    'Capacidade Máxima (cm)': [50.0, 76.0, 36.0],
    'Acumulado Atual (cm)': [0.48, 0.29, 0.51]
}
df_cap = pd.DataFrame(capacity_data)

# ============================================================================
# FIGURA 1: EFICIÊNCIA E CONTRIBUIÇÃO (2 painéis clean)
# ============================================================================
print("Gerando Figura 1: Eficiência por segmento (estilo profissional)...")

fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))

# Painel A: Contribuição com gráfico de barras horizontal elegante
ax = axes[0]
segments = df_eff['Segmento'].values
y_pos = np.arange(len(segments))
contributions = df_eff['Contribuição (%)'].values

bars = ax.barh(y_pos, contributions, height=0.6, 
               color=[COLORS[s] for s in segments], alpha=0.85)

ax.set_yticks(y_pos)
ax.set_yticklabels(segments)
ax.set_xlabel('Contribuição (%)', fontweight='bold')
ax.set_title('a', loc='left', fontweight='bold', fontsize=11)
ax.spines['left'].set_visible(False)
ax.tick_params(left=False)

# Adicionar valores
for i, (bar, val) in enumerate(zip(bars, contributions)):
    ax.text(val + 1, bar.get_y() + bar.get_height()/2, 
           f'{val:.1f}%', va='center', fontsize=8, fontweight='bold')

# Painel B: Eficiência vs Sedimentação (scatter elegante)
ax = axes[1]
x = df_eff['Sedimentação Total (cm)'].values
y = df_eff['Eficiência (×10⁻⁴ cm/mm)'].values

for i, seg in enumerate(segments):
    ax.scatter(x[i], y[i], s=200, color=COLORS[seg], alpha=0.85, 
              edgecolor='white', linewidth=2, zorder=3)
    ax.text(x[i], y[i] - 0.15, seg, ha='center', fontsize=8, fontweight='bold')

ax.set_xlabel('Sedimentação acumulada (cm)', fontweight='bold')
ax.set_ylabel('Eficiência (×10⁻⁴ cm/mm)', fontweight='bold')
ax.set_title('b', loc='left', fontweight='bold', fontsize=11)
ax.grid(alpha=0.2, linestyle='-', linewidth=0.5)

plt.tight_layout()
output1 = '1-MANUSCRITOS/1-CONTROLE_PLITOSSOLO/media/analises_estatisticas/fig_eficiencia_clean.png'
plt.savefig(output1, dpi=600, bbox_inches='tight', facecolor='white')
print(f"✓ Figura 1 salva: {output1}")
plt.close()

# ============================================================================
# FIGURA 2: PROJEÇÃO TEMPORAL DE PREENCHIMENTO (série temporal)
# ============================================================================
print("Gerando Figura 2: Projeção temporal de preenchimento...")

fig = plt.figure(figsize=(7, 5))
gs = GridSpec(3, 1, figure=fig, hspace=0.15, height_ratios=[1, 1, 1])

# Parâmetros para simulação temporal
anos_simulacao = 100
meses_por_ano = 12
total_meses = anos_simulacao * meses_por_ano

# Taxa de deposição mensal média (cm/mês) - calculada dos dados
taxa_deposicao = {
    'SUP': 0.48 / 24,  # total acumulado / número de meses
    'MED': 0.29 / 24,
    'INF': 0.51 / 24
}

# Simular preenchimento ao longo do tempo para cada segmento
for idx, (segment, cap_max) in enumerate(zip(['SUP', 'MED', 'INF'], 
                                              [50.0, 76.0, 36.0])):
    ax = fig.add_subplot(gs[idx])
    
    # Criar série temporal
    meses = np.arange(0, total_meses + 1)
    anos = meses / meses_por_ano
    
    # Acumulação linear até capacidade máxima
    acumulado = np.minimum(meses * taxa_deposicao[segment], cap_max)
    
    # Encontrar ponto de saturação
    idx_saturacao = np.argmax(acumulado >= cap_max * 0.99)
    anos_saturacao = anos[idx_saturacao]
    
    # Plot da projeção
    ax.fill_between(anos, 0, acumulado, color=COLORS[segment], alpha=0.3)
    ax.plot(anos, acumulado, color=COLORS[segment], linewidth=2, label=segment)
    
    # Linha de capacidade máxima
    ax.axhline(cap_max, color='black', linestyle='--', linewidth=1, alpha=0.4)
    ax.text(anos_simulacao * 0.98, cap_max * 1.02, 
           f'Cap. máx. = {cap_max:.0f} cm', 
           ha='right', fontsize=7, color='black', alpha=0.7)
    
    # Marcar ponto atual (2 anos)
    atual_anos = 2
    atual_acum = df_cap[df_cap['Segmento'] == segment]['Acumulado Atual (cm)'].values[0]
    ax.scatter(atual_anos, atual_acum, s=80, color=COLORS[segment], 
              edgecolor='white', linewidth=2, zorder=5, marker='o')
    ax.annotate('Atual\n(2023-2025)', xy=(atual_anos, atual_acum), 
               xytext=(atual_anos + 5, atual_acum + cap_max*0.15),
               fontsize=7, ha='left',
               arrowprops=dict(arrowstyle='->', color='black', lw=1, alpha=0.6))
    
    # Marcar ponto de saturação
    if anos_saturacao < anos_simulacao:
        ax.axvline(anos_saturacao, color=COLORS[segment], linestyle=':', 
                  linewidth=1.5, alpha=0.5)
        ax.text(anos_saturacao, cap_max * 0.5, 
               f'{anos_saturacao:.0f} anos\n(saturação)', 
               rotation=90, va='center', ha='right', fontsize=7,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                        edgecolor=COLORS[segment], alpha=0.8))
    
    # Configuração dos eixos
    ax.set_xlim(0, anos_simulacao)
    ax.set_ylim(0, cap_max * 1.15)
    
    if idx == 2:  # Último painel
        ax.set_xlabel('Tempo (anos)', fontweight='bold')
    else:
        ax.set_xticklabels([])
    
    ax.set_ylabel('Altura (cm)', fontweight='bold')
    ax.text(0.02, 0.95, f'{chr(97+idx)}  {segment}', 
           transform=ax.transAxes, fontsize=10, fontweight='bold',
           va='top', bbox=dict(boxstyle='round,pad=0.4', 
                              facecolor='white', edgecolor=COLORS[segment], lw=2))
    
    ax.grid(alpha=0.15, linestyle='-', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

output2 = '1-MANUSCRITOS/1-CONTROLE_PLITOSSOLO/media/analises_estatisticas/fig_projecao_temporal.png'
plt.savefig(output2, dpi=600, bbox_inches='tight', facecolor='white')
print(f"✓ Figura 2 salva: {output2}")
plt.close()

# ============================================================================
# FIGURA 3: RELAÇÃO PRECIPITAÇÃO-SEDIMENTAÇÃO (3 painéis minimalistas)
# ============================================================================
print("Gerando Figura 3: Precipitação vs Sedimentação...")

fig, axes = plt.subplots(1, 3, figsize=(7, 2.3), sharey=True)

for i, segment in enumerate(['SUP', 'MED', 'INF']):
    ax = axes[i]
    df_seg = df[df['AREA'] == segment].copy()
    
    # Scatter plot minimalista
    ax.scatter(df_seg['RAINFALL'], df_seg['FRACIONADO'], 
              s=30, color=COLORS[segment], alpha=0.6, edgecolor='none')
    
    # Regressão linear
    if len(df_seg) > 1:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            df_seg['RAINFALL'], df_seg['FRACIONADO'])
        x_line = np.linspace(df_seg['RAINFALL'].min(), df_seg['RAINFALL'].max(), 100)
        ax.plot(x_line, slope * x_line + intercept, 
               color=COLORS[segment], linewidth=1.5, linestyle='-', alpha=0.8)
        
        # R² em caixa elegante
        r2_text = f'$R^2$ = {r_value**2:.2f}'
        if p_value < 0.001:
            sig = '***'
        elif p_value < 0.01:
            sig = '**'
        elif p_value < 0.05:
            sig = '*'
        else:
            sig = 'ns'
        
        ax.text(0.05, 0.95, f'{r2_text}\n({sig})', 
               transform=ax.transAxes, fontsize=7,
               va='top', ha='left',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                        edgecolor=COLORS[segment], alpha=0.9, lw=1.5))
    
    # Configuração
    ax.set_xlabel('Precipitação (mm)', fontweight='bold')
    if i == 0:
        ax.set_ylabel('Sedimentação\nincremental (cm)', fontweight='bold')
    
    ax.set_title(f'{chr(97+i)}  {segment}', loc='left', 
                fontweight='bold', fontsize=10)
    ax.grid(alpha=0.15, linestyle='-', linewidth=0.5)
    ax.axhline(0, color='black', linewidth=0.8, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
output3 = '1-MANUSCRITOS/1-CONTROLE_PLITOSSOLO/media/analises_estatisticas/fig_precipitacao_sedimentacao.png'
plt.savefig(output3, dpi=600, bbox_inches='tight', facecolor='white')
print(f"✓ Figura 3 salva: {output3}")
plt.close()

# ============================================================================
# FIGURA 4: CAPACIDADE ATUAL vs DISPONÍVEL (visualização elegante)
# ============================================================================
print("Gerando Figura 4: Status de capacidade...")

fig, ax = plt.subplots(1, 1, figsize=(7, 3))

segments = ['SUP', 'MED', 'INF']
capacidades = [50.0, 76.0, 36.0]
ocupados = [0.48, 0.29, 0.51]
percentuais = [(o/c)*100 for o, c in zip(ocupados, capacidades)]

x = np.arange(len(segments))
width = 0.6

# Barras de capacidade total (cinza claro)
bars_total = ax.bar(x, capacidades, width, color='#E0E0E0', 
                    edgecolor='black', linewidth=1, alpha=0.4, 
                    label='Capacidade máxima')

# Barras de ocupado (cores dos segmentos)
bars_ocupado = ax.bar(x, ocupados, width, 
                      color=[COLORS[s] for s in segments], 
                      edgecolor='white', linewidth=2, alpha=0.9,
                      label='Ocupado (2023-2025)')

# Adicionar percentuais
for i, (cap, ocup, pct) in enumerate(zip(capacidades, ocupados, percentuais)):
    # Valor absoluto no topo da barra total
    ax.text(i, cap + 2, f'{cap:.0f} cm', ha='center', fontsize=8, 
           fontweight='bold', color='black')
    
    # Percentual dentro da barra ocupada (se visível)
    if ocup > 2:
        ax.text(i, ocup/2, f'{pct:.1f}%', ha='center', va='center', 
               fontsize=8, fontweight='bold', color='white')
    else:
        ax.text(i, ocup + 1, f'{pct:.1f}%', ha='center', fontsize=7, 
               fontweight='bold', color=COLORS[segments[i]])

# Configuração
ax.set_ylabel('Altura de sedimentação (cm)', fontweight='bold')
ax.set_xlabel('Segmento', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(segments, fontweight='bold')
ax.set_ylim(0, max(capacidades) * 1.15)
ax.legend(loc='upper left', frameon=True, fancybox=False, 
         edgecolor='black', framealpha=0.9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.15, linestyle='-', linewidth=0.5)

output4 = '1-MANUSCRITOS/1-CONTROLE_PLITOSSOLO/media/analises_estatisticas/fig_capacidade_status.png'
plt.savefig(output4, dpi=600, bbox_inches='tight', facecolor='white')
print(f"✓ Figura 4 salva: {output4}")
plt.close()

# ============================================================================
# FIGURA 5: EVOLUÇÃO TEMPORAL OBSERVADA (série de linha clean)
# ============================================================================
print("Gerando Figura 5: Série temporal observada...")

fig, ax = plt.subplots(1, 1, figsize=(7, 3))

for segment in ['SUP', 'MED', 'INF']:
    df_seg = df[df['AREA'] == segment].sort_values('DATA')
    ax.plot(df_seg['DATA'], df_seg['SEDIMENT'], 
           marker='o', markersize=4, linewidth=1.5, 
           color=COLORS[segment], label=segment, alpha=0.85,
           markeredgecolor='white', markeredgewidth=0.5)

ax.set_ylabel('Sedimentação acumulada (cm)', fontweight='bold')
ax.set_xlabel('Data', fontweight='bold')
ax.legend(loc='upper left', ncol=3, frameon=True, fancybox=False,
         edgecolor='black', framealpha=0.9)
ax.grid(alpha=0.15, linestyle='-', linewidth=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Formatar eixo x
import matplotlib.dates as mdates
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b/%y'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45, ha='right')

output5 = '1-MANUSCRITOS/1-CONTROLE_PLITOSSOLO/media/analises_estatisticas/fig_serie_temporal.png'
plt.savefig(output5, dpi=600, bbox_inches='tight', facecolor='white')
print(f"✓ Figura 5 salva: {output5}")
plt.close()

print("\n" + "="*70)
print("FIGURAS PROFISSIONAIS GERADAS COM SUCESSO!")
print("="*70)
print(f"\n1. Eficiência e contribuição: {output1}")
print(f"2. Projeção temporal de preenchimento: {output2}")
print(f"3. Precipitação vs Sedimentação: {output3}")
print(f"4. Status de capacidade: {output4}")
print(f"5. Série temporal observada: {output5}")
print("\nEstilo: Clean, minimalista, alta resolução (600 dpi)")
print("Paleta: Inspirada em Nature/Science")
