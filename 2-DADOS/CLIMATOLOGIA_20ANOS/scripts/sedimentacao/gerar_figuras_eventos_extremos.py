"""
ANÁLISE DETALHADA DE EVENTOS EXTREMOS COM SEDIMENTAÇÃO
Gera figuras específicas comparando eventos extremos de precipitação/EI30 
com os índices de sedimentação observados
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Configurações
plt.style.use('ggplot')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 300
sns.set_palette("husl")

# Diretórios
BASE_DIR = Path(__file__).parent.parent.parent
FIGURAS_DIR = BASE_DIR / "figuras" / "sedimentacao"
DADOS_DIR = BASE_DIR / "dados"

print("=" * 80)
print("ANÁLISE DE EVENTOS EXTREMOS × SEDIMENTAÇÃO")
print("=" * 80)

# Carregar dados integrados
df = pd.read_csv(DADOS_DIR / "dados_integrados_sedimentacao.csv")
df['DATA'] = pd.to_datetime(df['DATA'])

print(f"\n✓ Dados carregados: {df.shape[0]} registros")
print(f"Período: {df['DATA'].min().strftime('%Y-%m')} a {df['DATA'].max().strftime('%Y-%m')}")
print(f"Áreas: {', '.join(df['AREA'].unique())}")

# =============================================================================
# CLASSIFICAÇÃO DE EVENTOS
# =============================================================================
print("\n" + "-" * 80)
print("CLASSIFICAÇÃO DE EVENTOS POR INTENSIDADE")
print("-" * 80)

# Definir limiares percentis
limiares = {
    'Muito Baixo': 0,
    'Baixo': df['RAINFALL'].quantile(0.25),
    'Moderado': df['RAINFALL'].quantile(0.50),
    'Alto': df['RAINFALL'].quantile(0.75),
    'Muito Alto': df['RAINFALL'].quantile(0.90),
    'Extremo': df['RAINFALL'].quantile(0.95)
}

def classificar_evento(valor):
    if valor >= limiares['Extremo']:
        return 'Extremo'
    elif valor >= limiares['Muito Alto']:
        return 'Muito Alto'
    elif valor >= limiares['Alto']:
        return 'Alto'
    elif valor >= limiares['Moderado']:
        return 'Moderado'
    elif valor >= limiares['Baixo']:
        return 'Baixo'
    else:
        return 'Muito Baixo'

df['CLASSE_PRECIPITACAO'] = df['RAINFALL'].apply(classificar_evento)
df['CLASSE_EI30'] = df['EI30'].apply(classificar_evento)

print("\nLimiares de Precipitação:")
for classe, valor in limiares.items():
    print(f"  {classe:15s}: {valor:7.2f} mm")

# Estatísticas por classe
print("\nDistribuição de Eventos por Classe:")
dist_classes = df['CLASSE_PRECIPITACAO'].value_counts().sort_index()
for classe, count in dist_classes.items():
    print(f"  {classe:15s}: {count:3d} eventos ({count/len(df)*100:.1f}%)")

# =============================================================================
# FIGURA 1: PAINEL COMPARATIVO DETALHADO
# =============================================================================
print("\n" + "-" * 80)
print("GERANDO FIGURA 1: Painel Comparativo Eventos × Sedimentação")
print("-" * 80)

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Scatter Precipitação × Sedimentação com densidade
ax1 = fig.add_subplot(gs[0, 0])
for area in df['AREA'].unique():
    df_area = df[df['AREA'] == area]
    ax1.scatter(df_area['RAINFALL'], df_area['FRACIONADO'], 
               label=area, s=100, alpha=0.6, edgecolors='black', linewidth=0.5)

# Linha de tendência geral (remover NaN)
df_clean = df[['RAINFALL', 'FRACIONADO']].dropna()
if len(df_clean) > 1:
    X = df_clean['RAINFALL'].values.reshape(-1, 1)
    y = df_clean['FRACIONADO'].values
    reg = LinearRegression().fit(X, y)
    x_pred = np.linspace(df_clean['RAINFALL'].min(), df_clean['RAINFALL'].max(), 100)
    y_pred = reg.predict(x_pred.reshape(-1, 1))
    ax1.plot(x_pred, y_pred, 'r--', linewidth=2, alpha=0.8, label=f'Tendência (R²={reg.score(X, y):.3f})')

ax1.set_xlabel('Precipitação Mensal (mm)', fontweight='bold')
ax1.set_ylabel('Sedimentação Incremental (cm)', fontweight='bold')
ax1.set_title('Precipitação × Sedimentação', fontweight='bold', fontsize=12)
ax1.legend(loc='upper left', fontsize=9)
ax1.grid(True, alpha=0.3)

# 2. Scatter EI30 × Sedimentação
ax2 = fig.add_subplot(gs[0, 1])
for area in df['AREA'].unique():
    df_area = df[df['AREA'] == area]
    ax2.scatter(df_area['EI30'], df_area['FRACIONADO'], 
               label=area, s=100, alpha=0.6, edgecolors='black', linewidth=0.5)

df_clean_ei30 = df[['EI30', 'FRACIONADO']].dropna()
if len(df_clean_ei30) > 1:
    X_ei30 = df_clean_ei30['EI30'].values.reshape(-1, 1)
    y_ei30 = df_clean_ei30['FRACIONADO'].values
    reg_ei30 = LinearRegression().fit(X_ei30, y_ei30)
    x_pred_ei30 = np.linspace(df_clean_ei30['EI30'].min(), df_clean_ei30['EI30'].max(), 100)
    y_pred_ei30 = reg_ei30.predict(x_pred_ei30.reshape(-1, 1))
    ax2.plot(x_pred_ei30, y_pred_ei30, 'r--', linewidth=2, alpha=0.8, 
             label=f'Tendência (R²={reg_ei30.score(X_ei30, y_ei30):.3f})')

ax2.set_xlabel('EI30 (MJ·mm/ha·h)', fontweight='bold')
ax2.set_ylabel('Sedimentação Incremental (cm)', fontweight='bold')
ax2.set_title('Erosividade (EI30) × Sedimentação', fontweight='bold', fontsize=12)
ax2.legend(loc='upper left', fontsize=9)
ax2.grid(True, alpha=0.3)

# 3. Box plot: Sedimentação por Classe de Evento
ax3 = fig.add_subplot(gs[0, 2])
ordem_classes = ['Muito Baixo', 'Baixo', 'Moderado', 'Alto', 'Muito Alto', 'Extremo']
classes_presentes = [c for c in ordem_classes if c in df['CLASSE_PRECIPITACAO'].values]
dados_box = [df[df['CLASSE_PRECIPITACAO'] == c]['FRACIONADO'].dropna() for c in classes_presentes]

bp = ax3.boxplot(dados_box, labels=classes_presentes, patch_artist=True)
colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(classes_presentes)))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

ax3.set_ylabel('Sedimentação (cm)', fontweight='bold')
ax3.set_title('Sedimentação por Classe de Evento', fontweight='bold', fontsize=12)
ax3.tick_params(axis='x', rotation=45)
ax3.grid(True, alpha=0.3, axis='y')

# 4. Série temporal com eventos extremos destacados
ax4 = fig.add_subplot(gs[1, :])
eventos_extremos = df[df['CLASSE_PRECIPITACAO'].isin(['Extremo', 'Muito Alto'])]

ax4.plot(df['DATA'], df['RAINFALL'], '-o', color='steelblue', 
         linewidth=1.5, markersize=4, label='Precipitação', alpha=0.7)
ax4.scatter(eventos_extremos['DATA'], eventos_extremos['RAINFALL'], 
           color='red', s=200, marker='*', zorder=5, 
           label=f'Eventos Extremos (n={len(eventos_extremos)})', edgecolors='black')

ax4_twin = ax4.twinx()
ax4_twin.plot(df['DATA'], df['FRACIONADO'], '-s', color='brown', 
             linewidth=2, markersize=5, label='Sedimentação', alpha=0.8)

ax4.set_xlabel('Data', fontweight='bold')
ax4.set_ylabel('Precipitação (mm)', fontweight='bold', color='steelblue')
ax4_twin.set_ylabel('Sedimentação (cm)', fontweight='bold', color='brown')
ax4.set_title('Série Temporal: Precipitação e Sedimentação com Eventos Extremos', 
             fontweight='bold', fontsize=12)
ax4.tick_params(axis='y', labelcolor='steelblue')
ax4_twin.tick_params(axis='y', labelcolor='brown')
ax4.legend(loc='upper left')
ax4_twin.legend(loc='upper right')
ax4.grid(True, alpha=0.3)

# 5. Correlação entre índices (heatmap)
ax5 = fig.add_subplot(gs[2, 0])
correlacao = df[['RAINFALL', 'EI30', 'FRACIONADO', 'SEDIMENT']].corr()
sns.heatmap(correlacao, annot=True, fmt='.3f', cmap='coolwarm', center=0,
           square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax5)
ax5.set_title('Matriz de Correlação', fontweight='bold', fontsize=12)

# 6. Distribuição de EI30
ax6 = fig.add_subplot(gs[2, 1])
ax6.hist(df['EI30'], bins=20, color='orange', alpha=0.7, edgecolor='black')
ax6.axvline(df['EI30'].mean(), color='red', linestyle='--', linewidth=2, 
           label=f'Média: {df["EI30"].mean():.1f}')
ax6.axvline(df['EI30'].quantile(0.9), color='darkred', linestyle='--', linewidth=2,
           label=f'P90: {df["EI30"].quantile(0.9):.1f}')
ax6.set_xlabel('EI30 (MJ·mm/ha·h)', fontweight='bold')
ax6.set_ylabel('Frequência', fontweight='bold')
ax6.set_title('Distribuição de Erosividade', fontweight='bold', fontsize=12)
ax6.legend()
ax6.grid(True, alpha=0.3)

# 7. Eficiência erosiva (Sedimentação / Precipitação)
ax7 = fig.add_subplot(gs[2, 2])
df['EFICIENCIA_EROSIVA'] = df['FRACIONADO'] / (df['RAINFALL'] + 0.01)  # Evitar divisão por zero

for area in df['AREA'].unique():
    df_area = df[df['AREA'] == area]
    ax7.scatter(df_area['EI30'], df_area['EFICIENCIA_EROSIVA'],
               label=area, s=80, alpha=0.6, edgecolors='black', linewidth=0.5)

ax7.set_xlabel('EI30 (MJ·mm/ha·h)', fontweight='bold')
ax7.set_ylabel('Eficiência Erosiva (cm/mm)', fontweight='bold')
ax7.set_title('Eficiência Erosiva × EI30', fontweight='bold', fontsize=12)
ax7.legend()
ax7.grid(True, alpha=0.3)

plt.suptitle('ANÁLISE COMPARATIVA: EVENTOS EXTREMOS × SEDIMENTAÇÃO', 
            fontsize=16, fontweight='bold', y=0.995)

plt.savefig(FIGURAS_DIR / "12_painel_eventos_extremos_sedimentacao.png", 
           dpi=300, bbox_inches='tight')
print("✓ Figura salva: 12_painel_eventos_extremos_sedimentacao.png")
plt.close()

# =============================================================================
# FIGURA 2: ANÁLISE DE EVENTOS EXTREMOS ESPECÍFICOS
# =============================================================================
print("\n" + "-" * 80)
print("GERANDO FIGURA 2: Análise Específica de Eventos Extremos")
print("-" * 80)

eventos_ext = df[df['CLASSE_PRECIPITACAO'] == 'Extremo'].copy()
print(f"\nEventos extremos identificados: {len(eventos_ext)}")

if len(eventos_ext) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Comparação eventos extremos vs normais
    df_normal = df[~df['CLASSE_PRECIPITACAO'].isin(['Extremo', 'Muito Alto'])]
    
    dados_comp = [
        df_normal['FRACIONADO'].dropna(),
        eventos_ext['FRACIONADO'].dropna()
    ]
    
    bp1 = axes[0, 0].boxplot(dados_comp, labels=['Normal', 'Extremo'], patch_artist=True)
    bp1['boxes'][0].set_facecolor('lightblue')
    bp1['boxes'][1].set_facecolor('red')
    axes[0, 0].set_ylabel('Sedimentação (cm)', fontweight='bold')
    axes[0, 0].set_title('Sedimentação: Normal vs Extremo', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Teste estatístico
    if len(dados_comp[0]) > 0 and len(dados_comp[1]) > 0:
        stat, p_value = stats.mannwhitneyu(dados_comp[0], dados_comp[1], alternative='two-sided')
        axes[0, 0].text(0.5, 0.95, f'Mann-Whitney U: p={p_value:.4f}',
                       transform=axes[0, 0].transAxes, ha='center', va='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 2. Precipitação vs EI30 para eventos extremos
    axes[0, 1].scatter(eventos_ext['RAINFALL'], eventos_ext['EI30'], 
                      s=eventos_ext['FRACIONADO']*1000, alpha=0.6,
                      c=eventos_ext['FRACIONADO'], cmap='Reds', 
                      edgecolors='black', linewidth=1)
    
    axes[0, 1].set_xlabel('Precipitação (mm)', fontweight='bold')
    axes[0, 1].set_ylabel('EI30 (MJ·mm/ha·h)', fontweight='bold')
    axes[0, 1].set_title('Eventos Extremos: Tamanho = Sedimentação', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap='Reds', 
                               norm=plt.Normalize(vmin=eventos_ext['FRACIONADO'].min(),
                                                 vmax=eventos_ext['FRACIONADO'].max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axes[0, 1])
    cbar.set_label('Sedimentação (cm)', fontweight='bold')
    
    # 3. Série temporal de eventos extremos
    axes[1, 0].stem(eventos_ext['DATA'], eventos_ext['FRACIONADO'], 
                   basefmt=' ', linefmt='red', markerfmt='ro')
    axes[1, 0].set_xlabel('Data', fontweight='bold')
    axes[1, 0].set_ylabel('Sedimentação (cm)', fontweight='bold')
    axes[1, 0].set_title('Sedimentação em Eventos Extremos', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. Tabela de top eventos
    axes[1, 1].axis('tight')
    axes[1, 1].axis('off')
    
    top_eventos = eventos_ext.nlargest(10, 'RAINFALL')[['DATA', 'RAINFALL', 'EI30', 'FRACIONADO', 'AREA']]
    top_eventos_display = top_eventos.copy()
    top_eventos_display['DATA'] = top_eventos_display['DATA'].dt.strftime('%Y-%m')
    top_eventos_display['RAINFALL'] = top_eventos_display['RAINFALL'].round(1)
    top_eventos_display['EI30'] = top_eventos_display['EI30'].round(1)
    top_eventos_display['FRACIONADO'] = top_eventos_display['FRACIONADO'].round(4)
    
    table = axes[1, 1].table(cellText=top_eventos_display.values,
                            colLabels=['Data', 'Precip\n(mm)', 'EI30', 'Sed\n(cm)', 'Área'],
                            cellLoc='center', loc='center',
                            colWidths=[0.15, 0.15, 0.2, 0.15, 0.1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Colorir header
    for i in range(5):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    axes[1, 1].set_title('Top 10 Eventos Extremos', fontweight='bold', pad=20)
    
    plt.suptitle('EVENTOS EXTREMOS: ANÁLISE DETALHADA', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(FIGURAS_DIR / "13_eventos_extremos_detalhado.png", 
               dpi=300, bbox_inches='tight')
    print("✓ Figura salva: 13_eventos_extremos_detalhado.png")
    plt.close()

# =============================================================================
# FIGURA 3: ANÁLISE POR CLASSE DE INTENSIDADE
# =============================================================================
print("\n" + "-" * 80)
print("GERANDO FIGURA 3: Análise por Classe de Intensidade")
print("-" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Estatísticas por classe
stats_classe = df.groupby('CLASSE_PRECIPITACAO').agg({
    'RAINFALL': ['mean', 'max', 'count'],
    'EI30': ['mean', 'max'],
    'FRACIONADO': ['mean', 'std', 'sum']
}).round(3)

print("\nEstatísticas por Classe:")
print(stats_classe)

# 1. Média de sedimentação por classe
ax1 = axes[0, 0]
media_sed = df.groupby('CLASSE_PRECIPITACAO')['FRACIONADO'].mean().reindex(classes_presentes)
bars1 = ax1.bar(range(len(classes_presentes)), media_sed.values, 
               color=plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(classes_presentes))),
               edgecolor='black', linewidth=1.5)
ax1.set_xticks(range(len(classes_presentes)))
ax1.set_xticklabels(classes_presentes, rotation=45, ha='right')
ax1.set_ylabel('Sedimentação Média (cm)', fontweight='bold')
ax1.set_title('Sedimentação Média por Classe', fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

# 2. EI30 médio por classe
ax3 = axes[0, 1]
media_ei30 = df.groupby('CLASSE_PRECIPITACAO')['EI30'].mean().reindex(classes_presentes)
bars3 = ax3.bar(range(len(classes_presentes)), media_ei30.values,
               color=plt.cm.Oranges(np.linspace(0.3, 0.9, len(classes_presentes))),
               edgecolor='black', linewidth=1.5)
ax3.set_xticks(range(len(classes_presentes)))
ax3.set_xticklabels(classes_presentes, rotation=45, ha='right')
ax3.set_ylabel('EI30 Médio (MJ·mm/ha·h)', fontweight='bold')
ax3.set_title('Erosividade Média por Classe', fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# 3. Relação Sedimentação/Precipitação por classe
ax5 = axes[1, 0]
eficiencia = df.groupby('CLASSE_PRECIPITACAO').apply(
    lambda x: (x['FRACIONADO'].sum() / x['RAINFALL'].sum()) if x['RAINFALL'].sum() > 0 else 0
).reindex(classes_presentes)

bars5 = ax5.bar(range(len(classes_presentes)), eficiencia.values,
               color=plt.cm.RdPu(np.linspace(0.3, 0.9, len(classes_presentes))),
               edgecolor='black', linewidth=1.5)
ax5.set_xticks(range(len(classes_presentes)))
ax5.set_xticklabels(classes_presentes, rotation=45, ha='right')
ax5.set_ylabel('Eficiência (cm/mm)', fontweight='bold')
ax5.set_title('Eficiência Erosiva por Classe', fontweight='bold')
ax5.grid(True, alpha=0.3, axis='y')

# 4. Contribuição % para sedimentação total
ax6 = axes[1, 1]
contrib = (df.groupby('CLASSE_PRECIPITACAO')['FRACIONADO'].sum() / 
           df['FRACIONADO'].sum() * 100).reindex(classes_presentes)

bars6 = ax6.bar(range(len(classes_presentes)), contrib.values,
               color=plt.cm.Spectral(np.linspace(0.2, 0.8, len(classes_presentes))),
               edgecolor='black', linewidth=1.5)
ax6.set_xticks(range(len(classes_presentes)))
ax6.set_xticklabels(classes_presentes, rotation=45, ha='right')
ax6.set_ylabel('Contribuição (%)', fontweight='bold')
ax6.set_title('Contribuição para Sedimentação Total', fontweight='bold')
ax6.grid(True, alpha=0.3, axis='y')

# Adicionar valores nas barras
for bar in bars6:
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)

plt.suptitle('ANÁLISE POR CLASSE DE INTENSIDADE', 
            fontsize=14, fontweight='bold')
plt.tight_layout()

plt.savefig(FIGURAS_DIR / "14_analise_por_classe_intensidade.png", 
           dpi=300, bbox_inches='tight')
print("✓ Figura salva: 14_analise_por_classe_intensidade.png")
plt.close()

# =============================================================================
# RELATÓRIO
# =============================================================================
print("\n" + "=" * 80)
print("RELATÓRIO - EVENTOS EXTREMOS × SEDIMENTAÇÃO")
print("=" * 80)

relatorio = f"""
ANÁLISE DE EVENTOS EXTREMOS E SEDIMENTAÇÃO
Data: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

CLASSIFICAÇÃO DE EVENTOS:
  Eventos Extremos (P95): {len(df[df['CLASSE_PRECIPITACAO']=='Extremo'])}
  Eventos Muito Altos (P90-P95): {len(df[df['CLASSE_PRECIPITACAO']=='Muito Alto'])}
  Eventos Altos (P75-P90): {len(df[df['CLASSE_PRECIPITACAO']=='Alto'])}
  Eventos Moderados (P50-P75): {len(df[df['CLASSE_PRECIPITACAO']=='Moderado'])}

LIMIARES:
  P95 (Extremo): {limiares['Extremo']:.2f} mm
  P90 (Muito Alto): {limiares['Muito Alto']:.2f} mm
  P75 (Alto): {limiares['Alto']:.2f} mm
  P50 (Moderado): {limiares['Moderado']:.2f} mm

SEDIMENTAÇÃO POR CLASSE:
  Extremo: {df[df['CLASSE_PRECIPITACAO']=='Extremo']['FRACIONADO'].mean():.4f} cm/mês
  Muito Alto: {df[df['CLASSE_PRECIPITACAO']=='Muito Alto']['FRACIONADO'].mean():.4f} cm/mês
  Alto: {df[df['CLASSE_PRECIPITACAO']=='Alto']['FRACIONADO'].mean():.4f} cm/mês
  Moderado: {df[df['CLASSE_PRECIPITACAO']=='Moderado']['FRACIONADO'].mean():.4f} cm/mês

CORRELAÇÕES:
  Precipitação × Sedimentação: r={df['RAINFALL'].corr(df['FRACIONADO']):.4f}
  EI30 × Sedimentação: r={df['EI30'].corr(df['FRACIONADO']):.4f}

FIGURAS GERADAS:
  - 12_painel_eventos_extremos_sedimentacao.png (7 painéis)
  - 13_eventos_extremos_detalhado.png (4 painéis)
    - 14_analise_por_classe_intensidade.png (4 painéis)
"""

print(relatorio)

with open(DADOS_DIR / "relatorio_eventos_extremos.txt", 'w', encoding='utf-8') as f:
    f.write(relatorio)

print("✓ Relatório salvo: relatorio_eventos_extremos.txt")

# Salvar classificações
df_export = df[['DATA', 'AREA', 'RAINFALL', 'EI30', 'FRACIONADO', 'SEDIMENT', 
                'CLASSE_PRECIPITACAO', 'CLASSE_EI30', 'EFICIENCIA_EROSIVA']].copy()
df_export.to_csv(DADOS_DIR / "dados_classificados_eventos.csv", index=False)
print("✓ Dados classificados salvos: dados_classificados_eventos.csv")

print("\n" + "=" * 80)
print("ANÁLISE CONCLUÍDA COM SUCESSO!")
print("=" * 80)
