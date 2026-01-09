"""
ANÁLISE INTEGRADA: PRECIPITAÇÃO × EI30 × SEDIMENTAÇÃO REAL
Correlacionando índices de erosividade com medições de campo em ravina
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Configurações
plt.style.use('ggplot')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 300

# Diretórios
BASE_DIR = Path(__file__).parent.parent
FIGURAS_DIR = BASE_DIR / "figuras"
DADOS_DIR = BASE_DIR / "dados"
SEDIMENTOS_DIR = BASE_DIR / "sediments"

print("=" * 80)
print("ANÁLISE INTEGRADA: PRECIPITAÇÃO × EI30 × SEDIMENTAÇÃO REAL")
print("=" * 80)

# =============================================================================
# 1. CARREGAR DADOS DE SEDIMENTAÇÃO
# =============================================================================
print("\n1. Carregando dados de sedimentação...")
df_sed = pd.read_excel(SEDIMENTOS_DIR / "BD.xlsx", sheet_name="GRAFICO")

print(f"✓ Dados carregados: {df_sed.shape[0]} registros")
print(f"\nColunas: {df_sed.columns.tolist()}")
print(f"\nÁreas monitoradas: {df_sed['AREA'].unique()}")
print(f"Período: {df_sed['MONTH'].min()}/{df_sed['YEAR'].min()} a {df_sed['MONTH'].max()}/{df_sed['YEAR'].max()}")

# Criar coluna de data
df_sed['DATA'] = pd.to_datetime(df_sed[['YEAR', 'MONTH']].assign(DAY=1))

# Separar por área
areas = df_sed['AREA'].unique()
print(f"\n✓ Áreas identificadas: {', '.join(areas)}")

# =============================================================================
# 2. CALCULAR EI30 PARA O PERÍODO DE SEDIMENTAÇÃO
# =============================================================================
print("\n2. Calculando EI30 para o período de sedimentação...")

# Criar dados mensais fictícios de precipitação para o período (Jun/2023 - Mai/2025)
# Usando a precipitação mensal do arquivo de sedimentação
df_ei30 = df_sed[df_sed['AREA'] == 'SUP'][['DATA', 'RAINFALL']].copy()
df_ei30 = df_ei30.dropna(subset=['RAINFALL'])

# Calcular EI30 simplificado
# EI30 = 0.29 * (1 - 0.72 * exp(-0.05 * P)) * P * I30
# Assumindo I30 proporcional à precipitação mensal
df_ei30['EC'] = 0.29 * (1 - 0.72 * np.exp(-0.05 * df_ei30['RAINFALL']))
df_ei30['I30'] = df_ei30['RAINFALL'] / 30  # Intensidade média diária
df_ei30['EI30'] = df_ei30['EC'] * df_ei30['RAINFALL'] * df_ei30['I30']

print(f"✓ EI30 calculado para {len(df_ei30)} meses")
print(f"\nEI30 Médio: {df_ei30['EI30'].mean():.2f} MJ·mm/ha·h")
print(f"EI30 Máximo: {df_ei30['EI30'].max():.2f} MJ·mm/ha·h")

# =============================================================================
# 3. INTEGRAR DADOS
# =============================================================================
print("\n3. Integrando dados de precipitação, EI30 e sedimentação...")

# Mesclar dados por área
resultados = {}
for area in areas:
    df_area = df_sed[df_sed['AREA'] == area].copy()
    df_area = df_area.merge(df_ei30[['DATA', 'EI30']], on='DATA', how='left')
    resultados[area] = df_area

print(f"✓ Dados integrados para {len(areas)} áreas")

# =============================================================================
# 4. ANÁLISE DE CORRELAÇÃO
# =============================================================================
print("\n" + "=" * 80)
print("4. ANÁLISE DE CORRELAÇÃO")
print("=" * 80)

for area in areas:
    df_area = resultados[area]
    df_analise = df_area[['SEDIMENT', 'RAINFALL', 'EI30', 'FRACIONADO']].dropna()
    
    print(f"\n{'='*60}")
    print(f"ÁREA: {area}")
    print(f"{'='*60}")
    
    # Verificar se há dados suficientes
    if len(df_analise) < 3:
        print(f"\n⚠️  Dados insuficientes (n={len(df_analise)}). Pulando análise.")
        continue
    
    # Correlação de Pearson
    corr_precip_sed = stats.pearsonr(df_analise['RAINFALL'], df_analise['SEDIMENT'])
    corr_ei30_sed = stats.pearsonr(df_analise['EI30'], df_analise['SEDIMENT'])
    corr_frac_precip = stats.pearsonr(df_analise['RAINFALL'], df_analise['FRACIONADO'])
    
    print(f"\nCorrelações com Sedimentação Total:")
    print(f"  Precipitação: r={corr_precip_sed[0]:.4f}, p={corr_precip_sed[1]:.4f}")
    print(f"  EI30:         r={corr_ei30_sed[0]:.4f}, p={corr_ei30_sed[1]:.4f}")
    
    print(f"\nCorrelações com Sedimentação Incremental:")
    print(f"  Precipitação: r={corr_frac_precip[0]:.4f}, p={corr_frac_precip[1]:.4f}")
    
    # Regressão linear
    X_precip = df_analise['RAINFALL'].values.reshape(-1, 1)
    X_ei30 = df_analise['EI30'].values.reshape(-1, 1)
    y = df_analise['FRACIONADO'].values
    
    # Modelo 1: Precipitação → Sedimentação
    reg_precip = LinearRegression()
    reg_precip.fit(X_precip, y)
    y_pred_precip = reg_precip.predict(X_precip)
    r2_precip = r2_score(y, y_pred_precip)
    rmse_precip = np.sqrt(mean_squared_error(y, y_pred_precip))
    
    # Modelo 2: EI30 → Sedimentação
    reg_ei30 = LinearRegression()
    reg_ei30.fit(X_ei30, y)
    y_pred_ei30 = reg_ei30.predict(X_ei30)
    r2_ei30 = r2_score(y, y_pred_ei30)
    rmse_ei30 = np.sqrt(mean_squared_error(y, y_pred_ei30))
    
    print(f"\nModelo: Precipitação → Sedimentação")
    print(f"  R²: {r2_precip:.4f}")
    print(f"  RMSE: {rmse_precip:.6f}")
    print(f"  Equação: Sedimentação = {reg_precip.intercept_:.6f} + {reg_precip.coef_[0]:.6f} × Precipitação")
    
    print(f"\nModelo: EI30 → Sedimentação")
    print(f"  R²: {r2_ei30:.4f}")
    print(f"  RMSE: {rmse_ei30:.6f}")
    print(f"  Equação: Sedimentação = {reg_ei30.intercept_:.6f} + {reg_ei30.coef_[0]:.6f} × EI30")

# =============================================================================
# 5. VISUALIZAÇÕES
# =============================================================================
print("\n" + "=" * 80)
print("5. GERANDO VISUALIZAÇÕES")
print("=" * 80)

# Figura 1: Séries temporais integradas
fig, axes = plt.subplots(4, 1, figsize=(14, 12))

cores_area = {'SUP': 'blue', 'MED': 'green', 'INF': 'red'}

for area in areas:
    df_area = resultados[area]
    cor = cores_area.get(area, 'gray')
    
    # Precipitação
    axes[0].bar(df_area['DATA'], df_area['RAINFALL'], alpha=0.6, label=area, color=cor)
    
    # EI30
    axes[1].plot(df_area['DATA'], df_area['EI30'], marker='o', label=area, color=cor, linewidth=2)
    
    # Sedimentação acumulada
    axes[2].plot(df_area['DATA'], df_area['SEDIMENT'], marker='s', label=area, color=cor, linewidth=2)
    
    # Sedimentação incremental
    axes[3].bar(df_area['DATA'], df_area['FRACIONADO'], alpha=0.6, label=area, color=cor)

axes[0].set_title('Precipitação Mensal', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Precipitação (mm)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].set_title('Índice de Erosividade (EI30)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('EI30 (MJ·mm/ha·h)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].set_title('Sedimentação Acumulada', fontsize=12, fontweight='bold')
axes[2].set_ylabel('Sedimentação (cm)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

axes[3].set_title('Sedimentação Incremental Mensal', fontsize=12, fontweight='bold')
axes[3].set_ylabel('Sedimentação (cm)')
axes[3].set_xlabel('Data')
axes[3].legend()
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURAS_DIR / "9_series_temporais_integradas.png", dpi=300, bbox_inches='tight')
print("✓ Figura salva: 9_series_temporais_integradas.png")
plt.close()

# Figura 2: Correlações
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

for idx, area in enumerate(areas):
    df_area = resultados[area].dropna(subset=['RAINFALL', 'EI30', 'SEDIMENT', 'FRACIONADO'])
    
    if len(df_area) < 3:
        # Deixar subplot vazio com mensagem
        axes[0, idx].text(0.5, 0.5, f'{area}\nDados Insuficientes', 
                         ha='center', va='center', fontsize=14, fontweight='bold')
        axes[0, idx].set_xticks([])
        axes[0, idx].set_yticks([])
        axes[1, idx].text(0.5, 0.5, f'{area}\nDados Insuficientes', 
                         ha='center', va='center', fontsize=14, fontweight='bold')
        axes[1, idx].set_xticks([])
        axes[1, idx].set_yticks([])
        continue
    
    cor = cores_area.get(area, 'gray')
    
    # Precipitação vs Sedimentação Incremental
    axes[0, idx].scatter(df_area['RAINFALL'], df_area['FRACIONADO'], 
                        color=cor, alpha=0.6, s=100, edgecolors='black')
    
    # Regressão linear
    X = df_area['RAINFALL'].values.reshape(-1, 1)
    y = df_area['FRACIONADO'].values
    reg = LinearRegression().fit(X, y)
    y_pred = reg.predict(X)
    axes[0, idx].plot(df_area['RAINFALL'], y_pred, 'r--', linewidth=2)
    
    r2 = r2_score(y, y_pred)
    axes[0, idx].set_title(f'{area} - Precipitação vs Sedimentação\nR² = {r2:.4f}', 
                          fontweight='bold')
    axes[0, idx].set_xlabel('Precipitação (mm)')
    axes[0, idx].set_ylabel('Sedimentação Incremental (cm)')
    axes[0, idx].grid(True, alpha=0.3)
    
    # EI30 vs Sedimentação Incremental
    axes[1, idx].scatter(df_area['EI30'], df_area['FRACIONADO'], 
                        color=cor, alpha=0.6, s=100, edgecolors='black')
    
    X_ei30 = df_area['EI30'].values.reshape(-1, 1)
    reg_ei30 = LinearRegression().fit(X_ei30, y)
    y_pred_ei30 = reg_ei30.predict(X_ei30)
    axes[1, idx].plot(df_area['EI30'], y_pred_ei30, 'r--', linewidth=2)
    
    r2_ei30 = r2_score(y, y_pred_ei30)
    axes[1, idx].set_title(f'{area} - EI30 vs Sedimentação\nR² = {r2_ei30:.4f}', 
                          fontweight='bold')
    axes[1, idx].set_xlabel('EI30 (MJ·mm/ha·h)')
    axes[1, idx].set_ylabel('Sedimentação Incremental (cm)')
    axes[1, idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURAS_DIR / "10_correlacoes_precipitacao_ei30_sedimentacao.png", dpi=300, bbox_inches='tight')
print("✓ Figura salva: 10_correlacoes_precipitacao_ei30_sedimentacao.png")
plt.close()

# Figura 3: Matriz de correlação
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for idx, area in enumerate(areas):
    df_corr = resultados[area][['SEDIMENT', 'RAINFALL', 'EI30', 'FRACIONADO']].dropna()
    
    if len(df_corr) < 3:
        axes[idx].text(0.5, 0.5, f'{area}\nDados Insuficientes', 
                      ha='center', va='center', fontsize=14, fontweight='bold')
        axes[idx].set_xticks([])
        axes[idx].set_yticks([])
        continue
    
    corr_matrix = df_corr.corr()
    
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
               center=0, vmin=-1, vmax=1, ax=axes[idx],
               square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    axes[idx].set_title(f'Matriz de Correlação - {area}', fontweight='bold')

plt.tight_layout()
plt.savefig(FIGURAS_DIR / "11_matriz_correlacao_areas.png", dpi=300, bbox_inches='tight')
print("✓ Figura salva: 11_matriz_correlacao_areas.png")
plt.close()

# =============================================================================
# 6. SALVAR RESULTADOS
# =============================================================================
print("\n" + "=" * 80)
print("6. SALVANDO RESULTADOS")
print("=" * 80)

# Consolidar dados
df_consolidado = pd.concat([resultados[area].assign(AREA=area) for area in areas])
df_consolidado.to_csv(DADOS_DIR / "dados_integrados_sedimentacao.csv", index=False)
print("✓ Dados salvos: dados_integrados_sedimentacao.csv")

# Estatísticas por área
estatisticas = []
for area in areas:
    df_area = resultados[area].dropna(subset=['RAINFALL', 'EI30', 'FRACIONADO'])
    
    if len(df_area) < 3:
        continue
    
    estat = {
        'AREA': area,
        'N_Observacoes': len(df_area),
        'Sedimentacao_Total_cm': df_area['SEDIMENT'].max(),
        'Sedimentacao_Media_Mensal_cm': df_area['FRACIONADO'].mean(),
        'Precipitacao_Media_mm': df_area['RAINFALL'].mean(),
        'EI30_Medio': df_area['EI30'].mean(),
        'Corr_Precip_Sed': stats.pearsonr(df_area['RAINFALL'], df_area['FRACIONADO'])[0],
        'Corr_EI30_Sed': stats.pearsonr(df_area['EI30'], df_area['FRACIONADO'])[0],
        'R2_Precip': r2_score(df_area['FRACIONADO'], 
                             LinearRegression().fit(df_area[['RAINFALL']], 
                                                   df_area['FRACIONADO']).predict(df_area[['RAINFALL']])),
        'R2_EI30': r2_score(df_area['FRACIONADO'], 
                           LinearRegression().fit(df_area[['EI30']], 
                                                 df_area['FRACIONADO']).predict(df_area[['EI30']]))
    }
    estatisticas.append(estat)

df_estatisticas = pd.DataFrame(estatisticas)
df_estatisticas.to_csv(DADOS_DIR / "estatisticas_por_area.csv", index=False)
print("✓ Estatísticas salvas: estatisticas_por_area.csv")

print("\n" + "=" * 80)
print("RESUMO DAS ESTATÍSTICAS POR ÁREA")
print("=" * 80)
print(df_estatisticas.to_string(index=False))

print("\n" + "=" * 80)
print("ANÁLISE INTEGRADA CONCLUÍDA!")
print("=" * 80)
print(f"\nArquivos gerados:")
print(f"  Figuras: {FIGURAS_DIR}")
print(f"    - 9_series_temporais_integradas.png")
print(f"    - 10_correlacoes_precipitacao_ei30_sedimentacao.png")
print(f"    - 11_matriz_correlacao_areas.png")
print(f"\n  Dados: {DADOS_DIR}")
print(f"    - dados_integrados_sedimentacao.csv")
print(f"    - estatisticas_por_area.csv")
