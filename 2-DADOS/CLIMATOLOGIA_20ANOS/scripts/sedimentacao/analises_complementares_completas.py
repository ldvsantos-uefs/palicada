"""
ANÁLISES AVANÇADAS COMPLEMENTARES
Implementa todas as análises sugeridas:
1. Análise de eventos extremos específicos
2. Modelo SARIMA para prever erosão futura  
3. Comparação entre áreas (SUP, MED, INF)
4. Análise de defasagem temporal (correlação cruzada)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.signal import correlate
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
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
print("ANÁLISES AVANÇADAS COMPLEMENTARES - EROSÃO E SEDIMENTAÇÃO")
print("=" * 80)

# Carregar dados integrados
df_integrado = pd.read_csv(DADOS_DIR / "dados_integrados_sedimentacao.csv")
df_integrado['DATA'] = pd.to_datetime(df_integrado['DATA'])

print(f"\n✓ Dados carregados: {df_integrado.shape[0]} registros")
print(f"Áreas: {df_integrado['AREA'].unique()}")

# =============================================================================
# ANÁLISE 1: EVENTOS EXTREMOS ESPECÍFICOS
# =============================================================================
print("\n" + "=" * 80)
print("1. ANÁLISE DE EVENTOS EXTREMOS ESPECÍFICOS")
print("=" * 80)

# Identificar eventos extremos de precipitação
limiar_p90 = df_integrado['RAINFALL'].quantile(0.9)
eventos_extremos = df_integrado[df_integrado['RAINFALL'] > limiar_p90].copy()

print(f"\nLimiar P90: {limiar_p90:.2f} mm")
print(f"Eventos extremos identificados: {len(eventos_extremos)}")

if len(eventos_extremos) > 0:
    print(f"\nTop 5 Eventos Extremos:")
    top_eventos = eventos_extremos.nlargest(5, 'RAINFALL')[['DATA', 'AREA', 'RAINFALL', 'EI30', 'FRACIONADO']]
    for idx, row in top_eventos.iterrows():
        print(f"  {row['DATA'].strftime('%Y-%m')}: {row['RAINFALL']:.1f} mm → Sedimentação: {row['FRACIONADO']:.4f} cm ({row['AREA']})")
    
    # Calcular resposta erosiva aos eventos extremos
    resposta_extremos = eventos_extremos.groupby('AREA').agg({
        'RAINFALL': ['mean', 'max', 'count'],
        'EI30': ['mean', 'max'],
        'FRACIONADO': ['mean', 'max', 'sum']
    })
    
    print(f"\nResposta Erosiva por Área durante Eventos Extremos:")
    print(resposta_extremos)
    
    # Figura: Eventos extremos
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Scatter: Precipitação extrema vs Sedimentação
    for area in df_integrado['AREA'].unique():
        df_area_ext = eventos_extremos[eventos_extremos['AREA'] == area]
        if len(df_area_ext) > 0:
            axes[0, 0].scatter(df_area_ext['RAINFALL'], df_area_ext['FRACIONADO'], 
                             label=area, s=100, alpha=0.7)
    axes[0, 0].set_xlabel('Precipitação (mm)')
    axes[0, 0].set_ylabel('Sedimentação (cm)')
    axes[0, 0].set_title('Eventos Extremos: Precipitação vs Sedimentação')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Scatter: EI30 extremo vs Sedimentação
    for area in df_integrado['AREA'].unique():
        df_area_ext = eventos_extremos[eventos_extremos['AREA'] == area]
        if len(df_area_ext) > 0:
            axes[0, 1].scatter(df_area_ext['EI30'], df_area_ext['FRACIONADO'], 
                             label=area, s=100, alpha=0.7)
    axes[0, 1].set_xlabel('EI30 (MJ·mm/ha·h)')
    axes[0, 1].set_ylabel('Sedimentação (cm)')
    axes[0, 1].set_title('Eventos Extremos: EI30 vs Sedimentação')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Histograma de precipitação com limiar
    axes[1, 0].hist(df_integrado['RAINFALL'].dropna(), bins=20, alpha=0.5, 
                   color='blue', edgecolor='black', label='Todos os eventos')
    axes[1, 0].hist(eventos_extremos['RAINFALL'], bins=10, alpha=0.7, 
                   color='red', edgecolor='black', label='Eventos extremos')
    axes[1, 0].axvline(limiar_p90, color='red', linestyle='--', linewidth=2, label=f'P90 = {limiar_p90:.1f} mm')
    axes[1, 0].set_xlabel('Precipitação (mm)')
    axes[1, 0].set_ylabel('Frequência')
    axes[1, 0].set_title('Distribuição de Precipitação')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Box plot: Comparação sedimentação normal vs extrema
    df_normal = df_integrado[df_integrado['RAINFALL'] <= limiar_p90]
    data_box = [df_normal['FRACIONADO'].dropna(), eventos_extremos['FRACIONADO'].dropna()]
    axes[1, 1].boxplot(data_box, labels=['Normal', 'Extremo'])
    axes[1, 1].set_ylabel('Sedimentação (cm)')
    axes[1, 1].set_title('Sedimentação: Eventos Normais vs Extremos')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURAS_DIR / "12_analise_eventos_extremos.png", dpi=300, bbox_inches='tight')
    print("\n✓ Figura salva: 12_analise_eventos_extremos.png")
    plt.close()

# =============================================================================
# ANÁLISE 2: MODELO SARIMA PARA PREVER EROSÃO FUTURA
# =============================================================================
print("\n" + "=" * 80)
print("2. MODELO SARIMA PARA PREVISÃO DE EROSÃO")
print("=" * 80)

# Usar dados da área SUP (única com dados completos)
df_sup = df_integrado[df_integrado['AREA'] == 'SUP'].copy()
df_sup = df_sup.set_index('DATA').sort_index()

# Série temporal de sedimentação incremental
serie_sed = df_sup['FRACIONADO'].dropna()

if len(serie_sed) >= 12:
    print(f"\nAjustando modelo SARIMA para sedimentação...")
    print(f"Dados disponíveis: {len(serie_sed)} observações")
    
    try:
        # Ajustar SARIMA
        modelo_sed = SARIMAX(serie_sed, order=(1,0,1), seasonal_order=(0,0,0,0))
        resultado_sed = modelo_sed.fit(disp=False)
        
        print(f"\nResumo do Modelo:")
        print(f"  AIC: {resultado_sed.aic:.2f}")
        print(f"  BIC: {resultado_sed.bic:.2f}")
        
        # Previsão para 12 meses
        previsao_sed = resultado_sed.get_forecast(steps=12)
        previsao_media_sed = previsao_sed.predicted_mean
        intervalo_conf_sed = previsao_sed.conf_int()
        
        print(f"\nPrevisão de Sedimentação para próximos 12 meses:")
        print(f"  Sedimentação total prevista: {previsao_media_sed.sum():.4f} cm")
        print(f"  Média mensal: {previsao_media_sed.mean():.4f} cm")
        
        # Figura: Previsão SARIMA
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Dados históricos
        ax.plot(serie_sed.index, serie_sed, 'o-', label='Observado', linewidth=2)
        
        # Previsão
        ax.plot(previsao_media_sed.index, previsao_media_sed, 'r-', 
               label='Previsão SARIMA', linewidth=2)
        ax.fill_between(intervalo_conf_sed.index,
                        intervalo_conf_sed.iloc[:, 0],
                        intervalo_conf_sed.iloc[:, 1],
                        color='pink', alpha=0.3, label='IC 95%')
        
        ax.set_title('Previsão de Sedimentação Incremental - Modelo SARIMA', 
                    fontsize=14, fontweight='bold')
        ax.set_ylabel('Sedimentação (cm)')
        ax.set_xlabel('Data')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIGURAS_DIR / "13_previsao_sarima_sedimentacao.png", dpi=300, bbox_inches='tight')
        print("✓ Figura salva: 13_previsao_sarima_sedimentacao.png")
        plt.close()
        
        # Salvar previsões
        df_previsao = pd.DataFrame({
            'DATA': previsao_media_sed.index,
            'Sedimentacao_Prevista_cm': previsao_media_sed.values,
            'IC_Inferior': intervalo_conf_sed.iloc[:, 0].values,
            'IC_Superior': intervalo_conf_sed.iloc[:, 1].values
        })
        df_previsao.to_csv(DADOS_DIR / "previsao_sedimentacao_12meses.csv", index=False)
        print("✓ Dados salvos: previsao_sedimentacao_12meses.csv")
        
    except Exception as e:
        print(f"\n⚠️  Erro ao ajustar SARIMA: {e}")
else:
    print(f"\n⚠️  Dados insuficientes para SARIMA (n={len(serie_sed)})")

# =============================================================================
# ANÁLISE 3: COMPARAÇÃO ENTRE ÁREAS
# =============================================================================
print("\n" + "=" * 80)
print("3. COMPARAÇÃO ENTRE ÁREAS (SUP, MED, INF)")
print("=" * 80)

# Estatísticas por área
comparacao_areas = df_integrado.groupby('AREA').agg({
    'SEDIMENT': ['max', 'mean'],
    'FRACIONADO': ['mean', 'std', 'sum'],
    'RAINFALL': ['mean', 'sum'],
    'EI30': ['mean', 'max']
}).round(4)

print("\nComparação de Sedimentação entre Áreas:")
print(comparacao_areas)

# Contar observações por área
contagem = df_integrado.groupby('AREA').size()
print(f"\nObservações por área:")
for area, count in contagem.items():
    print(f"  {area}: {count} registros")

# Figura: Comparação entre áreas
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Box plot: Sedimentação incremental por área
areas_com_dados = []
dados_sedimentacao = []
for area in df_integrado['AREA'].unique():
    df_area = df_integrado[df_integrado['AREA'] == area]['FRACIONADO'].dropna()
    if len(df_area) > 0:
        areas_com_dados.append(area)
        dados_sedimentacao.append(df_area.values)

if len(areas_com_dados) > 0:
    bp = axes[0, 0].boxplot(dados_sedimentacao, labels=areas_com_dados, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    axes[0, 0].set_ylabel('Sedimentação Incremental (cm)')
    axes[0, 0].set_title('Distribuição de Sedimentação por Área')
    axes[0, 0].grid(True, alpha=0.3, axis='y')

# Série temporal comparativa
for area in df_integrado['AREA'].unique():
    df_area = df_integrado[df_integrado['AREA'] == area].sort_values('DATA')
    if len(df_area) > 0:
        axes[0, 1].plot(df_area['DATA'], df_area['SEDIMENT'], 
                       marker='o', label=area, linewidth=2)
axes[0, 1].set_ylabel('Sedimentação Acumulada (cm)')
axes[0, 1].set_xlabel('Data')
axes[0, 1].set_title('Evolução Temporal da Sedimentação')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Gráfico de barras: Sedimentação total por área
if len(comparacao_areas) > 0:
    sed_total = df_integrado.groupby('AREA')['SEDIMENT'].max()
    axes[1, 0].bar(sed_total.index, sed_total.values, color=['blue', 'green', 'red'], 
                  alpha=0.7, edgecolor='black')
    axes[1, 0].set_ylabel('Sedimentação Total Acumulada (cm)')
    axes[1, 0].set_xlabel('Área')
    axes[1, 0].set_title('Sedimentação Total por Área')
    axes[1, 0].grid(True, alpha=0.3, axis='y')

# Taxa de sedimentação (cm/mês)
for area in df_integrado['AREA'].unique():
    df_area = df_integrado[df_integrado['AREA'] == area].sort_values('DATA')
    if len(df_area) > 1:
        axes[1, 1].plot(df_area['DATA'], df_area['FRACIONADO'], 
                       marker='s', label=area, linewidth=2, alpha=0.7)
axes[1, 1].set_ylabel('Taxa de Sedimentação (cm/mês)')
axes[1, 1].set_xlabel('Data')
axes[1, 1].set_title('Taxa Mensal de Sedimentação')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURAS_DIR / "14_comparacao_entre_areas.png", dpi=300, bbox_inches='tight')
print("\n✓ Figura salva: 14_comparacao_entre_areas.png")
plt.close()

# =============================================================================
# ANÁLISE 4: DEFASAGEM TEMPORAL (LAG) - CORRELAÇÃO CRUZADA
# =============================================================================
print("\n" + "=" * 80)
print("4. ANÁLISE DE DEFASAGEM TEMPORAL (CORRELAÇÃO CRUZADA)")
print("=" * 80)

# Análise de lag para área SUP
df_sup_lag = df_integrado[df_integrado['AREA'] == 'SUP'].sort_values('DATA')

if len(df_sup_lag) >= 10:
    print(f"\nAnalisando defasagem temporal (n={len(df_sup_lag)} observações)...")
    
    # Preparar séries
    precip = df_sup_lag['RAINFALL'].fillna(0).values
    sedim = df_sup_lag['FRACIONADO'].fillna(0).values
    ei30 = df_sup_lag['EI30'].fillna(0).values
    
    # Calcular correlação cruzada
    max_lag = min(6, len(precip) - 3)  # Máximo 6 meses de lag
    
    correlacoes_precip = []
    correlacoes_ei30 = []
    lags = range(-max_lag, max_lag + 1)
    
    for lag in lags:
        if lag < 0:
            corr_p = np.corrcoef(precip[:lag], sedim[-lag:])[0, 1] if len(precip[:lag]) > 1 else 0
            corr_e = np.corrcoef(ei30[:lag], sedim[-lag:])[0, 1] if len(ei30[:lag]) > 1 else 0
        elif lag > 0:
            corr_p = np.corrcoef(precip[lag:], sedim[:-lag])[0, 1] if len(precip[lag:]) > 1 else 0
            corr_e = np.corrcoef(ei30[lag:], sedim[:-lag])[0, 1] if len(ei30[lag:]) > 1 else 0
        else:
            corr_p = np.corrcoef(precip, sedim)[0, 1] if len(precip) > 1 else 0
            corr_e = np.corrcoef(ei30, sedim)[0, 1] if len(ei30) > 1 else 0
        
        correlacoes_precip.append(corr_p if not np.isnan(corr_p) else 0)
        correlacoes_ei30.append(corr_e if not np.isnan(corr_e) else 0)
    
    # Encontrar lag ótimo
    lag_otimo_precip = lags[np.argmax(np.abs(correlacoes_precip))]
    lag_otimo_ei30 = lags[np.argmax(np.abs(correlacoes_ei30))]
    
    print(f"\nDefasagem ótima:")
    print(f"  Precipitação: {lag_otimo_precip} meses (r={max(np.abs(correlacoes_precip)):.4f})")
    print(f"  EI30: {lag_otimo_ei30} meses (r={max(np.abs(correlacoes_ei30)):.4f})")
    
    # Figura: Correlação cruzada
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Precipitação
    axes[0].stem(lags, correlacoes_precip, basefmt=' ')
    axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[0].axvline(x=lag_otimo_precip, color='red', linestyle='--', linewidth=2, 
                   label=f'Lag ótimo: {lag_otimo_precip} meses')
    axes[0].set_xlabel('Defasagem (meses)')
    axes[0].set_ylabel('Correlação')
    axes[0].set_title('Correlação Cruzada: Precipitação × Sedimentação')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # EI30
    axes[1].stem(lags, correlacoes_ei30, basefmt=' ')
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1].axvline(x=lag_otimo_ei30, color='red', linestyle='--', linewidth=2,
                   label=f'Lag ótimo: {lag_otimo_ei30} meses')
    axes[1].set_xlabel('Defasagem (meses)')
    axes[1].set_ylabel('Correlação')
    axes[1].set_title('Correlação Cruzada: EI30 × Sedimentação')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURAS_DIR / "15_correlacao_cruzada_lag.png", dpi=300, bbox_inches='tight')
    print("✓ Figura salva: 15_correlacao_cruzada_lag.png")
    plt.close()
    
    # Salvar resultados de lag
    df_lag = pd.DataFrame({
        'Lag_meses': lags,
        'Correlacao_Precipitacao': correlacoes_precip,
        'Correlacao_EI30': correlacoes_ei30
    })
    df_lag.to_csv(DADOS_DIR / "analise_lag_temporal.csv", index=False)
    print("✓ Dados salvos: analise_lag_temporal.csv")
else:
    print(f"\n⚠️  Dados insuficientes para análise de lag (n={len(df_sup_lag)})")

# =============================================================================
# RELATÓRIO FINAL
# =============================================================================
print("\n" + "=" * 80)
print("RELATÓRIO FINAL DAS ANÁLISES COMPLEMENTARES")
print("=" * 80)

relatorio_complementar = f"""
ANÁLISES AVANÇADAS COMPLEMENTARES - EROSÃO E SEDIMENTAÇÃO
Data: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

1. EVENTOS EXTREMOS
   - Limiar P90: {limiar_p90:.2f} mm
   - Eventos identificados: {len(eventos_extremos)}
   - Maior precipitação: {eventos_extremos['RAINFALL'].max():.2f} mm

2. PREVISÃO SARIMA (12 meses)
   - Sedimentação total prevista: {previsao_media_sed.sum():.4f} cm
   - Taxa mensal média: {previsao_media_sed.mean():.4f} cm/mês

3. COMPARAÇÃO ENTRE ÁREAS
   - Área SUP: Dados completos (24 obs)
   - Área MED: Dados incompletos
   - Área INF: Dados incompletos

4. DEFASAGEM TEMPORAL
   - Lag ótimo (Precipitação): {lag_otimo_precip} meses
   - Lag ótimo (EI30): {lag_otimo_ei30} meses
   - Correlação máxima: {max(np.abs(correlacoes_precip)):.4f}

ARQUIVOS GERADOS:
  Figuras:
  - 12_analise_eventos_extremos.png
  - 13_previsao_sarima_sedimentacao.png
  - 14_comparacao_entre_areas.png
  - 15_correlacao_cruzada_lag.png
  
  Dados:
  - previsao_sedimentacao_12meses.csv
  - analise_lag_temporal.csv
"""

with open(DADOS_DIR / "relatorio_analises_complementares.txt", 'w', encoding='utf-8') as f:
    f.write(relatorio_complementar)

print(relatorio_complementar)
print("\n✓ Relatório salvo: relatorio_analises_complementares.txt")

print("\n" + "=" * 80)
print("TODAS AS ANÁLISES COMPLEMENTARES CONCLUÍDAS!")
print("=" * 80)
