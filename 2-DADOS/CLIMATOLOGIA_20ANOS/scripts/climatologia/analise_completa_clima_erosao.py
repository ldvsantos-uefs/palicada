"""
Análise Avançada Completa de Séries Temporais Climáticas com Foco em Erosão

Implementa todas as análises estatísticas avançadas:
1. Decomposição da série temporal (tendência, sazonalidade, resíduo)
2. Testes de estacionariedade (Dickey-Fuller aumentado)
3. Diferenciação sazonal para remover padrões cíclicos
4. Modelos SARIMA para capturar padrões sazonais
5. Análise de extremos usando distribuição GEV
6. Identificação de eventos extremos de precipitação
7. Cálculo de índices de erosividade (EI30)
8. Modelos IDF (Intensidade-Duração-Frequência)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from scipy.optimize import minimize
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, acf, pacf
from pymannkendall import original_test
import warnings

# Configurações
warnings.filterwarnings('ignore')
plt.style.use('ggplot')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 300

# Diretórios
BASE_DIR = Path(__file__).parent.parent
FIGURAS_DIR = BASE_DIR / "figuras"
DADOS_DIR = BASE_DIR / "dados"
FIGURAS_DIR.mkdir(exist_ok=True)
DADOS_DIR.mkdir(exist_ok=True)

# Carregar dados
print("=" * 80)
print("ANÁLISE AVANÇADA DE SÉRIES TEMPORAIS CLIMÁTICAS COM FOCO EM EROSÃO")
print("=" * 80)
print("\nCarregando dados...")
df = pd.read_csv(DADOS_DIR / "serie_precipitacao_20anos.csv", parse_dates=['Data'], index_col='Data')
df = df.rename(columns={'Precipitacao_mm': 'precipitacao'})
print(f"✓ Dados carregados: {df.shape[0]} registros de {df.index.min().date()} a {df.index.max().date()}")

# Preparar dados mensais
mensal = df['precipitacao'].resample('M').sum()

# =============================================================================
# ANÁLISE 1: DECOMPOSIÇÃO DA SÉRIE TEMPORAL
# =============================================================================
print("\n" + "=" * 80)
print("1. DECOMPOSIÇÃO DA SÉRIE TEMPORAL")
print("=" * 80)

decomposicao = sm.tsa.seasonal_decompose(mensal, model='additive', period=12)

# Plotar decomposição
fig, axes = plt.subplots(4, 1, figsize=(14, 10))
decomposicao.observed.plot(ax=axes[0], title='Série Original')
axes[0].set_ylabel('Precipitação (mm)')
decomposicao.trend.plot(ax=axes[1], title='Tendência')
axes[1].set_ylabel('Tendência (mm)')
decomposicao.seasonal.plot(ax=axes[2], title='Sazonalidade')
axes[2].set_ylabel('Sazonalidade (mm)')
decomposicao.resid.plot(ax=axes[3], title='Resíduo')
axes[3].set_ylabel('Resíduo (mm)')
plt.tight_layout()
plt.savefig(FIGURAS_DIR / "1_decomposicao_serie_temporal.png", dpi=300, bbox_inches='tight')
print("✓ Figura salva: 1_decomposicao_serie_temporal.png")
plt.close()

# Estatísticas da decomposição
print(f"\nEstatísticas da Decomposição:")
print(f"  Tendência média: {decomposicao.trend.mean():.2f} mm")
print(f"  Amplitude sazonal: {decomposicao.seasonal.max() - decomposicao.seasonal.min():.2f} mm")
print(f"  Desvio padrão dos resíduos: {decomposicao.resid.std():.2f} mm")

# =============================================================================
# ANÁLISE 2: TESTES DE ESTACIONARIEDADE (DICKEY-FULLER AUMENTADO)
# =============================================================================
print("\n" + "=" * 80)
print("2. TESTE DE ESTACIONARIEDADE (DICKEY-FULLER AUMENTADO)")
print("=" * 80)

# Teste ADF na série original
adf_original = adfuller(mensal.dropna(), autolag='AIC')
print("\nSérie Original:")
print(f"  Estatística ADF: {adf_original[0]:.4f}")
print(f"  p-valor: {adf_original[1]:.4f}")
print(f"  Valores críticos:")
for key, value in adf_original[4].items():
    print(f"    {key}: {value:.4f}")
print(f"  Conclusão: {'Série ESTACIONÁRIA' if adf_original[1] < 0.05 else 'Série NÃO estacionária'}")

# Teste ADF na série diferenciada
mensal_diff = mensal.diff().dropna()
adf_diff = adfuller(mensal_diff, autolag='AIC')
print("\nSérie Diferenciada (1ª diferença):")
print(f"  Estatística ADF: {adf_diff[0]:.4f}")
print(f"  p-valor: {adf_diff[1]:.4f}")
print(f"  Conclusão: {'Série ESTACIONÁRIA' if adf_diff[1] < 0.05 else 'Série NÃO estacionária'}")

# Plotar série original vs diferenciada
fig, axes = plt.subplots(2, 1, figsize=(14, 8))
mensal.plot(ax=axes[0], title='Série Original')
axes[0].set_ylabel('Precipitação (mm)')
axes[0].grid(True)
mensal_diff.plot(ax=axes[1], title='Série Diferenciada (1ª diferença)', color='orange')
axes[1].set_ylabel('Diferença de Precipitação (mm)')
axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
axes[1].grid(True)
plt.tight_layout()
plt.savefig(FIGURAS_DIR / "2_teste_estacionariedade.png", dpi=300, bbox_inches='tight')
print("✓ Figura salva: 2_teste_estacionariedade.png")
plt.close()

# =============================================================================
# ANÁLISE 3: DIFERENCIAÇÃO SAZONAL
# =============================================================================
print("\n" + "=" * 80)
print("3. DIFERENCIAÇÃO SAZONAL")
print("=" * 80)

# Diferenciação sazonal (lag=12 meses)
mensal_sazonal_diff = mensal.diff(12).dropna()

# Teste ADF na série com diferenciação sazonal
adf_sazonal = adfuller(mensal_sazonal_diff, autolag='AIC')
print(f"\nSérie com Diferenciação Sazonal (lag=12):")
print(f"  Estatística ADF: {adf_sazonal[0]:.4f}")
print(f"  p-valor: {adf_sazonal[1]:.4f}")
print(f"  Conclusão: {'Série ESTACIONÁRIA' if adf_sazonal[1] < 0.05 else 'Série NÃO estacionária'}")

# Plotar ACF e PACF
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# ACF e PACF da série original
acf_vals = acf(mensal.dropna(), nlags=40)
pacf_vals = pacf(mensal.dropna(), nlags=40)
axes[0, 0].stem(range(len(acf_vals)), acf_vals, basefmt=' ')
axes[0, 0].set_title('ACF - Série Original')
axes[0, 0].set_xlabel('Lag')
axes[0, 0].set_ylabel('Autocorrelação')
axes[0, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axes[0, 0].axhline(y=1.96/np.sqrt(len(mensal)), color='r', linestyle='--')
axes[0, 0].axhline(y=-1.96/np.sqrt(len(mensal)), color='r', linestyle='--')
axes[0, 0].grid(True)

axes[0, 1].stem(range(len(pacf_vals)), pacf_vals, basefmt=' ')
axes[0, 1].set_title('PACF - Série Original')
axes[0, 1].set_xlabel('Lag')
axes[0, 1].set_ylabel('Autocorrelação Parcial')
axes[0, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axes[0, 1].axhline(y=1.96/np.sqrt(len(mensal)), color='r', linestyle='--')
axes[0, 1].axhline(y=-1.96/np.sqrt(len(mensal)), color='r', linestyle='--')
axes[0, 1].grid(True)

# ACF e PACF da série com diferenciação sazonal
acf_vals_saz = acf(mensal_sazonal_diff.dropna(), nlags=40)
pacf_vals_saz = pacf(mensal_sazonal_diff.dropna(), nlags=40)
axes[1, 0].stem(range(len(acf_vals_saz)), acf_vals_saz, basefmt=' ')
axes[1, 0].set_title('ACF - Série com Diferenciação Sazonal')
axes[1, 0].set_xlabel('Lag')
axes[1, 0].set_ylabel('Autocorrelação')
axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axes[1, 0].axhline(y=1.96/np.sqrt(len(mensal_sazonal_diff)), color='r', linestyle='--')
axes[1, 0].axhline(y=-1.96/np.sqrt(len(mensal_sazonal_diff)), color='r', linestyle='--')
axes[1, 0].grid(True)

axes[1, 1].stem(range(len(pacf_vals_saz)), pacf_vals_saz, basefmt=' ')
axes[1, 1].set_title('PACF - Série com Diferenciação Sazonal')
axes[1, 1].set_xlabel('Lag')
axes[1, 1].set_ylabel('Autocorrelação Parcial')
axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axes[1, 1].axhline(y=1.96/np.sqrt(len(mensal_sazonal_diff)), color='r', linestyle='--')
axes[1, 1].axhline(y=-1.96/np.sqrt(len(mensal_sazonal_diff)), color='r', linestyle='--')
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig(FIGURAS_DIR / "3_diferenciacao_sazonal_acf_pacf.png", dpi=300, bbox_inches='tight')
print("✓ Figura salva: 3_diferenciacao_sazonal_acf_pacf.png")
plt.close()

# =============================================================================
# ANÁLISE 4: MODELAGEM SARIMA
# =============================================================================
print("\n" + "=" * 80)
print("4. MODELAGEM SARIMA PARA PREVISÃO SAZONAL")
print("=" * 80)

# Ajustar modelo SARIMA
print("\nAjustando modelo SARIMA(1,1,1)(1,1,1,12)...")
modelo_sarima = SARIMAX(mensal, order=(1,1,1), seasonal_order=(1,1,1,12))
resultado_sarima = modelo_sarima.fit(disp=False)

print("\nResumo do Modelo:")
print(f"  AIC: {resultado_sarima.aic:.2f}")
print(f"  BIC: {resultado_sarima.bic:.2f}")
print(f"  Log-Likelihood: {resultado_sarima.llf:.2f}")

# Previsão para 24 meses
previsao = resultado_sarima.get_forecast(steps=24)
previsao_media = previsao.predicted_mean
intervalo_conf = previsao.conf_int()

# Plotar previsão
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(mensal.index, mensal, label='Observado', linewidth=1.5)
ax.plot(previsao_media.index, previsao_media, color='red', label='Previsão SARIMA', linewidth=2)
ax.fill_between(intervalo_conf.index, 
                intervalo_conf.iloc[:,0], 
                intervalo_conf.iloc[:,1], 
                color='pink', alpha=0.3, label='Intervalo de Confiança 95%')
ax.set_title('Previsão de Precipitação Mensal - Modelo SARIMA', fontsize=14, fontweight='bold')
ax.set_ylabel('Precipitação (mm)')
ax.set_xlabel('Data')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURAS_DIR / "4_previsao_sarima.png", dpi=300, bbox_inches='tight')
print("✓ Figura salva: 4_previsao_sarima.png")
plt.close()

# Diagnóstico do modelo
fig = resultado_sarima.plot_diagnostics(figsize=(14, 10))
plt.tight_layout()
plt.savefig(FIGURAS_DIR / "4_diagnostico_sarima.png", dpi=300, bbox_inches='tight')
print("✓ Figura salva: 4_diagnostico_sarima.png")
plt.close()

# =============================================================================
# ANÁLISE 5: DISTRIBUIÇÃO GEV PARA EXTREMOS
# =============================================================================
print("\n" + "=" * 80)
print("5. ANÁLISE DE EXTREMOS - DISTRIBUIÇÃO GEV")
print("=" * 80)

# Extrair máximos anuais
maximos_anuais = df['precipitacao'].resample('Y').max()
print(f"\nMáximos anuais extraídos: {len(maximos_anuais)} anos")

# Ajustar distribuição GEV
gev_params = stats.genextreme.fit(maximos_anuais)
print(f"\nParâmetros da Distribuição GEV:")
print(f"  c (shape): {gev_params[0]:.4f}")
print(f"  loc (location): {gev_params[1]:.4f}")
print(f"  scale: {gev_params[2]:.4f}")

# Calcular períodos de retorno
periodos_retorno = np.array([2, 5, 10, 20, 50, 100])
valores_retorno = stats.genextreme.ppf(1 - 1/periodos_retorno, *gev_params)

print(f"\nValores de Retorno (mm/dia):")
for periodo, valor in zip(periodos_retorno, valores_retorno):
    print(f"  TR={periodo} anos: {valor:.2f} mm")

# Plotar ajuste GEV
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Histograma com PDF ajustado
axes[0].hist(maximos_anuais, bins=15, density=True, alpha=0.7, color='skyblue', edgecolor='black')
x_gev = np.linspace(maximos_anuais.min(), maximos_anuais.max(), 200)
axes[0].plot(x_gev, stats.genextreme.pdf(x_gev, *gev_params), 'r-', linewidth=2, label='GEV Ajustada')
axes[0].set_xlabel('Precipitação Máxima Anual (mm)')
axes[0].set_ylabel('Densidade de Probabilidade')
axes[0].set_title('Distribuição GEV - Máximos Anuais')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Curva de período de retorno
axes[1].scatter(periodos_retorno, valores_retorno, s=100, color='red', zorder=5, label='Valores Calculados')
axes[1].plot(periodos_retorno, valores_retorno, 'r--', linewidth=1.5)
axes[1].set_xlabel('Período de Retorno (anos)')
axes[1].set_ylabel('Precipitação Máxima (mm/dia)')
axes[1].set_title('Curva de Período de Retorno')
axes[1].set_xscale('log')
axes[1].grid(True, alpha=0.3, which='both')
axes[1].legend()

plt.tight_layout()
plt.savefig(FIGURAS_DIR / "5_analise_extremos_gev.png", dpi=300, bbox_inches='tight')
print("✓ Figura salva: 5_analise_extremos_gev.png")
plt.close()

# Salvar valores de retorno
df_retorno = pd.DataFrame({
    'Periodo_Retorno_anos': periodos_retorno,
    'Precipitacao_mm': valores_retorno
})
df_retorno.to_csv(DADOS_DIR / "valores_retorno_gev.csv", index=False)
print("✓ Dados salvos: valores_retorno_gev.csv")

# =============================================================================
# ANÁLISE 6: EVENTOS EXTREMOS DE PRECIPITAÇÃO
# =============================================================================
print("\n" + "=" * 80)
print("6. IDENTIFICAÇÃO DE EVENTOS EXTREMOS")
print("=" * 80)

# Identificar eventos extremos (percentil 95)
limiar_p95 = np.percentile(df['precipitacao'], 95)
print(f"\nLimiar de eventos extremos (P95): {limiar_p95:.2f} mm")

eventos_extremos = df[df['precipitacao'] > limiar_p95].reset_index()
if not eventos_extremos.empty:
    eventos_extremos['diff'] = eventos_extremos['Data'].diff().dt.days.fillna(0)
    eventos_extremos['grupo'] = (eventos_extremos['diff'] > 1).cumsum()
    
    resumo_eventos = eventos_extremos.groupby('grupo').agg(
        inicio=('Data', 'min'),
        fim=('Data', 'max'),
        duracao=('Data', lambda x: (x.max() - x.min()).days + 1),
        precipitacao_total=('precipitacao', 'sum'),
        intensidade_max=('precipitacao', 'max')
    ).reset_index(drop=True)
    
    print(f"Total de eventos extremos identificados: {len(resumo_eventos)}")
    print(f"Duração média dos eventos: {resumo_eventos['duracao'].mean():.1f} dias")
    print(f"Precipitação total média por evento: {resumo_eventos['precipitacao_total'].mean():.2f} mm")
    print(f"Intensidade máxima média: {resumo_eventos['intensidade_max'].mean():.2f} mm/dia")
    
    # Plotar eventos extremos
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Série temporal com eventos destacados
    axes[0, 0].plot(df.index, df['precipitacao'], color='gray', alpha=0.5, linewidth=0.5)
    axes[0, 0].scatter(eventos_extremos['Data'], eventos_extremos['precipitacao'], 
                      color='red', s=20, alpha=0.6, label='Eventos Extremos')
    axes[0, 0].axhline(y=limiar_p95, color='red', linestyle='--', label=f'Limiar P95 ({limiar_p95:.1f} mm)')
    axes[0, 0].set_title('Série Temporal com Eventos Extremos')
    axes[0, 0].set_ylabel('Precipitação (mm)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Distribuição de durações
    axes[0, 1].hist(resumo_eventos['duracao'], bins=20, color='steelblue', edgecolor='black')
    axes[0, 1].set_title('Distribuição de Duração dos Eventos')
    axes[0, 1].set_xlabel('Duração (dias)')
    axes[0, 1].set_ylabel('Frequência')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Distribuição de precipitação total
    axes[1, 0].hist(resumo_eventos['precipitacao_total'], bins=20, color='green', edgecolor='black')
    axes[1, 0].set_title('Distribuição de Precipitação Total por Evento')
    axes[1, 0].set_xlabel('Precipitação Total (mm)')
    axes[1, 0].set_ylabel('Frequência')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Distribuição de intensidade máxima
    axes[1, 1].hist(resumo_eventos['intensidade_max'], bins=20, color='orange', edgecolor='black')
    axes[1, 1].set_title('Distribuição de Intensidade Máxima')
    axes[1, 1].set_xlabel('Intensidade Máxima (mm/dia)')
    axes[1, 1].set_ylabel('Frequência')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURAS_DIR / "6_eventos_extremos.png", dpi=300, bbox_inches='tight')
    print("✓ Figura salva: 6_eventos_extremos.png")
    plt.close()
    
    # Salvar resumo de eventos
    resumo_eventos.to_csv(DADOS_DIR / "eventos_extremos_detalhados.csv", index=False)
    print("✓ Dados salvos: eventos_extremos_detalhados.csv")

# =============================================================================
# ANÁLISE 7: ÍNDICES DE EROSIVIDADE (EI30)
# =============================================================================
print("\n" + "=" * 80)
print("7. ÍNDICES DE EROSIVIDADE (EI30)")
print("=" * 80)

# Calcular EI30 mensal
ec = 0.29 * (1 - 0.72 * np.exp(-0.05 * df['precipitacao']))
i30 = df['precipitacao'].resample('M').max()
ei30_mensal = (ec * df['precipitacao']).resample('M').sum() * i30

print(f"\nEstatísticas do EI30 Mensal:")
print(f"  Média: {ei30_mensal.mean():.2f} MJ·mm/ha·h")
print(f"  Desvio padrão: {ei30_mensal.std():.2f} MJ·mm/ha·h")
print(f"  Máximo: {ei30_mensal.max():.2f} MJ·mm/ha·h")
print(f"  Mínimo: {ei30_mensal.min():.2f} MJ·mm/ha·h")

# EI30 anual
ei30_anual = ei30_mensal.resample('Y').sum()
print(f"\nEI30 Anual Médio: {ei30_anual.mean():.2f} MJ·mm/ha·h")

# Plotar EI30
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Série temporal mensal
axes[0].plot(ei30_mensal.index, ei30_mensal, linewidth=1.5, color='darkred')
axes[0].set_title('Índice de Erosividade (EI30) Mensal', fontsize=14, fontweight='bold')
axes[0].set_ylabel('EI30 (MJ·mm/ha·h)')
axes[0].grid(True, alpha=0.3)

# Série temporal anual
axes[1].bar(ei30_anual.index.year, ei30_anual, color='darkgreen', edgecolor='black', alpha=0.7)
axes[1].axhline(y=ei30_anual.mean(), color='red', linestyle='--', linewidth=2, label=f'Média: {ei30_anual.mean():.0f}')
axes[1].set_title('Índice de Erosividade (EI30) Anual', fontsize=14, fontweight='bold')
axes[1].set_ylabel('EI30 (MJ·mm/ha·h)')
axes[1].set_xlabel('Ano')
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(FIGURAS_DIR / "7_indices_erosividade_ei30.png", dpi=300, bbox_inches='tight')
print("✓ Figura salva: 7_indices_erosividade_ei30.png")
plt.close()

# Sazonalidade do EI30
ei30_boxplot = ei30_mensal.groupby(ei30_mensal.index.month)

fig, ax = plt.subplots(figsize=(12, 6))
ei30_mensal_mes = [ei30_mensal[ei30_mensal.index.month == i].values for i in range(1, 13)]
bp = ax.boxplot(ei30_mensal_mes, labels=['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 
                                          'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'],
                patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
    patch.set_alpha(0.7)
ax.set_title('Sazonalidade do EI30', fontsize=14, fontweight='bold')
ax.set_ylabel('EI30 (MJ·mm/ha·h)')
ax.set_xlabel('Mês')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(FIGURAS_DIR / "7_ei30_sazonalidade.png", dpi=300, bbox_inches='tight')
print("✓ Figura salva: 7_ei30_sazonalidade.png")
plt.close()

# Salvar dados
ei30_mensal.to_csv(DADOS_DIR / "ei30_mensal.csv")
ei30_anual.to_csv(DADOS_DIR / "ei30_anual.csv")
print("✓ Dados salvos: ei30_mensal.csv, ei30_anual.csv")

# =============================================================================
# ANÁLISE 8: CURVAS IDF (INTENSIDADE-DURAÇÃO-FREQUÊNCIA)
# =============================================================================
print("\n" + "=" * 80)
print("8. CURVAS IDF (INTENSIDADE-DURAÇÃO-FREQUÊNCIA)")
print("=" * 80)

# Calcular intensidades para diferentes durações
duracoes = np.array([1, 2, 3, 5, 7, 10, 15, 20, 30])  # dias
periodos_retorno_idf = np.array([2, 5, 10, 20, 50, 100])  # anos

# Matriz para armazenar intensidades
intensidades_idf = np.zeros((len(duracoes), len(periodos_retorno_idf)))

print("\nCalculando intensidades para diferentes durações...")
for i, duracao in enumerate(duracoes):
    # Extrair máximas precipitações acumuladas para cada duração
    precip_acumulada = df['precipitacao'].rolling(window=duracao).sum()
    maximos_duracao = precip_acumulada.resample('Y').max().dropna()
    
    # Ajustar GEV para esta duração
    if len(maximos_duracao) > 3:
        try:
            gev_params_dur = stats.genextreme.fit(maximos_duracao)
            valores_retorno_dur = stats.genextreme.ppf(1 - 1/periodos_retorno_idf, *gev_params_dur)
            # Converter para intensidade (mm/hora)
            intensidades_idf[i, :] = valores_retorno_dur / (duracao * 24)
        except:
            intensidades_idf[i, :] = np.nan

# Plotar curvas IDF
fig, ax = plt.subplots(figsize=(12, 8))
cores = plt.cm.viridis(np.linspace(0, 1, len(periodos_retorno_idf)))

for j, (tr, cor) in enumerate(zip(periodos_retorno_idf, cores)):
    mask = ~np.isnan(intensidades_idf[:, j])
    if mask.any():
        ax.plot(duracoes[mask], intensidades_idf[mask, j], 'o-', 
               label=f'TR = {tr} anos', linewidth=2, markersize=8, color=cor)

ax.set_xlabel('Duração (dias)', fontsize=12)
ax.set_ylabel('Intensidade (mm/h)', fontsize=12)
ax.set_title('Curvas IDF - Intensidade-Duração-Frequência', fontsize=14, fontweight='bold')
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(True, alpha=0.3, which='both')
ax.legend(loc='best', fontsize=10)
plt.tight_layout()
plt.savefig(FIGURAS_DIR / "8_curvas_idf.png", dpi=300, bbox_inches='tight')
print("✓ Figura salva: 8_curvas_idf.png")
plt.close()

# Salvar dados IDF
df_idf = pd.DataFrame(intensidades_idf, 
                      index=[f'{d}d' for d in duracoes],
                      columns=[f'TR_{tr}anos' for tr in periodos_retorno_idf])
df_idf.to_csv(DADOS_DIR / "curvas_idf.csv")
print("✓ Dados salvos: curvas_idf.csv")

print(f"\nTabela de Intensidades IDF (mm/h):")
print(df_idf.round(3))

# =============================================================================
# RELATÓRIO FINAL
# =============================================================================
print("\n" + "=" * 80)
print("RELATÓRIO FINAL")
print("=" * 80)

relatorio = f"""
ANÁLISE AVANÇADA DE SÉRIES TEMPORAIS CLIMÁTICAS - RELATÓRIO FINAL
Data: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

1. DECOMPOSIÇÃO DA SÉRIE TEMPORAL
   - Tendência média: {decomposicao.trend.mean():.2f} mm
   - Amplitude sazonal: {decomposicao.seasonal.max() - decomposicao.seasonal.min():.2f} mm
   - Desvio padrão dos resíduos: {decomposicao.resid.std():.2f} mm

2. TESTE DE ESTACIONARIEDADE
   - Série Original: {'ESTACIONÁRIA' if adf_original[1] < 0.05 else 'NÃO ESTACIONÁRIA'} (p={adf_original[1]:.4f})
   - Série Diferenciada: {'ESTACIONÁRIA' if adf_diff[1] < 0.05 else 'NÃO ESTACIONÁRIA'} (p={adf_diff[1]:.4f})
   - Série Dif. Sazonal: {'ESTACIONÁRIA' if adf_sazonal[1] < 0.05 else 'NÃO ESTACIONÁRIA'} (p={adf_sazonal[1]:.4f})

3. MODELO SARIMA
   - AIC: {resultado_sarima.aic:.2f}
   - BIC: {resultado_sarima.bic:.2f}
   - Log-Likelihood: {resultado_sarima.llf:.2f}

4. ANÁLISE DE EXTREMOS (GEV)
   - Parâmetro shape (c): {gev_params[0]:.4f}
   - Parâmetro location: {gev_params[1]:.4f}
   - Parâmetro scale: {gev_params[2]:.4f}
   
   Valores de Retorno:
"""

for periodo, valor in zip(periodos_retorno, valores_retorno):
    relatorio += f"   - TR={periodo} anos: {valor:.2f} mm\n"

relatorio += f"""
5. EVENTOS EXTREMOS
   - Limiar P95: {limiar_p95:.2f} mm
   - Total de eventos: {len(resumo_eventos) if not eventos_extremos.empty else 0}
   - Duração média: {resumo_eventos['duracao'].mean():.1f} dias
   - Precipitação total média: {resumo_eventos['precipitacao_total'].mean():.2f} mm

6. ÍNDICES DE EROSIVIDADE (EI30)
   - EI30 mensal médio: {ei30_mensal.mean():.2f} MJ·mm/ha·h
   - EI30 anual médio: {ei30_anual.mean():.2f} MJ·mm/ha·h
   - Máximo mensal: {ei30_mensal.max():.2f} MJ·mm/ha·h

7. ARQUIVOS GERADOS
   Figuras (8 arquivos):
   - 1_decomposicao_serie_temporal.png
   - 2_teste_estacionariedade.png
   - 3_diferenciacao_sazonal_acf_pacf.png
   - 4_previsao_sarima.png
   - 4_diagnostico_sarima.png
   - 5_analise_extremos_gev.png
   - 6_eventos_extremos.png
   - 7_indices_erosividade_ei30.png
   - 7_ei30_sazonalidade.png
   - 8_curvas_idf.png
   
   Dados (6 arquivos):
   - valores_retorno_gev.csv
   - eventos_extremos_detalhados.csv
   - ei30_mensal.csv
   - ei30_anual.csv
   - curvas_idf.csv
"""

# Salvar relatório
with open(DADOS_DIR / "relatorio_completo.txt", 'w', encoding='utf-8') as f:
    f.write(relatorio)

print(relatorio)
print("\n✓ Relatório salvo: relatorio_completo.txt")
print("\n" + "=" * 80)
print("ANÁLISE CONCLUÍDA COM SUCESSO!")
print("=" * 80)
print(f"\nTodos os resultados salvos em:")
print(f"  Figuras: {FIGURAS_DIR}")
print(f"  Dados: {DADOS_DIR}")
