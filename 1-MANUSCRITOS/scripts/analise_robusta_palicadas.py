"""
Análises robustas para responder ao objetivo do trabalho:
Avaliar a eficiência de paliçadas de bambu no controle de erosão em ravinas

Análises implementadas:
1. Regressão Múltipla Hierárquica
2. Modelo Preditivo com Random Forest
3. Análise de Eficiência por Segmento
4. Simulação de Capacidade Residual e Tempo até Saturação
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CARREGAR DADOS
# ============================================================================
print("=" * 80)
print("ANÁLISES ROBUSTAS: EFICIÊNCIA DE PALIÇADAS DE BAMBU")
print("=" * 80)

# Caminho do arquivo
file_path = '2-DADOS/CLIMATOLOGIA_20ANOS/dados/dados_integrados_sedimentacao.csv'

# Carregar dados
df = pd.read_csv(file_path)

# Verificar estrutura
print("\nEstrutura dos dados:")
print(df.head())
print(f"\nDimensões: {df.shape[0]} observações, {df.shape[1]} variáveis")
print(f"Colunas: {list(df.columns)}")

# Preparar variáveis
df['DATA'] = pd.to_datetime(df['DATA'])
df['MES'] = df['DATA'].dt.month
df['ANO'] = df['DATA'].dt.year

# Criar trimestre
df['TRIMESTRE'] = pd.cut(df['MES'], 
                          bins=[0, 3, 6, 9, 12], 
                          labels=['Q1', 'Q2', 'Q3', 'Q4'])

# Criar lag de precipitação (defasagem de 1 mês)
df_sorted = df.sort_values(['AREA', 'DATA'])
df_sorted['RAINFALL_LAG1'] = df_sorted.groupby('AREA')['RAINFALL'].shift(1)

# Estatísticas descritivas
print("\n" + "=" * 80)
print("ESTATÍSTICAS DESCRITIVAS")
print("=" * 80)
print("\nPor segmento:")
print(df.groupby('AREA')[['RAINFALL', 'FRACIONADO']].describe().round(4))

# ============================================================================
# ANÁLISE 1: REGRESSÃO MÚLTIPLA HIERÁRQUICA
# ============================================================================
print("\n" + "=" * 80)
print("ANÁLISE 1: REGRESSÃO MÚLTIPLA HIERÁRQUICA")
print("=" * 80)
print("\nObjetivo: Modelar sedimentação considerando múltiplos fatores")
print("Variáveis: Precipitação + Lag + Segmento + Trimestre")

# Preparar dados (remover NaN do lag)
df_reg = df_sorted.dropna(subset=['RAINFALL_LAG1']).copy()

# Criar dummies ANTES de selecionar y
df_reg = pd.get_dummies(df_reg, columns=['AREA', 'TRIMESTRE'], drop_first=True, dtype=float)

# Garantir que não há colunas object
segment_cols = [col for col in df_reg.columns if 'AREA_' in col]
trimestre_cols = [col for col in df_reg.columns if 'TRIMESTRE_' in col]

# Verificar tipos
for col in segment_cols + trimestre_cols:
    df_reg[col] = df_reg[col].astype(float)

# Variável dependente
y = df_reg['FRACIONADO'].values

# Modelo 1: Apenas Precipitação
X1 = df_reg[['RAINFALL']].values
X1 = sm.add_constant(X1)

model1 = sm.OLS(y, X1).fit()
print("\n" + "-" * 80)
print("MODELO 1: Sedimentação ~ Precipitação")
print("-" * 80)
print(f"R² = {model1.rsquared:.4f}")
print(f"R² ajustado = {model1.rsquared_adj:.4f}")
print(f"AIC = {model1.aic:.2f}")
print(f"BIC = {model1.bic:.2f}")

# Modelo 2: Precipitação + Lag
X2 = df_reg[['RAINFALL', 'RAINFALL_LAG1']].values
X2 = sm.add_constant(X2)

model2 = sm.OLS(y, X2).fit()
print("\n" + "-" * 80)
print("MODELO 2: Sedimentação ~ Precipitação + Lag")
print("-" * 80)
print(f"R² = {model2.rsquared:.4f}")
print(f"R² ajustado = {model2.rsquared_adj:.4f}")
print(f"AIC = {model2.aic:.2f}")
print(f"BIC = {model2.bic:.2f}")
print(f"Incremento R²: {model2.rsquared - model1.rsquared:.4f}")

# Modelo 3: Precipitação + Lag + Segmento
X3 = df_reg[['RAINFALL', 'RAINFALL_LAG1'] + segment_cols].values
X3 = sm.add_constant(X3)

model3 = sm.OLS(y, X3).fit()
print("\n" + "-" * 80)
print("MODELO 3: Sedimentação ~ Precipitação + Lag + Segmento")
print("-" * 80)
print(f"R² = {model3.rsquared:.4f}")
print(f"R² ajustado = {model3.rsquared_adj:.4f}")
print(f"AIC = {model3.aic:.2f}")
print(f"BIC = {model3.bic:.2f}")
print(f"Incremento R²: {model3.rsquared - model2.rsquared:.4f}")

# Modelo 4: Modelo Completo (Precipitação + Lag + Segmento + Trimestre)
X4 = df_reg[['RAINFALL', 'RAINFALL_LAG1'] + segment_cols + trimestre_cols].values
X4 = sm.add_constant(X4)

model4 = sm.OLS(y, X4).fit()
print("\n" + "-" * 80)
print("MODELO 4: MODELO COMPLETO")
print("Sedimentação ~ Precipitação + Lag + Segmento + Trimestre")
print("-" * 80)
print(f"R² = {model4.rsquared:.4f}")
print(f"R² ajustado = {model4.rsquared_adj:.4f}")
print(f"AIC = {model4.aic:.2f}")
print(f"BIC = {model4.bic:.2f}")
print(f"Incremento R²: {model4.rsquared - model3.rsquared:.4f}")

print("\nCoeficientes do modelo completo:")
print(model4.summary2().tables[1])

# Comparação de modelos
print("\n" + "-" * 80)
print("COMPARAÇÃO DE MODELOS (Teste F)")
print("-" * 80)
print(f"Modelo 2 vs Modelo 1: ", end="")
ftest_21 = anova_lm(model1, model2)
print(f"F = {ftest_21['F'][1]:.3f}, p = {ftest_21['Pr(>F)'][1]:.4f}")

print(f"Modelo 3 vs Modelo 2: ", end="")
ftest_32 = anova_lm(model2, model3)
print(f"F = {ftest_32['F'][1]:.3f}, p = {ftest_32['Pr(>F)'][1]:.4f}")

print(f"Modelo 4 vs Modelo 3: ", end="")
ftest_43 = anova_lm(model3, model4)
print(f"F = {ftest_43['F'][1]:.3f}, p = {ftest_43['Pr(>F)'][1]:.4f}")

# Variância explicada por cada componente
print("\n" + "-" * 80)
print("DECOMPOSIÇÃO DA VARIÂNCIA EXPLICADA")
print("-" * 80)
print(f"Precipitação apenas:           R² = {model1.rsquared:.4f} ({model1.rsquared*100:.1f}%)")
print(f"+ Lag:                         ΔR² = {model2.rsquared - model1.rsquared:.4f} ({(model2.rsquared - model1.rsquared)*100:.1f}%)")
print(f"+ Segmento:                    ΔR² = {model3.rsquared - model2.rsquared:.4f} ({(model3.rsquared - model2.rsquared)*100:.1f}%)")
print(f"+ Trimestre:                   ΔR² = {model4.rsquared - model3.rsquared:.4f} ({(model4.rsquared - model3.rsquared)*100:.1f}%)")
print(f"Total explicado (Modelo 4):    R² = {model4.rsquared:.4f} ({model4.rsquared*100:.1f}%)")
print(f"Não explicado:                     = {1-model4.rsquared:.4f} ({(1-model4.rsquared)*100:.1f}%)")

# ============================================================================
# ANÁLISE 2: MODELO PREDITIVO COM RANDOM FOREST
# ============================================================================
print("\n" + "=" * 80)
print("ANÁLISE 2: MODELO PREDITIVO (RANDOM FOREST)")
print("=" * 80)
print("\nObjetivo: Prever sedimentação com validação cruzada")

# Preparar features
feature_cols = ['RAINFALL', 'RAINFALL_LAG1', 'MES'] + segment_cols
X_rf = df_reg[feature_cols].values
y_rf = df_reg['FRACIONADO'].values

# Criar modelo Random Forest
rf_model = RandomForestRegressor(
    n_estimators=500,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

# Validação cruzada (K-Fold) - sem paralelização no Windows
cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf_model, X_rf, y_rf, cv=cv, scoring='r2')

print("\n" + "-" * 80)
print("DESEMPENHO DO RANDOM FOREST (Validação Cruzada 5-Fold)")
print("-" * 80)
print(f"R² médio: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"R² por fold: {[f'{s:.4f}' for s in cv_scores]}")

# Treinar modelo completo
rf_model.fit(X_rf, y_rf)
y_pred_rf = rf_model.predict(X_rf)

# Métricas no conjunto completo
r2_full = r2_score(y_rf, y_pred_rf)
rmse = np.sqrt(mean_squared_error(y_rf, y_pred_rf))
mae = mean_absolute_error(y_rf, y_pred_rf)

print("\n" + "-" * 80)
print("DESEMPENHO NO CONJUNTO COMPLETO")
print("-" * 80)
print(f"R² = {r2_full:.4f}")
print(f"RMSE = {rmse:.4f} cm")
print(f"MAE = {mae:.4f} cm")

# Importância das features
importances = rf_model.feature_importances_
feature_names = ['Precipitação', 'Precipitação Lag1', 'Mês'] + \
                [col.replace('AREA_', 'Segmento ') for col in segment_cols]

print("\n" + "-" * 80)
print("IMPORTÂNCIA DAS VARIÁVEIS")
print("-" * 80)
importance_df = pd.DataFrame({
    'Variável': feature_names,
    'Importância': importances
}).sort_values('Importância', ascending=False)
print(importance_df.to_string(index=False))

# ============================================================================
# ANÁLISE 3: EFICIÊNCIA POR SEGMENTO
# ============================================================================
print("\n" + "=" * 80)
print("ANÁLISE 3: EFICIÊNCIA DAS PALIÇADAS POR SEGMENTO")
print("=" * 80)
print("\nObjetivo: Quantificar desempenho relativo entre segmentos")

# Calcular métricas de eficiência
efficiency_metrics = []

for segment in ['SUP', 'MED', 'INF']:
    df_seg = df[df['AREA'] == segment].copy()
    
    # Sedimentação total e média
    sed_total = df_seg['FRACIONADO'].sum()
    sed_media = df_seg['FRACIONADO'].mean()
    
    # Precipitação total e média
    rain_total = df_seg['RAINFALL'].sum()
    rain_media = df_seg['RAINFALL'].mean()
    
    # Eficiência (sedimentação por unidade de chuva)
    efficiency = sed_total / rain_total if rain_total > 0 else 0
    
    # Coeficiente de variação
    cv = df_seg['FRACIONADO'].std() / df_seg['FRACIONADO'].mean() * 100
    
    # Correlação chuva-sedimentação
    corr, pval = stats.pearsonr(df_seg['RAINFALL'], df_seg['FRACIONADO'])
    
    # Regressão linear simples
    X_seg = sm.add_constant(df_seg[['RAINFALL']])
    model_seg = sm.OLS(df_seg['FRACIONADO'], X_seg).fit()
    
    # Contribuição relativa no total
    contrib = (sed_total / df['FRACIONADO'].sum()) * 100
    
    efficiency_metrics.append({
        'Segmento': segment,
        'Sedimentação Total (cm)': sed_total,
        'Sedimentação Média (cm/mês)': sed_media,
        'Precipitação Total (mm)': rain_total,
        'Precipitação Média (mm/mês)': rain_media,
        'Eficiência (cm/mm × 10⁴)': efficiency * 10000,
        'CV (%)': cv,
        'Correlação (r)': corr,
        'p-valor': pval,
        'R²': model_seg.rsquared,
        'Contribuição Total (%)': contrib
    })

efficiency_df = pd.DataFrame(efficiency_metrics)

print("\n" + "-" * 80)
print("MÉTRICAS DE DESEMPENHO POR SEGMENTO")
print("-" * 80)
print(efficiency_df.to_string(index=False))

# Teste de diferenças entre segmentos (ANOVA)
print("\n" + "-" * 80)
print("TESTE DE DIFERENÇAS ENTRE SEGMENTOS (ANOVA)")
print("-" * 80)

# ANOVA para sedimentação
f_stat, p_val = stats.f_oneway(
    df[df['AREA'] == 'SUP']['FRACIONADO'],
    df[df['AREA'] == 'MED']['FRACIONADO'],
    df[df['AREA'] == 'INF']['FRACIONADO']
)
print(f"Sedimentação entre segmentos: F = {f_stat:.3f}, p = {p_val:.4f}")

if p_val < 0.05:
    print("Há diferenças significativas entre segmentos (p < 0.05)")
else:
    print("Não há diferenças significativas entre segmentos (p ≥ 0.05)")

# ============================================================================
# ANÁLISE 4: SIMULAÇÃO DE CAPACIDADE RESIDUAL
# ============================================================================
print("\n" + "=" * 80)
print("ANÁLISE 4: CAPACIDADE RESIDUAL E PROJEÇÃO DE SATURAÇÃO")
print("=" * 80)
print("\nObjetivo: Projetar tempo até saturação das paliçadas")

# Parâmetros observados (do manuscrito)
capacidade_maxima = {
    'SUP': 50.0,  # cm
    'MED': 76.0,  # cm
    'INF': 36.0   # cm
}

# Acumulado atual no período monitorado
acumulado_atual = df.groupby('AREA')['FRACIONADO'].sum().to_dict()

# Capacidade residual
print("\n" + "-" * 80)
print("CAPACIDADE DE RETENÇÃO")
print("-" * 80)
for segment in ['SUP', 'MED', 'INF']:
    cap_max = capacidade_maxima[segment]
    acum = acumulado_atual[segment]
    residual = cap_max - acum
    pct_ocupado = (acum / cap_max) * 100
    
    print(f"\n{segment}:")
    print(f"  Capacidade máxima:      {cap_max:.2f} cm")
    print(f"  Acumulado atual:        {acum:.2f} cm")
    print(f"  Capacidade residual:    {residual:.2f} cm")
    print(f"  Ocupação:               {pct_ocupado:.1f}%")

# Projeção de tempo até saturação
print("\n" + "-" * 80)
print("PROJEÇÃO DE TEMPO ATÉ SATURAÇÃO")
print("-" * 80)

# Taxa de deposição mensal média por segmento
taxa_deposicao = df.groupby('AREA')['FRACIONADO'].mean().to_dict()

# Cenários de precipitação (P50, P75, P90, P95)
percentiles = df['RAINFALL'].quantile([0.50, 0.75, 0.90, 0.95])
print(f"\nLimiares de precipitação mensal:")
print(f"  P50 = {percentiles[0.50]:.2f} mm/mês")
print(f"  P75 = {percentiles[0.75]:.2f} mm/mês")
print(f"  P90 = {percentiles[0.90]:.2f} mm/mês")
print(f"  P95 = {percentiles[0.95]:.2f} mm/mês")

# Para cada segmento, calcular tempo até saturação
print("\n" + "-" * 80)
print("TEMPO ESTIMADO ATÉ SATURAÇÃO (meses)")
print("-" * 80)

# Usar modelo de regressão para projetar deposição sob diferentes cenários
projections = []

for segment in ['SUP', 'MED', 'INF']:
    df_seg = df[df['AREA'] == segment].copy()
    
    # Modelo de regressão linear para este segmento
    X_seg = sm.add_constant(df_seg[['RAINFALL']])
    model_seg = sm.OLS(df_seg['FRACIONADO'], X_seg).fit()
    
    residual = capacidade_maxima[segment] - acumulado_atual[segment]
    
    for p_label, p_val in [('P50', percentiles[0.50]), 
                            ('P75', percentiles[0.75]),
                            ('P90', percentiles[0.90]), 
                            ('P95', percentiles[0.95])]:
        # Prever deposição mensal sob este cenário
        pred_deposition = model_seg.predict([1, p_val])[0]
        
        if pred_deposition > 0:
            months_to_full = residual / pred_deposition
        else:
            months_to_full = np.inf
        
        projections.append({
            'Segmento': segment,
            'Cenário': p_label,
            'Precipitação (mm/mês)': p_val,
            'Deposição prevista (cm/mês)': pred_deposition,
            'Capacidade residual (cm)': residual,
            'Tempo até saturação (meses)': months_to_full
        })

proj_df = pd.DataFrame(projections)
print(proj_df.to_string(index=False))

# Cenário crítico (média dos últimos 6 meses)
print("\n" + "-" * 80)
print("CENÁRIO CRÍTICO: Taxa de deposição dos últimos 6 meses")
print("-" * 80)

df_recent = df.sort_values('DATA').tail(18)  # últimos 6 meses × 3 segmentos
taxa_recent = df_recent.groupby('AREA')['FRACIONADO'].mean().to_dict()

for segment in ['SUP', 'MED', 'INF']:
    residual = capacidade_maxima[segment] - acumulado_atual[segment]
    taxa = taxa_recent.get(segment, taxa_deposicao[segment])
    
    if taxa > 0:
        months_critical = residual / taxa
    else:
        months_critical = np.inf
    
    print(f"\n{segment}:")
    print(f"  Taxa recente:           {taxa:.4f} cm/mês")
    print(f"  Capacidade residual:    {residual:.2f} cm")
    print(f"  Tempo até saturação:    {months_critical:.1f} meses")

# ============================================================================
# SÍNTESE FINAL
# ============================================================================
print("\n" + "=" * 80)
print("SÍNTESE DAS ANÁLISES")
print("=" * 80)

print("\n1. REGRESSÃO MÚLTIPLA HIERÁRQUICA:")
print(f"   - Modelo completo explica {model4.rsquared*100:.1f}% da variância")
print(f"   - Precipitação contribui com {model1.rsquared*100:.1f}%")
print(f"   - Lag adiciona {(model2.rsquared - model1.rsquared)*100:.1f}%")
print(f"   - Segmento adiciona {(model3.rsquared - model2.rsquared)*100:.1f}%")
print(f"   - Trimestre adiciona {(model4.rsquared - model3.rsquared)*100:.1f}%")

print("\n2. MODELO PREDITIVO:")
print(f"   - Random Forest: R² = {cv_scores.mean():.4f} (validação cruzada)")
print(f"   - RMSE = {rmse:.4f} cm")
print(f"   - Variável mais importante: {importance_df.iloc[0]['Variável']}")

print("\n3. EFICIÊNCIA POR SEGMENTO:")
for _, row in efficiency_df.iterrows():
    print(f"   - {row['Segmento']}: Eficiência = {row['Eficiência (cm/mm × 10⁴)']:.2f} ×10⁻⁴, " +
          f"R² = {row['R²']:.2f}, Contribuição = {row['Contribuição Total (%)']:.1f}%")

print("\n4. CAPACIDADE RESIDUAL:")
for segment in ['SUP', 'MED', 'INF']:
    residual = capacidade_maxima[segment] - acumulado_atual[segment]
    pct_ocupado = (acumulado_atual[segment] / capacidade_maxima[segment]) * 100
    print(f"   - {segment}: Ocupação = {pct_ocupado:.1f}%, Residual = {residual:.1f} cm")

print("\n" + "=" * 80)
print("ANÁLISES CONCLUÍDAS!")
print("=" * 80)
