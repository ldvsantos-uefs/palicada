"""
Análise Estatística: Comparação de Modelos e Teste de Mediação
================================================================
Compara Modelo 1 (Sedimentação ~ Precipitação) vs Modelo 2 (Sedimentação ~ EI30)
e testa mediação: Precipitação → EI30 → Sedimentação
"""

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.metrics import r2_score

def calculate_aic(n, rss, k):
    """
    Calcula AIC (Akaike Information Criterion)
    n: número de observações
    rss: soma dos quadrados residuais
    k: número de parâmetros (incluindo intercepto)
    """
    if rss <= 0:
        return np.inf
    aic = n * np.log(rss / n) + 2 * k
    return aic

def mediation_analysis(df, x_col, m_col, y_col):
    """
    Teste de mediação Baron & Kenny (1986)
    X → M → Y
    """
    # Remove NaN
    df_clean = df[[x_col, m_col, y_col]].dropna()
    
    # Passo 1: X → Y (efeito total c)
    X1 = sm.add_constant(df_clean[x_col])
    model_c = sm.OLS(df_clean[y_col], X1).fit()
    c = model_c.params[x_col]
    p_c = model_c.pvalues[x_col]
    
    # Passo 2: X → M (caminho a)
    model_a = sm.OLS(df_clean[m_col], X1).fit()
    a = model_a.params[x_col]
    p_a = model_a.pvalues[x_col]
    
    # Passo 3: X + M → Y (efeito direto c' e caminho b)
    X2 = sm.add_constant(df_clean[[x_col, m_col]])
    model_b = sm.OLS(df_clean[y_col], X2).fit()
    c_prime = model_b.params[x_col]
    b = model_b.params[m_col]
    p_c_prime = model_b.pvalues[x_col]
    p_b = model_b.pvalues[m_col]
    
    # Efeito indireto (mediação)
    indirect_effect = a * b
    
    # Proporção mediada
    if c != 0:
        proportion_mediated = (c - c_prime) / c
    else:
        proportion_mediated = np.nan
    
    return {
        'total_effect_c': c,
        'p_total': p_c,
        'path_a': a,
        'p_a': p_a,
        'path_b': b,
        'p_b': p_b,
        'direct_effect_c_prime': c_prime,
        'p_direct': p_c_prime,
        'indirect_effect': indirect_effect,
        'proportion_mediated': proportion_mediated
    }

def main():
    # Carregar dados
    df = pd.read_csv(r"C:\Users\vidal\OneDrive\Documentos\13 - CLONEGIT\artigo-posdoc\3-EROSIBIDADE\2-DADOS\CLIMATOLOGIA_20ANOS\dados\dados_integrados_sedimentacao.csv")
    
    # Propagar RAINFALL e EI30 de SUP para outros segmentos
    climate_map = df[df['AREA'] == 'SUP'][['DATA', 'RAINFALL', 'EI30']].set_index('DATA')
    
    def fill_climate(row):
        if pd.isna(row['RAINFALL']) or pd.isna(row['EI30']):
            if row['DATA'] in climate_map.index:
                row['RAINFALL'] = climate_map.loc[row['DATA'], 'RAINFALL']
                row['EI30'] = climate_map.loc[row['DATA'], 'EI30']
        return row

    df = df.apply(fill_climate, axis=1)
    
    # Filtrar dados válidos (FRACIONADO >= 0 para análise de regressão)
    df_reg = df[df['FRACIONADO'] >= 0].dropna(subset=['RAINFALL', 'EI30', 'FRACIONADO'])
    
    print("="*80)
    print("ANÁLISE COMPARATIVA DE MODELOS E TESTE DE MEDIAÇÃO")
    print("="*80)
    
    # =========================================================================
    # 1. ANÁLISE GLOBAL (todos os segmentos juntos)
    # =========================================================================
    print("\n" + "="*80)
    print("1. ANÁLISE GLOBAL (todos os segmentos)")
    print("="*80)
    
    # Modelo 1: Sedimentação ~ Precipitação
    X_rainfall = sm.add_constant(df_reg['RAINFALL'])
    model1 = sm.OLS(df_reg['FRACIONADO'], X_rainfall).fit()
    
    # Modelo 2: Sedimentação ~ EI30
    X_ei30 = sm.add_constant(df_reg['EI30'])
    model2 = sm.OLS(df_reg['FRACIONADO'], X_ei30).fit()
    
    # Cálculo de métricas
    n = len(df_reg)
    
    # R² e R² ajustado
    r2_m1 = model1.rsquared
    r2_adj_m1 = model1.rsquared_adj
    r2_m2 = model2.rsquared
    r2_adj_m2 = model2.rsquared_adj
    
    # AIC
    aic_m1 = model1.aic
    aic_m2 = model2.aic
    delta_aic = aic_m1 - aic_m2
    
    # Correlações de Pearson
    r_rainfall, p_rainfall = stats.pearsonr(df_reg['RAINFALL'], df_reg['FRACIONADO'])
    r_ei30, p_ei30 = stats.pearsonr(df_reg['EI30'], df_reg['FRACIONADO'])
    
    print("\nModelo 1: Sedimentação ~ Precipitação")
    print(f"  Correlação de Pearson: r = {r_rainfall:.4f}, p = {p_rainfall:.4f}")
    print(f"  R² = {r2_m1:.4f}")
    print(f"  R² ajustado = {r2_adj_m1:.4f}")
    print(f"  AIC = {aic_m1:.2f}")
    print(f"  Coeficiente: β = {model1.params['RAINFALL']:.6f} (p = {model1.pvalues['RAINFALL']:.4f})")
    
    print("\nModelo 2: Sedimentação ~ EI30")
    print(f"  Correlação de Pearson: r = {r_ei30:.4f}, p = {p_ei30:.4f}")
    print(f"  R² = {r2_m2:.4f}")
    print(f"  R² ajustado = {r2_adj_m2:.4f}")
    print(f"  AIC = {aic_m2:.2f}")
    print(f"  Coeficiente: β = {model2.params['EI30']:.6f} (p = {model2.pvalues['EI30']:.4f})")
    
    print("\nComparação:")
    print(f"  ΔAIC = {delta_aic:.2f} (Modelo 1 - Modelo 2)")
    if delta_aic > 2:
        print(f"  → Modelo 2 (EI30) é substancialmente melhor (ΔAIC > 2)")
    elif delta_aic < -2:
        print(f"  → Modelo 1 (Precipitação) é substancialmente melhor (ΔAIC < -2)")
    else:
        print(f"  → Modelos equivalentes (|ΔAIC| ≤ 2)")
    
    print(f"\n  Melhoria em R²: {(r2_m2 - r2_m1)*100:.2f}%")
    print(f"  Melhoria em R² ajustado: {(r2_adj_m2 - r2_adj_m1)*100:.2f}%")
    
    # =========================================================================
    # 2. TESTE DE MEDIAÇÃO: Precipitação → EI30 → Sedimentação
    # =========================================================================
    print("\n" + "="*80)
    print("2. TESTE DE MEDIAÇÃO (Baron & Kenny, 1986)")
    print("   Precipitação → EI30 → Sedimentação")
    print("="*80)
    
    med_results = mediation_analysis(df_reg, 'RAINFALL', 'EI30', 'FRACIONADO')
    
    print(f"\nPasso 1: X (Precipitação) → Y (Sedimentação)")
    print(f"  Efeito total (c): {med_results['total_effect_c']:.6f} (p = {med_results['p_total']:.4f})")
    
    print(f"\nPasso 2: X (Precipitação) → M (EI30)")
    print(f"  Caminho a: {med_results['path_a']:.6f} (p = {med_results['p_a']:.4f})")
    
    print(f"\nPasso 3: X (Precipitação) + M (EI30) → Y (Sedimentação)")
    print(f"  Caminho b (EI30 → Sedimentação): {med_results['path_b']:.6f} (p = {med_results['p_b']:.4f})")
    print(f"  Efeito direto (c'): {med_results['direct_effect_c_prime']:.6f} (p = {med_results['p_direct']:.4f})")
    
    print(f"\nEfeito indireto (mediação): {med_results['indirect_effect']:.6f}")
    print(f"Proporção mediada: {med_results['proportion_mediated']*100:.1f}%")
    
    # Interpretação
    print("\nInterpretação:")
    if med_results['p_a'] < 0.05 and med_results['p_b'] < 0.05:
        if med_results['p_direct'] >= 0.05:
            print("  → MEDIAÇÃO COMPLETA: Precipitação afeta Sedimentação completamente via EI30")
        else:
            print("  → MEDIAÇÃO PARCIAL: Precipitação afeta Sedimentação parcialmente via EI30")
            print(f"    (~{med_results['proportion_mediated']*100:.0f}% do efeito é mediado)")
    else:
        print("  → Sem evidência de mediação significativa")
    
    # =========================================================================
    # 3. ANÁLISE POR SEGMENTO
    # =========================================================================
    print("\n" + "="*80)
    print("3. ANÁLISE POR SEGMENTO (SUP, MED, INF)")
    print("="*80)
    
    for area in ['SUP', 'MED', 'INF']:
        df_area = df_reg[df_reg['AREA'] == area]
        
        if len(df_area) < 10:
            print(f"\n{area}: Dados insuficientes (n={len(df_area)})")
            continue
            
        print(f"\n{'-'*60}")
        print(f"Segmento: {area} (n={len(df_area)})")
        print(f"{'-'*60}")
        
        # Modelo 1: Precipitação
        X1 = sm.add_constant(df_area['RAINFALL'])
        m1 = sm.OLS(df_area['FRACIONADO'], X1).fit()
        
        # Modelo 2: EI30
        X2 = sm.add_constant(df_area['EI30'])
        m2 = sm.OLS(df_area['FRACIONADO'], X2).fit()
        
        # Correlações
        r_rain = df_area[['RAINFALL', 'FRACIONADO']].corr().iloc[0, 1]
        r_ei = df_area[['EI30', 'FRACIONADO']].corr().iloc[0, 1]
        
        print(f"\nModelo 1 (Precipitação):")
        print(f"  r = {r_rain:.4f}, R² = {m1.rsquared:.4f}, R² adj = {m1.rsquared_adj:.4f}")
        print(f"  AIC = {m1.aic:.2f}")
        
        print(f"\nModelo 2 (EI30):")
        print(f"  r = {r_ei:.4f}, R² = {m2.rsquared:.4f}, R² adj = {m2.rsquared_adj:.4f}")
        print(f"  AIC = {m2.aic:.2f}")
        
        print(f"\nΔAIC = {m1.aic - m2.aic:.2f}")
        if (m1.aic - m2.aic) > 2:
            print(f"  → EI30 é melhor preditor para {area}")
        elif (m1.aic - m2.aic) < -2:
            print(f"  → Precipitação é melhor preditor para {area}")
        else:
            print(f"  → Modelos equivalentes para {area}")
    
    # =========================================================================
    # 4. VERIFICAÇÃO DE MULTICOLINEARIDADE (VIF)
    # =========================================================================
    print("\n" + "="*80)
    print("4. VERIFICAÇÃO DE MULTICOLINEARIDADE (VIF)")
    print("="*80)
    
    # Calcular VIF se usássemos os dois preditores juntos (NÃO RECOMENDADO!)
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    X_both = df_reg[['RAINFALL', 'EI30']].dropna()
    
    # VIF para Precipitação
    vif_rainfall = variance_inflation_factor(X_both.values, 0)
    # VIF para EI30
    vif_ei30 = variance_inflation_factor(X_both.values, 1)
    
    print(f"\nSe usássemos ambos preditores simultaneamente:")
    print(f"  VIF (Precipitação) = {vif_rainfall:.2f}")
    print(f"  VIF (EI30) = {vif_ei30:.2f}")
    
    if vif_rainfall > 10 or vif_ei30 > 10:
        print("\n  ⚠️  VIF > 10: MULTICOLINEARIDADE SEVERA!")
        print("  → Justifica a estratégia de modelos univariados separados")
    elif vif_rainfall > 5 or vif_ei30 > 5:
        print("\n  ⚠️  VIF > 5: MULTICOLINEARIDADE MODERADA")
        print("  → Justifica a estratégia de modelos univariados separados")
    else:
        print("\n  ✓ VIF < 5: Sem multicolinearidade severa")
    
    # Correlação entre os preditores
    r_pred, p_pred = stats.pearsonr(X_both['RAINFALL'], X_both['EI30'])
    print(f"\nCorrelação entre Precipitação e EI30:")
    print(f"  r = {r_pred:.4f} (p = {p_pred:.4f})")
    print(f"  R² = {r_pred**2:.4f} (EI30 explica {r_pred**2*100:.1f}% da variância de Precipitação)")
    
    print("\n" + "="*80)
    print("ANÁLISE CONCLUÍDA")
    print("="*80)

if __name__ == "__main__":
    main()
