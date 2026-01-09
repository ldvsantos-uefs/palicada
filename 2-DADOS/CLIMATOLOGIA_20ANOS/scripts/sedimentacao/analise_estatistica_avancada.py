import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scipy.stats as stats
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

def cohen_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)

# Load data
df = pd.read_csv('CLIMATOLOGIA_20ANOS/dados/dados_integrados_sedimentacao.csv')

# Clean data
df = df[['AREA', 'FRACIONADO']].dropna()
# Ensure AREA is treated as categorical
df['AREA'] = df['AREA'].astype('category')

# Descriptive stats
print("Descriptive Statistics:")
print(df.groupby('AREA')['FRACIONADO'].agg(['mean', 'std', 'count']))

# ANOVA
model = ols('FRACIONADO ~ AREA', data=df).fit()
aov_table = sm.stats.anova_lm(model, typ=2)
print("\nANOVA Table:")
print(aov_table)

# Calculate Partial Eta Squared (for one-way ANOVA, it's SS_effect / (SS_effect + SS_error))
ss_effect = aov_table.loc['AREA', 'sum_sq']
ss_error = aov_table.loc['Residual', 'sum_sq']
eta_squared = ss_effect / (ss_effect + ss_error)
print(f"\nEffect Size (Partial Eta Squared): {eta_squared:.4f}")

# Pairwise Comparisons (Manual for Cohen's d and CI)
groups = df['AREA'].unique()
import itertools

print("\nPairwise Comparisons (Format: DM; 95% CI; p-value; d):")
for g1, g2 in itertools.combinations(groups, 2):
    data1 = df[df['AREA'] == g1]['FRACIONADO']
    data2 = df[df['AREA'] == g2]['FRACIONADO']
    
    # T-test independent
    t_stat, p_val = stats.ttest_ind(data1, data2)
    
    # Mean Difference
    diff = np.mean(data1) - np.mean(data2)
    
    # CI of Difference
    cm = sm.stats.CompareMeans(sm.stats.DescrStatsW(data1), sm.stats.DescrStatsW(data2))
    ci = cm.tconfint_diff(usevar='pooled')
    
    # Cohen's d
    d_val = cohen_d(data1, data2)
    
    print(f"{g1} vs {g2}:")
    print(f"  DM = {diff:.4f}")
    print(f"  95% CI [{ci[0]:.4f}; {ci[1]:.4f}]")
    print(f"  p = {p_val:.4f}")
    print(f"  d = {d_val:.4f}")
    print("-" * 30)
