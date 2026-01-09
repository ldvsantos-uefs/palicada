import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import glm
from statsmodels.genmod.families import Gaussian, Gamma
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Load data
df = pd.read_csv('CLIMATOLOGIA_20ANOS/dados/dados_integrados_sedimentacao.csv')

# Filter for relevant columns
df = df[['AREA', 'FRACIONADO', 'SEDIMENT']]

# Descriptive statistics
desc = df.groupby('AREA')['FRACIONADO'].agg(['mean', 'std', 'sem', 'count'])
print("Descriptive Statistics (Monthly Incremental Sedimentation):")
print(desc)

# Total accumulation
total = df.groupby('AREA')['SEDIMENT'].max()
print("\nTotal Accumulation (cm):")
print(total)

# Statistical Test (ANOVA) on Incremental Sedimentation
# We want to see if the rate of deposition differs by segment
areas = sorted(df['AREA'].dropna().unique().tolist())
groups = [df.loc[df['AREA'] == a, 'FRACIONADO'] for a in areas]
if all(len(g) >= 2 for g in groups):
    f_val, p_val = stats.f_oneway(*groups)
    print(f"\nANOVA One-way (Fracionado ~ Area): F={f_val:.4f}, p={p_val:.4f}")
else:
    print("\nANOVA One-way (Fracionado ~ Area): amostra insuficiente em um ou mais grupos")

# Post-hoc Tukey
tukey = pairwise_tukeyhsd(endog=df['FRACIONADO'], groups=df['AREA'], alpha=0.05)
print("\nTukey HSD:")
print(tukey)

# GLM Analysis
# Model: Fracionado ~ Area
# Using Gaussian family as a start
model = glm('FRACIONADO ~ AREA', data=df, family=Gaussian()).fit()
print("\nGLM Summary:")
print(model.summary())

# Prepare text for the user
# Format: Mean ± SE (Letter)
# We need to assign letters based on Tukey
# Logic: If Tukey rejects null, they are different.
# Let's manually interpret the Tukey results for the report.
