"""
FIGURA 21B: DIAGNÓSTICO DE INTENSIDADE DAS CHUVAS
Relação EI30 vs Precipitação e Distribuição da Razão Erosiva
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Configurações
plt.style.use('ggplot')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 300

# Diretórios
BASE_DIR = Path(__file__).parent.parent.parent
FIGURAS_DIR = BASE_DIR / "figuras" / "sedimentacao"
DADOS_DIR = BASE_DIR / "dados"

print("=" * 80)
print("FIGURA 21B: DIAGNÓSTICO DE INTENSIDADE")
print("=" * 80)

# =============================================================================
# CARREGAR DADOS
# =============================================================================
df = pd.read_csv(DADOS_DIR / "dados_integrados_sedimentacao.csv")
df['DATA'] = pd.to_datetime(df['DATA'])
df_valid = df[(df['AREA'] == 'SUP') & (df['RAINFALL'] > 0) & (df['EI30'] > 0)].copy()

# Calcular índice de intensidade
df_valid['RAZAO_EROSIVA'] = df_valid['EI30'] / df_valid['RAINFALL']
limiar_razao = df_valid['RAZAO_EROSIVA'].median()
df_valid['TIPO_CHUVA'] = df_valid['RAZAO_EROSIVA'].apply(
    lambda x: 'TORRENCIAL' if x > limiar_razao else 'PROLONGADA'
)

limiar_ei30 = df_valid['EI30'].quantile(0.95)
eventos_extremos = df_valid[df_valid['EI30'] >= limiar_ei30].copy()
torrenciais = df_valid[df_valid['TIPO_CHUVA'] == 'TORRENCIAL']
prolongadas = df_valid[df_valid['TIPO_CHUVA'] == 'PROLONGADA']

print(f"\n✓ Eventos válidos: {len(df_valid)}")
print(f"✓ Chuvas TORRENCIAIS: {len(torrenciais)} (Razão média = {torrenciais['RAZAO_EROSIVA'].mean():.2f})")
print(f"✓ Chuvas PROLONGADAS: {len(prolongadas)} (Razão média = {prolongadas['RAZAO_EROSIVA'].mean():.2f})")
print(f"✓ Limiar Razão Erosiva: {limiar_razao:.2f}")
print(f"✓ Eventos extremos P95: {len(eventos_extremos)}")

# =============================================================================
# FIGURA COM 2 PAINÉIS
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))

# -------------------------
# PAINEL 1: EI30 × PRECIPITAÇÃO
# -------------------------
# Scatter colorido por tipo
ax1.scatter(torrenciais['RAINFALL'], torrenciais['EI30'],
           s=200, c='red', marker='^', alpha=0.75, edgecolors='darkred',
           linewidths=2, label='TORRENCIAIS')

ax1.scatter(prolongadas['RAINFALL'], prolongadas['EI30'],
           s=200, c='blue', marker='o', alpha=0.75, edgecolors='navy',
           linewidths=2, label='PROLONGADAS')

# Eventos extremos
ax1.scatter(eventos_extremos['RAINFALL'], eventos_extremos['EI30'],
           s=500, c='gold', marker='*', edgecolors='darkorange',
           linewidths=3, label='EXTREMOS P95', zorder=5)

# Linha de tendência
X = df_valid['RAINFALL'].values.reshape(-1, 1)
y = df_valid['EI30'].values
reg = LinearRegression().fit(X, y)
x_pred = np.linspace(df_valid['RAINFALL'].min(), df_valid['RAINFALL'].max(), 100)
y_pred = reg.predict(x_pred.reshape(-1, 1))
r2 = reg.score(X, y)

ax1.plot(x_pred, y_pred, '--', color='black', linewidth=3, 
        label=f'Regressão Linear (R²={r2:.3f})', alpha=0.7)

# Linha do limiar P95
ax1.axhline(y=limiar_ei30, color='orange', linestyle=':', linewidth=3,
           label=f'P95 EI30 = {limiar_ei30:.0f}', alpha=0.7)

# Anotar eventos extremos
for _, evt in eventos_extremos.iterrows():
    ax1.annotate(f"{evt['DATA'].strftime('%Y-%m')}\nRazão={evt['RAZAO_EROSIVA']:.2f}",
                xy=(evt['RAINFALL'], evt['EI30']),
                xytext=(15, 15), textcoords='offset points',
                fontsize=9, fontweight='bold', color='darkorange',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.8),
                arrowprops=dict(arrowstyle='->', color='darkorange', lw=2))

ax1.set_xlabel('Precipitação Total (mm)', fontsize=15, fontweight='bold')
ax1.set_ylabel('Erosividade EI30 (MJ mm ha⁻¹ h⁻¹)', fontsize=15, fontweight='bold')
ax1.set_title('Diagnóstico de Intensidade: EI30 vs Precipitação\n' + 
             'Alta razão EI30/P → Chuva TORRENCIAL | Baixa razão → Chuva PROLONGADA',
             fontsize=14, fontweight='bold', pad=15)
ax1.legend(loc='upper left', fontsize=12, framealpha=0.95)
ax1.grid(True, alpha=0.3)
ax1.tick_params(labelsize=12)

# Adicionar equação de regressão
eq_text = f'EI30 = {reg.coef_[0]:.2f} × P + {reg.intercept_:.2f}'
ax1.text(0.05, 0.95, eq_text, transform=ax1.transAxes,
        fontsize=12, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# -------------------------
# PAINEL 2: DISTRIBUIÇÃO DA RAZÃO EROSIVA
# -------------------------
bins = np.linspace(df_valid['RAZAO_EROSIVA'].min(), 
                   df_valid['RAZAO_EROSIVA'].max(), 20)

# Histogramas
n_torr, bins_torr, patches_torr = ax2.hist(torrenciais['RAZAO_EROSIVA'], 
                                            bins=bins, alpha=0.7, 
                                            color='red', edgecolor='darkred', 
                                            linewidth=2, label='TORRENCIAIS', 
                                            density=True)

n_prol, bins_prol, patches_prol = ax2.hist(prolongadas['RAZAO_EROSIVA'], 
                                            bins=bins, alpha=0.7,
                                            color='blue', edgecolor='navy', 
                                            linewidth=2, label='PROLONGADAS', 
                                            density=True)

# Linha do limiar
ax2.axvline(x=limiar_razao, color='black', linestyle='--', linewidth=3.5,
           label=f'Limiar (mediana) = {limiar_razao:.2f}', alpha=0.8)

# Adicionar curvas de densidade
from scipy.stats import gaussian_kde
if len(torrenciais) > 1:
    kde_torr = gaussian_kde(torrenciais['RAZAO_EROSIVA'])
    x_kde = np.linspace(df_valid['RAZAO_EROSIVA'].min(), 
                        df_valid['RAZAO_EROSIVA'].max(), 200)
    ax2.plot(x_kde, kde_torr(x_kde), 'r-', linewidth=3, 
            label='Densidade TORRENCIAIS', alpha=0.8)

if len(prolongadas) > 1:
    kde_prol = gaussian_kde(prolongadas['RAZAO_EROSIVA'])
    ax2.plot(x_kde, kde_prol(x_kde), 'b-', linewidth=3, 
            label='Densidade PROLONGADAS', alpha=0.8)

ax2.set_xlabel('Razão Erosiva (EI30 / Precipitação)', fontsize=15, fontweight='bold')
ax2.set_ylabel('Densidade de Probabilidade', fontsize=15, fontweight='bold')
ax2.set_title('Distribuição da Razão Erosiva\n' + 
             'Indicador de Intensidade da Chuva',
             fontsize=14, fontweight='bold', pad=15)
ax2.legend(loc='upper right', fontsize=11, framealpha=0.95)
ax2.grid(True, alpha=0.3, axis='y')
ax2.tick_params(labelsize=12)

# Adicionar estatísticas no gráfico
stats_text = f"TORRENCIAIS:\nMédia = {torrenciais['RAZAO_EROSIVA'].mean():.2f}\nDP = {torrenciais['RAZAO_EROSIVA'].std():.2f}\n\nPROLONGADAS:\nMédia = {prolongadas['RAZAO_EROSIVA'].mean():.2f}\nDP = {prolongadas['RAZAO_EROSIVA'].std():.2f}"
ax2.text(0.97, 0.97, stats_text, transform=ax2.transAxes,
        fontsize=11, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

# -------------------------
# Título geral
# -------------------------
fig.suptitle('Diagnóstico de Intensidade das Chuvas Erosivas\n' + 
            'Método: Razão Erosiva (EI30/Precipitação) como Proxy de Intensidade',
            fontsize=18, fontweight='bold', y=0.98)

plt.tight_layout()

# Salvar
output_path = FIGURAS_DIR / "21b_diagnostico_intensidade_chuvas.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n✅ Figura 21B salva: {output_path}")

# =============================================================================
# ESTATÍSTICAS
# =============================================================================
print("\n" + "=" * 80)
print("ESTATÍSTICAS DO DIAGNÓSTICO")
print("=" * 80)

corr_ei30_precip, p_ei30_precip = stats.pearsonr(df_valid['RAINFALL'], df_valid['EI30'])
print(f"\n📈 CORRELAÇÃO EI30 × Precipitação:")
print(f"  r = {corr_ei30_precip:.4f} (p = {p_ei30_precip:.6f})")
print(f"  R² = {r2:.4f} ({r2*100:.2f}% da variância explicada)")

print(f"\n🎯 TESTE T ENTRE GRUPOS:")
t_stat, p_value = stats.ttest_ind(torrenciais['RAZAO_EROSIVA'], 
                                   prolongadas['RAZAO_EROSIVA'])
print(f"  t = {t_stat:.4f}, p = {p_value:.6f}")
if p_value < 0.05:
    print(f"  ✅ Diferença SIGNIFICATIVA entre grupos (p < 0.05)")
else:
    print(f"  ❌ Diferença NÃO significativa (p ≥ 0.05)")

print("\n" + "=" * 80)
print("✅ FIGURA 21B CONCLUÍDA!")
print("=" * 80)
