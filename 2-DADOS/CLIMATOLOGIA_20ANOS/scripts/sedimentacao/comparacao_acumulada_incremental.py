"""
ANÁLISE COMPARATIVA: SEDIMENTAÇÃO ACUMULADA vs INCREMENTAL
Demonstra por que sedimentação incremental tem melhor correlação temporal com chuva
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
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
print("ANÁLISE COMPARATIVA: ACUMULADA vs INCREMENTAL")
print("=" * 80)

# Carregar dados
df = pd.read_csv(DADOS_DIR / "dados_integrados_sedimentacao.csv")
df['DATA'] = pd.to_datetime(df['DATA'])

# Focar na área SUP (dados completos)
df_sup = df[df['AREA'] == 'SUP'].sort_values('DATA').copy()

print(f"\n✓ Analisando área SUP: {len(df_sup)} observações")

# =============================================================================
# ANÁLISE DE CORRELAÇÃO: ACUMULADA vs INCREMENTAL
# =============================================================================
print("\n" + "-" * 80)
print("COMPARAÇÃO DE CORRELAÇÕES")
print("-" * 80)

# Remover NaN
df_clean = df_sup[['RAINFALL', 'EI30', 'SEDIMENT', 'FRACIONADO']].dropna()

# Correlações com ACUMULADA
corr_precip_acum, p_precip_acum = stats.pearsonr(df_clean['RAINFALL'], df_clean['SEDIMENT'])
corr_ei30_acum, p_ei30_acum = stats.pearsonr(df_clean['EI30'], df_clean['SEDIMENT'])

# Correlações com INCREMENTAL
corr_precip_incr, p_precip_incr = stats.pearsonr(df_clean['RAINFALL'], df_clean['FRACIONADO'])
corr_ei30_incr, p_ei30_incr = stats.pearsonr(df_clean['EI30'], df_clean['FRACIONADO'])

print("\nCORRELAÇÕES COM SEDIMENTAÇÃO ACUMULADA:")
print(f"  Precipitação × Acumulada: r = {corr_precip_acum:.4f} (p = {p_precip_acum:.4f})")
print(f"  EI30 × Acumulada:         r = {corr_ei30_acum:.4f} (p = {p_ei30_acum:.4f})")

print("\nCORRELAÇÕES COM SEDIMENTAÇÃO INCREMENTAL (mensal):")
print(f"  Precipitação × Incremental: r = {corr_precip_incr:.4f} (p = {p_precip_incr:.4f})")
print(f"  EI30 × Incremental:         r = {corr_ei30_incr:.4f} (p = {p_ei30_incr:.4f})")

print(f"\n{'='*80}")
print("INTERPRETAÇÃO:")
if abs(corr_precip_incr) > abs(corr_precip_acum):
    print(f"✓ INCREMENTAL tem correlação {abs(corr_precip_incr)/abs(corr_precip_acum):.2f}x MELHOR")
    print("  → Reflete resposta imediata à precipitação do mês")
else:
    print("✓ ACUMULADA tem correlação maior (efeito de integração temporal)")

# Modelos de regressão
X_precip = df_clean['RAINFALL'].values.reshape(-1, 1)
X_ei30 = df_clean['EI30'].values.reshape(-1, 1)

# Modelo para acumulada
reg_acum_p = LinearRegression().fit(X_precip, df_clean['SEDIMENT'])
r2_acum_p = reg_acum_p.score(X_precip, df_clean['SEDIMENT'])

reg_acum_e = LinearRegression().fit(X_ei30, df_clean['SEDIMENT'])
r2_acum_e = reg_acum_e.score(X_ei30, df_clean['SEDIMENT'])

# Modelo para incremental
reg_incr_p = LinearRegression().fit(X_precip, df_clean['FRACIONADO'])
r2_incr_p = reg_incr_p.score(X_precip, df_clean['FRACIONADO'])

reg_incr_e = LinearRegression().fit(X_ei30, df_clean['FRACIONADO'])
r2_incr_e = reg_incr_e.score(X_ei30, df_clean['FRACIONADO'])

print(f"\nR² DOS MODELOS PREDITIVOS:")
print(f"  Precipitação → Acumulada:    R² = {r2_acum_p:.4f} ({r2_acum_p*100:.2f}% explicado)")
print(f"  Precipitação → Incremental:  R² = {r2_incr_p:.4f} ({r2_incr_p*100:.2f}% explicado)")
print(f"  EI30 → Acumulada:            R² = {r2_acum_e:.4f} ({r2_acum_e*100:.2f}% explicado)")
print(f"  EI30 → Incremental:          R² = {r2_incr_e:.4f} ({r2_incr_e*100:.2f}% explicado)")

# =============================================================================
# FIGURA COMPARATIVA
# =============================================================================
print("\n" + "-" * 80)
print("GERANDO FIGURA COMPARATIVA")
print("-" * 80)

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

# ============== PAINEL SUPERIOR: SÉRIES TEMPORAIS ==============
ax1 = fig.add_subplot(gs[0, :])
ax1_twin = ax1.twinx()

# Precipitação
ax1.bar(df_sup['DATA'], df_sup['RAINFALL'], width=20, alpha=0.3, 
        color='steelblue', label='Precipitação Mensal', edgecolor='navy')

# Sedimentação acumulada
ax1_twin.plot(df_sup['DATA'], df_sup['SEDIMENT'], '-o', color='red', 
             linewidth=2.5, markersize=7, label='Sedimentação Acumulada',
             markeredgecolor='darkred', markeredgewidth=0.5)

# Sedimentação incremental (escala secundária)
ax1_twin2 = ax1.twinx()
ax1_twin2.spines['right'].set_position(('outward', 60))
ax1_twin2.plot(df_sup['DATA'], df_sup['FRACIONADO'], '-s', color='green', 
              linewidth=2.5, markersize=7, label='Sedimentação Incremental',
              markeredgecolor='darkgreen', markeredgewidth=0.5, alpha=0.8)

ax1.set_xlabel('Data', fontweight='bold', fontsize=12)
ax1.set_ylabel('Precipitação (mm)', fontweight='bold', fontsize=12, color='steelblue')
ax1_twin.set_ylabel('Sedimentação Acumulada (cm)', fontweight='bold', fontsize=12, color='red')
ax1_twin2.set_ylabel('Sedimentação Incremental (cm/mês)', fontweight='bold', fontsize=12, color='green')

ax1.tick_params(axis='y', labelcolor='steelblue')
ax1_twin.tick_params(axis='y', labelcolor='red')
ax1_twin2.tick_params(axis='y', labelcolor='green')

ax1.set_title('Comparação Temporal: Acumulada (tendência) vs Incremental (eventos)',
             fontweight='bold', fontsize=13, pad=10)

# Legendas combinadas
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_twin.get_legend_handles_labels()
lines3, labels3 = ax1_twin2.get_legend_handles_labels()
ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3,
          loc='upper left', fontsize=10, framealpha=0.95)

ax1.grid(True, alpha=0.3)

# ============== PAINEL MÉDIO: CORRELAÇÕES COM ACUMULADA ==============
# Precipitação × Acumulada
ax2 = fig.add_subplot(gs[1, 0])
ax2.scatter(df_clean['RAINFALL'], df_clean['SEDIMENT'], s=80, alpha=0.6,
           color='orangered', edgecolors='darkred', linewidth=0.5)

# Linha de tendência
x_pred = np.linspace(df_clean['RAINFALL'].min(), df_clean['RAINFALL'].max(), 100)
y_pred = reg_acum_p.predict(x_pred.reshape(-1, 1))
ax2.plot(x_pred, y_pred, 'r--', linewidth=2.5, alpha=0.8,
        label=f'R² = {r2_acum_p:.3f}\nr = {corr_precip_acum:.3f}')

ax2.set_xlabel('Precipitação Mensal (mm)', fontweight='bold')
ax2.set_ylabel('Sedimentação Acumulada (cm)', fontweight='bold')
ax2.set_title('Precipitação × Acumulada (Tendência de Longo Prazo)', fontweight='bold')
ax2.legend(loc='upper left', fontsize=10)
ax2.grid(True, alpha=0.3)

# EI30 × Acumulada
ax3 = fig.add_subplot(gs[1, 1])
ax3.scatter(df_clean['EI30'], df_clean['SEDIMENT'], s=80, alpha=0.6,
           color='coral', edgecolors='darkred', linewidth=0.5)

x_pred_ei30 = np.linspace(df_clean['EI30'].min(), df_clean['EI30'].max(), 100)
y_pred_ei30 = reg_acum_e.predict(x_pred_ei30.reshape(-1, 1))
ax3.plot(x_pred_ei30, y_pred_ei30, 'r--', linewidth=2.5, alpha=0.8,
        label=f'R² = {r2_acum_e:.3f}\nr = {corr_ei30_acum:.3f}')

ax3.set_xlabel('EI30 (MJ·mm/ha·h)', fontweight='bold')
ax3.set_ylabel('Sedimentação Acumulada (cm)', fontweight='bold')
ax3.set_title('EI30 × Acumulada (Tendência de Longo Prazo)', fontweight='bold')
ax3.legend(loc='upper left', fontsize=10)
ax3.grid(True, alpha=0.3)

# ============== PAINEL INFERIOR: CORRELAÇÕES COM INCREMENTAL ==============
# Precipitação × Incremental
ax4 = fig.add_subplot(gs[2, 0])
ax4.scatter(df_clean['RAINFALL'], df_clean['FRACIONADO'], s=80, alpha=0.6,
           color='limegreen', edgecolors='darkgreen', linewidth=0.5)

x_pred = np.linspace(df_clean['RAINFALL'].min(), df_clean['RAINFALL'].max(), 100)
y_pred = reg_incr_p.predict(x_pred.reshape(-1, 1))
ax4.plot(x_pred, y_pred, 'g--', linewidth=2.5, alpha=0.8,
        label=f'R² = {r2_incr_p:.3f}\nr = {corr_precip_incr:.3f}')

ax4.set_xlabel('Precipitação Mensal (mm)', fontweight='bold')
ax4.set_ylabel('Sedimentação Incremental (cm/mês)', fontweight='bold')
ax4.set_title('Precipitação × Incremental (Resposta Imediata)', fontweight='bold')
ax4.legend(loc='upper left', fontsize=10)
ax4.grid(True, alpha=0.3)

# EI30 × Incremental
ax5 = fig.add_subplot(gs[2, 1])
ax5.scatter(df_clean['EI30'], df_clean['FRACIONADO'], s=80, alpha=0.6,
           color='yellowgreen', edgecolors='darkgreen', linewidth=0.5)

x_pred_ei30 = np.linspace(df_clean['EI30'].min(), df_clean['EI30'].max(), 100)
y_pred_ei30 = reg_incr_e.predict(x_pred_ei30.reshape(-1, 1))
ax5.plot(x_pred_ei30, y_pred_ei30, 'g--', linewidth=2.5, alpha=0.8,
        label=f'R² = {r2_incr_e:.3f}\nr = {corr_ei30_incr:.3f}')

ax5.set_xlabel('EI30 (MJ·mm/ha·h)', fontweight='bold')
ax5.set_ylabel('Sedimentação Incremental (cm/mês)', fontweight='bold')
ax5.set_title('EI30 × Incremental (Resposta Imediata)', fontweight='bold')
ax5.legend(loc='upper left', fontsize=10)
ax5.grid(True, alpha=0.3)

plt.suptitle('ANÁLISE COMPARATIVA: SEDIMENTAÇÃO ACUMULADA vs INCREMENTAL',
            fontsize=15, fontweight='bold', y=0.995)

plt.savefig(FIGURAS_DIR / "16_comparacao_acumulada_vs_incremental.png",
           dpi=300, bbox_inches='tight')
print("✓ Figura salva: 16_comparacao_acumulada_vs_incremental.png")
plt.close()

# =============================================================================
# INTERPRETAÇÃO CIENTÍFICA
# =============================================================================
print("\n" + "=" * 80)
print("INTERPRETAÇÃO CIENTÍFICA")
print("=" * 80)

interpretacao = f"""
POR QUE USAR SEDIMENTAÇÃO INCREMENTAL?

1. RESPOSTA TEMPORAL DIRETA
   • Acumulada: reflete histórico completo (integral temporal)
   • Incremental: reflete resposta ao evento ESPECÍFICO do mês
   
   → Para correlacionar com chuva mensal, INCREMENTAL é apropriado

2. CORRELAÇÕES OBSERVADAS:
   
   ACUMULADA (r = {corr_precip_acum:.3f}):
   • Correlação {('FORTE' if abs(corr_precip_acum) > 0.7 else 'MODERADA' if abs(corr_precip_acum) > 0.4 else 'FRACA')}
   • R² = {r2_acum_p:.3f} → {r2_acum_p*100:.1f}% da variação explicada
   • Representa TENDÊNCIA de longo prazo
   • Efeito cumulativo de todos os eventos anteriores
   
   INCREMENTAL (r = {corr_precip_incr:.3f}):
   • Correlação {('FORTE' if abs(corr_precip_incr) > 0.7 else 'MODERADA' if abs(corr_precip_incr) > 0.4 else 'FRACA')}
   • R² = {r2_incr_p:.3f} → {r2_incr_p*100:.1f}% da variação explicada
   • Representa RESPOSTA imediata ao evento
   • Permite identificar eventos extremos isolados

3. VANTAGENS DA ANÁLISE INCREMENTAL:

   ✓ Identifica EVENTOS ESPECÍFICOS de alta erosão
   ✓ Não contamina análise com histórico acumulado
   ✓ Permite correlação causa-efeito temporal
   ✓ Adequada para validar índices como EI30
   ✓ Detecta padrões sazonais e picos

4. QUANDO USAR CADA UMA:

   ACUMULADA:
   • Avaliar perda total de solo ao longo do tempo
   • Dimensionar bacias de sedimentação
   • Estimar volume de assoreamento
   • Análise de tendências de longo prazo
   
   INCREMENTAL:
   • Correlacionar com eventos climáticos
   • Validar modelos de erosão (RUSLE, WEPP)
   • Identificar períodos críticos
   • Avaliar eficácia de práticas conservacionistas
   • Estudos processo-resposta

5. IMPLICAÇÃO PARA ESTA PESQUISA:

   {'✓ INCREMENTAL mostra correlação MELHOR' if abs(corr_precip_incr) > abs(corr_precip_acum) else '✓ ACUMULADA mostra correlação melhor (efeito integrado)'}
   
   Isso {'confirma' if abs(corr_precip_incr) > abs(corr_precip_acum) else 'sugere'} que:
   • Sedimentação mensal responde {'diretamente' if abs(corr_precip_incr) > abs(corr_precip_acum) else 'de forma integrada'} à precipitação
   • Eventos isolados são {'detectáveis' if abs(corr_precip_incr) > abs(corr_precip_acum) else 'mascarados pelo acúmulo'}
   • {'Análise incremental é apropriada para este estudo' if abs(corr_precip_incr) > abs(corr_precip_acum) else 'Processos de longo prazo dominam'}

CONCLUSÃO:
Para estudos de EVENTOS e validação de índices de erosividade,
SEDIMENTAÇÃO INCREMENTAL é a variável mais apropriada, pois:
  1. Representa taxa mensal (cm/mês)
  2. Permite correlação temporal direta
  3. Identifica picos e eventos extremos
  4. Adequada para escala mensal dos dados climáticos
"""

print(interpretacao)

# Salvar relatório
with open(DADOS_DIR / "relatorio_acumulada_vs_incremental.txt", 'w', encoding='utf-8') as f:
    f.write(interpretacao)

print("\n✓ Relatório salvo: relatorio_acumulada_vs_incremental.txt")

print("\n" + "=" * 80)
print("ANÁLISE CONCLUÍDA!")
print("=" * 80)
