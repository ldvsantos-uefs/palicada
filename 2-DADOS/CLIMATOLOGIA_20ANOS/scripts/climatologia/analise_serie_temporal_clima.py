"""
Script de Análise de Série Temporal Climatológica (2005-2025)
Realiza análises estatísticas avançadas na série de precipitação gerada.
Inclui: Decomposição Sazonal, Análise de Anomalias e Tendências.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Configuração de estilo
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['figure.dpi'] = 300

# Diretórios
BASE_DIR = Path(__file__).parent.parent
FIGURAS_DIR = BASE_DIR / "figuras"
DADOS_DIR = BASE_DIR / "dados"

FIGURAS_DIR.mkdir(exist_ok=True)
DADOS_DIR.mkdir(exist_ok=True)

def gerar_serie_precipitacao_20anos(seed=42):
    """
    Gera a mesma série de precipitação do script anterior, mas com semente fixa
    para garantir reprodutibilidade nas análises estatísticas.
    """
    np.random.seed(seed)
    
    inicio = datetime(2005, 11, 1)
    fim = datetime(2025, 11, 30)
    datas = pd.date_range(start=inicio, end=fim, freq='D')
    
    normais_mensais = {
        1: 78, 2: 98, 3: 122, 4: 168, 5: 198, 6: 142,
        7: 115, 8: 92, 9: 68, 10: 52, 11: 48, 12: 58
    }
    
    precipitacao_diaria = []
    
    for data in datas:
        mes = data.month
        normal = normais_mensais[mes]
        dias_mes = (data.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
        dias_mes = dias_mes.day
        precip_media_diaria = normal / dias_mes
        
        if np.random.random() > 0.65:
            precip = np.random.gamma(shape=2, scale=precip_media_diaria * 1.5)
            precip = min(precip, normal * 0.15)
            precipitacao_diaria.append(precip)
        else:
            precipitacao_diaria.append(0)
    
    df = pd.DataFrame({'Data': datas, 'Precipitacao_mm': precipitacao_diaria})
    df.set_index('Data', inplace=True)
    return df

def analisar_decomposicao_sazonal(df):
    """
    Realiza decomposição sazonal da série mensal (Tendência + Sazonalidade + Resíduo)
    """
    # Reamostrar para médias mensais para análise de decomposição
    df_mensal = df['Precipitacao_mm'].resample('M').sum()
    
    # Decomposição aditiva (Série = Tendência + Sazonalidade + Resíduo)
    decomposicao = seasonal_decompose(df_mensal, model='additive', period=12)
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    
    decomposicao.observed.plot(ax=ax1, color='#1f77b4', linewidth=1)
    ax1.set_ylabel('Observado')
    ax1.set_title('Decomposição da Série Temporal de Precipitação (Mensal)', fontweight='bold')
    
    decomposicao.trend.plot(ax=ax2, color='#d62728', linewidth=2)
    ax2.set_ylabel('Tendência')
    
    decomposicao.seasonal.plot(ax=ax3, color='#2ca02c', linewidth=1)
    ax3.set_ylabel('Sazonalidade')
    
    decomposicao.resid.plot(ax=ax4, color='black', linewidth=0.5, marker='o', markersize=2, linestyle='None')
    ax4.set_ylabel('Resíduos')
    ax4.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    
    plt.xlabel('Ano')
    plt.tight_layout()
    
    output_file = FIGURAS_DIR / "analise_decomposicao_temporal.png"
    plt.savefig(output_file, bbox_inches='tight', facecolor='white')
    print(f"✓ Gráfico de decomposição salvo: {output_file}")
    plt.close()

def analisar_anomalias(df):
    """
    Calcula e plota anomalias de precipitação (Desvio da média climatológica)
    """
    # Calcular média climatológica para cada mês (Climatologia)
    df['Mes'] = df.index.month
    df['Ano'] = df.index.year
    
    # Acumulado mensal real
    df_mensal = df.groupby(['Ano', 'Mes'])['Precipitacao_mm'].sum().reset_index()
    
    # Criar coluna de data para plotagem (Dia 1 de cada mês)
    df_mensal['Data'] = pd.to_datetime(dict(year=df_mensal['Ano'], month=df_mensal['Mes'], day=1))
    
    # Média de longo prazo para cada mês (1 a 12)
    climatologia = df_mensal.groupby('Mes')['Precipitacao_mm'].mean()
    
    # Calcular anomalia
    df_mensal['Climatologia'] = df_mensal['Mes'].map(climatologia)
    df_mensal['Anomalia'] = df_mensal['Precipitacao_mm'] - df_mensal['Climatologia']
    
    # Plotar
    fig, ax = plt.subplots(figsize=(14, 6))
    
    cores = ['#d62728' if x < 0 else '#1f77b4' for x in df_mensal['Anomalia']]
    
    ax.bar(df_mensal['Data'], df_mensal['Anomalia'], color=cores, width=25, alpha=0.8)
    
    # Linha de média móvel da anomalia (tendência de seca/chuva)
    anomalia_smooth = df_mensal.set_index('Data')['Anomalia'].rolling(window=12).mean()
    ax.plot(anomalia_smooth.index, anomalia_smooth, color='black', linewidth=2, linestyle='-', label='Média Móvel (12 meses)')
    
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_title('Anomalias de Precipitação Mensal (Desvio da Média Climatológica)', fontweight='bold')
    ax.set_ylabel('Anomalia de Precipitação (mm)')
    ax.set_xlabel('Ano')
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.legend()
    
    output_file = FIGURAS_DIR / "analise_anomalias_precipitacao.png"
    plt.savefig(output_file, bbox_inches='tight', facecolor='white')
    print(f"✓ Gráfico de anomalias salvo: {output_file}")
    plt.close()

def analisar_heatmap_sazonal(df):
    """
    Cria um heatmap (Ano x Mês) para visualizar padrões de seca e chuva
    """
    df_mensal = df.groupby([df.index.year, df.index.month])['Precipitacao_mm'].sum().unstack()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(df_mensal, cmap='YlGnBu', annot=False, fmt='.0f', cbar_kws={'label': 'Precipitação (mm)'})
    
    plt.title('Matriz de Precipitação Mensal (2005-2025)', fontweight='bold')
    plt.ylabel('Ano')
    plt.xlabel('Mês')
    
    output_file = FIGURAS_DIR / "analise_heatmap_precipitacao.png"
    plt.savefig(output_file, bbox_inches='tight', facecolor='white')
    print(f"✓ Heatmap salvo: {output_file}")
    plt.close()

if __name__ == '__main__':
    print("Iniciando Análise Estatística da Série Temporal...")
    
    # 1. Gerar Dados
    df = gerar_serie_precipitacao_20anos()
    print(f"Dados gerados: {len(df)} dias.")
    
    # 2. Decomposição Sazonal
    analisar_decomposicao_sazonal(df)
    
    # 3. Análise de Anomalias
    analisar_anomalias(df)
    
    # 4. Heatmap Sazonal
    analisar_heatmap_sazonal(df)
    
    print("\nAnálises concluídas com sucesso!")
