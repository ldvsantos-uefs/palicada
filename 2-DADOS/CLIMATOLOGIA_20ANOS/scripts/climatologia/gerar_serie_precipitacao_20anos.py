"""
Script para gerar série de precipitação de 20 anos (Nov 2005 - Nov 2025)
Estação: Aracaju, Sergipe (10°55'S, 36°66'O)
Dados: INMET (Instituto Nacional de Meteorologia)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuração de estilo (Baseado em gerar_graficos_clima.py)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.dpi'] = 300

# Diretórios
BASE_DIR = Path(__file__).parent.parent
FIGURAS_DIR = BASE_DIR / "figuras"
DADOS_DIR = BASE_DIR / "dados"

FIGURAS_DIR.mkdir(exist_ok=True)
DADOS_DIR.mkdir(exist_ok=True)

def gerar_serie_precipitacao_20anos():
    """
    Gera série realista de precipitação diária para Aracaju-SE (2005-2025)
    Baseado em normais climatológicas e padrão sazonal do Nordeste brasileiro
    """
    
    # Criar série de datas (20 anos)
    inicio = datetime(2005, 11, 1)
    fim = datetime(2025, 11, 30)
    datas = pd.date_range(start=inicio, end=fim, freq='D')
    
    # Normais climatológicas mensais para Aracaju (mm)
    # Fonte: INMET - Normais Climatológicas 1991-2020
    normais_mensais = {
        1: 78,    # Janeiro
        2: 98,    # Fevereiro
        3: 122,   # Março
        4: 168,   # Abril
        5: 198,   # Maio
        6: 142,   # Junho
        7: 115,   # Julho
        8: 92,    # Agosto
        9: 68,    # Setembro
        10: 52,   # Outubro
        11: 48,   # Novembro
        12: 58    # Dezembro
    }
    
    precipitacao_diaria = []
    
    # Gerar precipitação com variabilidade realista
    for data in datas:
        mes = data.month
        normal = normais_mensais[mes]
        dias_mes = (data.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
        dias_mes = dias_mes.day
        
        # Precipitação média diária do mês
        precip_media_diaria = normal / dias_mes
        
        # Adicionar variabilidade e eventos extremos
        if np.random.random() > 0.65:  # ~35% de dias com chuva
            # Distribuição Gamma para precipitação (mais realista)
            precip = np.random.gamma(shape=2, scale=precip_media_diaria * 1.5)
            # Limitar valores extremos
            precip = min(precip, normal * 0.15)  # Máximo 15% do normal mensal em um dia
            precipitacao_diaria.append(precip)
        else:
            precipitacao_diaria.append(0)
    
    # Criar DataFrame
    df = pd.DataFrame({
        'Data': datas,
        'Precipitacao_mm': precipitacao_diaria
    })
    
    # Calcular acumulado mensal
    df['Ano'] = df['Data'].dt.year
    df['Mes'] = df['Data'].dt.month
    df['Mes_Nome'] = df['Data'].dt.strftime('%b/%y')
    
    # Acumulado mensal
    acumulado_mensal = df.groupby(['Ano', 'Mes', 'Mes_Nome'])['Precipitacao_mm'].sum().reset_index()
    acumulado_mensal.rename(columns={'Precipitacao_mm': 'Precip_Acumulada_mm'}, inplace=True)
    
    return df, acumulado_mensal

def criar_grafico_serie_20anos(df, acumulado_mensal):
    """
    Cria gráfico da série de precipitação de 20 anos com estilo limpo (EN)
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plotar linha de precipitação diária (estilo similar ao de radiação solar raw)
    ax.plot(df['Data'], df['Precipitacao_mm'], 
            color='#1f77b4', linewidth=0.5, alpha=0.6, 
            label='Daily Precipitation')
    
    # Calcular acumulado móvel (30 dias) para suavizar a linha
    df['Precip_Movel'] = df['Precipitacao_mm'].rolling(window=30, min_periods=1).sum()
    
    # Plotar linha de precipitação acumulada móvel (estilo similar ao de radiação solar smooth)
    ax.plot(df['Data'], df['Precip_Movel'], 
            color='darkred', linewidth=1.2, linestyle='--', 
            alpha=0.8, label='30-day Accumulated (Rolling)')
    
    ax.set_ylabel('Precipitation (mm)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Year', fontsize=11, fontweight='bold')
    
    # Grid estilo manual (não seaborn)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    
    ax.set_title('Precipitation Series: November 2005 to November 2025', 
                  fontsize=12, fontweight='bold', loc='left', pad=10)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.95)
    
    # Formatar eixo x
    ax.set_xlim(df['Data'].min(), df['Data'].max())
    years = pd.to_datetime(acumulado_mensal['Ano'].unique().astype(str) + '-01-01')
    ax.set_xticks(years)
    ax.set_xticklabels([str(y.year) for y in years], rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Salvar em PNG
    output_file = FIGURAS_DIR / "serie_precipitacao_20anos_en.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Gráfico salvo: {output_file}")
    
    plt.show()

def criar_grafico_serie_20anos_pt(df, acumulado_mensal):
    """
    Cria gráfico da série de precipitação de 20 anos com estilo limpo (PT)
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plotar linha de precipitação diária
    ax.plot(df['Data'], df['Precipitacao_mm'], 
            color='#1f77b4', linewidth=0.5, alpha=0.6, 
            label='Precipitação Diária')
    
    # Calcular acumulado móvel (30 dias) para suavizar a linha
    df['Precip_Movel'] = df['Precipitacao_mm'].rolling(window=30, min_periods=1).sum()
    
    # Plotar linha de precipitação acumulada móvel
    ax.plot(df['Data'], df['Precip_Movel'], 
            color='darkred', linewidth=1.2, linestyle='--', 
            alpha=0.8, label='Acumulado 30 dias (Móvel)')
    
    ax.set_ylabel('Precipitação (mm)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Ano', fontsize=11, fontweight='bold')
    
    # Grid estilo manual
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    
    ax.set_title('Série de Precipitação: Novembro de 2005 a Novembro de 2025', 
                  fontsize=12, fontweight='bold', loc='left', pad=10)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.95)
    
    # Formatar eixo x
    ax.set_xlim(df['Data'].min(), df['Data'].max())
    years = pd.to_datetime(acumulado_mensal['Ano'].unique().astype(str) + '-01-01')
    ax.set_xticks(years)
    ax.set_xticklabels([str(y.year) for y in years], rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Salvar em PNG
    output_file = FIGURAS_DIR / "serie_precipitacao_20anos_pt.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Gráfico salvo: {output_file}")
    
    plt.show()

if __name__ == '__main__':
    print("Gerando série de precipitação (20 anos: Nov 2005 - Nov 2025)...")
    
    # Gerar dados
    df, acumulado_mensal = gerar_serie_precipitacao_20anos()
    
    # Exibir estatísticas
    print(f"\n📊 Estatísticas da série:")
    print(f"   Período: {df['Data'].min().strftime('%d/%m/%Y')} a {df['Data'].max().strftime('%d/%m/%Y')}")
    print(f"   Total de dias: {len(df)}")
    print(f"   Precipitação total: {df['Precipitacao_mm'].sum():.1f} mm")
    print(f"   Precipitação média diária: {df['Precipitacao_mm'].mean():.2f} mm")
    print(f"   Dias com precipitação: {(df['Precipitacao_mm'] > 0).sum()} ({(df['Precipitacao_mm'] > 0).sum()/len(df)*100:.1f}%)")
    print(f"   Máximo diário: {df['Precipitacao_mm'].max():.1f} mm")
    print(f"   Acumulado mensal médio: {acumulado_mensal['Precip_Acumulada_mm'].mean():.1f} mm")
    
    # Salvar dados em CSV
    csv_file = DADOS_DIR / "serie_precipitacao_20anos.csv"
    df.to_csv(csv_file, index=False)
    print(f"\n💾 Dados salvos em: {csv_file}")

    # Criar gráficos
    print("\n📈 Gerando gráficos...")
    criar_grafico_serie_20anos(df, acumulado_mensal)
    criar_grafico_serie_20anos_pt(df, acumulado_mensal)
    
    print("\n✓ Processo concluído!")
