"""
Script para Exploração dos Dados de Sedimentação
Análise preliminar dos arquivos Excel da pasta sediments
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Diretórios
BASE_DIR = Path(__file__).parent.parent
SEDIMENTOS_DIR = BASE_DIR / "sediments"

print("=" * 80)
print("EXPLORAÇÃO DOS DADOS DE SEDIMENTAÇÃO")
print("=" * 80)

# Listar arquivos na pasta
print("\nArquivos disponíveis:")
for arquivo in SEDIMENTOS_DIR.glob("*.xlsx"):
    print(f"  - {arquivo.name}")

# Explorar primeiro arquivo: BD.xlsx
print("\n" + "=" * 80)
print("ARQUIVO 1: BD.xlsx")
print("=" * 80)

try:
    # Ler todas as planilhas do arquivo
    excel_file = pd.ExcelFile(SEDIMENTOS_DIR / "BD.xlsx")
    print(f"\nPlanilhas encontradas: {excel_file.sheet_names}")
    
    for sheet_name in excel_file.sheet_names:
        print(f"\n{'='*60}")
        print(f"Planilha: {sheet_name}")
        print(f"{'='*60}")
        
        df = pd.read_excel(SEDIMENTOS_DIR / "BD.xlsx", sheet_name=sheet_name)
        
        print(f"\nDimensões: {df.shape[0]} linhas x {df.shape[1]} colunas")
        print(f"\nColunas:")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i}. {col}")
        
        print(f"\nPrimeiras 5 linhas:")
        print(df.head())
        
        print(f"\nInformações do DataFrame:")
        print(df.info())
        
        print(f"\nEstatísticas descritivas:")
        print(df.describe())
        
except Exception as e:
    print(f"Erro ao ler BD.xlsx: {e}")

# Explorar segundo arquivo: Dados de Sedimentação - TCC e EE.xlsx
print("\n" + "=" * 80)
print("ARQUIVO 2: Dados de Sedimentação - TCC e EE.xlsx")
print("=" * 80)

try:
    excel_file = pd.ExcelFile(SEDIMENTOS_DIR / "Dados de Sedimentação - TCC e EE.xlsx")
    print(f"\nPlanilhas encontradas: {excel_file.sheet_names}")
    
    for sheet_name in excel_file.sheet_names:
        print(f"\n{'='*60}")
        print(f"Planilha: {sheet_name}")
        print(f"{'='*60}")
        
        df = pd.read_excel(SEDIMENTOS_DIR / "Dados de Sedimentação - TCC e EE.xlsx", sheet_name=sheet_name)
        
        print(f"\nDimensões: {df.shape[0]} linhas x {df.shape[1]} colunas")
        print(f"\nColunas:")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i}. {col}")
        
        print(f"\nPrimeiras 5 linhas:")
        print(df.head())
        
        print(f"\nInformações do DataFrame:")
        print(df.info())
        
        print(f"\nEstatísticas descritivas:")
        print(df.describe())
        
except Exception as e:
    print(f"Erro ao ler Dados de Sedimentação - TCC e EE.xlsx: {e}")

print("\n" + "=" * 80)
print("EXPLORAÇÃO CONCLUÍDA")
print("=" * 80)
