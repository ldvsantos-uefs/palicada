import pandas as pd

df = pd.read_csv('dados/dados_integrados_sedimentacao.csv')
df['DATA'] = pd.to_datetime(df['DATA'])

periodo = df[(df['DATA'] >= '2024-01') & (df['DATA'] <= '2024-03')]

print('DADOS 2024-01 a 2024-03:')
print(periodo[['DATA', 'AREA', 'RAINFALL', 'SEDIMENT', 'FRACIONADO']].sort_values('DATA'))

print(f'\n\nLimiares P95:')
print(f'  Acumulada: {df["SEDIMENT"].quantile(0.95):.4f} cm')
print(f'  Incremental: {df["FRACIONADO"].quantile(0.95):.4f} cm')
print(f'  Precipitação: {df["RAINFALL"].quantile(0.95):.2f} mm')

print(f'\n\nMáximos no período:')
print(f'  Acumulada: {periodo["SEDIMENT"].max():.4f} cm')
print(f'  Incremental: {periodo["FRACIONADO"].max():.4f} cm')
print(f'  Precipitação: {periodo["RAINFALL"].max():.2f} mm')
