# Relatório Paralelo: Sedimentação e Evidências Complementares

Este relatório reúne gráficos e métricas complementares para apoiar a reorganização do manuscrito sobre paliçadas, sem alterar o texto do artigo.

## 1. Base de dados utilizada

As figuras e métricas foram derivadas de:

- `CLIMATOLOGIA_20ANOS/sediments/BD.xlsx` na aba `GRAFICO` (séries mensais por segmento)
- `CLIMATOLOGIA_20ANOS/dados/dados_integrados_sedimentacao.csv` (séries mensais por segmento com EI30 e DATA)

Observação metodológica importante: no CSV integrado, a coluna `RAINFALL` está preenchida apenas para o segmento SUP, porém a precipitação é uma covariável externa comum aos três segmentos. Para análises comparativas entre segmentos, a precipitação foi associada por `DATA` a todos os segmentos.

## 2. Série temporal integrada

A figura abaixo apresenta a sedimentação fracionada por segmento e, em painel separado, as séries mensais de precipitação e EI30.

![](RELATORIO_PARALELO_SEDIMENTOS_media/fig01_series_temporais_integradas.png)

## 3. Sazonalidade e janela de maior deposição

A distribuição mensal do incremento fracionado (cm) evidencia a variabilidade intra-anual entre os segmentos.

![](RELATORIO_PARALELO_SEDIMENTOS_media/fig02_sazonalidade_boxplot_fracionado.png)

Contribuição por trimestre no total do incremento fracionado positivo (isto é, `FRACIONADO` truncado em zero para focar deposição) no período 2023-06 a 2025-05:

| Trimestre | Contribuição (%) |
|----------:|-----------------:|
| Q1        | 41,99            |
| Q2        | 29,97            |
| Q3        | 24,85            |
| Q4        | 3,19             |

## 4. Contribuição relativa por segmento

Contribuição relativa no total do incremento fracionado positivo no período:

| Segmento | Contribuição (%) |
|:--------:|-----------------:|
| SUP      | 35,00            |
| INF      | 33,66            |
| MED      | 31,34            |

![](RELATORIO_PARALELO_SEDIMENTOS_media/fig03_contribuicao_segmentos.png)

Ocorre um mês com `FRACIONADO` negativo no segmento SUP, o que sugere um registro com balanço erosivo/remoção local no período, enquanto MED e INF não apresentaram valores negativos no mesmo conjunto de dados.

## 5. Extremos de deposição

Distribuição do incremento fracionado e limiares globais P90 e P95 para o incremento fracionado positivo:

![](RELATORIO_PARALELO_SEDIMENTOS_media/fig04_distribuicao_extremos_fracionado.png)

Resumo dos extremos (`FRACIONADO` positivo):

- P90 = 0,0578 cm
- P95 = 0,0853 cm
- Eventos ≥ P95 = 4 (no conjunto total)
- Contribuição dos eventos ≥ P95 = 40,56% do total depositado

Contribuição relativa dos eventos ≥ P95 dentro de cada segmento:

| Segmento | Contribuição dos ≥ P95 (%) |
|:--------:|---------------------------:|
| INF      | 44,55                      |
| SUP      | 39,55                      |
| MED      | 37,40                      |

## 6. Por que o R² global é baixo e qual segmento se ajusta melhor

Quando se ajusta um modelo linear simples entre deposição mensal (`FRACIONADO` positivo) e precipitação ou EI30 no conjunto completo, a heterogeneidade entre segmentos e a natureza episódica da deposição reduzem o R² global.

Para apoiar a escolha do melhor segmento para uma leitura regressiva, a figura abaixo apresenta o ajuste linear por segmento, com estatísticas r, R² e p em cada painel.

![](RELATORIO_PARALELO_SEDIMENTOS_media/fig05_regressao_por_segmento.png)

Síntese do ajuste linear por segmento:

- MED apresenta os maiores R², com R² ≈ 0,30 para chuva e R² ≈ 0,32 para EI30, ambos com p < 0,01.
- INF apresenta ajuste intermediário, com R² ≈ 0,14 para chuva.
- SUP apresenta ajuste fraco, com R² ≈ 0,07 para chuva.

Nota: as métricas completas foram salvas em `RELATORIO_PARALELO_SEDIMENTOS_media/metricas_relatorio_paralelo.json`.
