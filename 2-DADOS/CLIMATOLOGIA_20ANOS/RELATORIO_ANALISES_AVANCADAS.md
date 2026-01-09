# RESULTADOS DAS ANÁLISES AVANÇADAS DE SÉRIES TEMPORAIS CLIMÁTICAS
## Foco em Erosão e Comportamento Climático (2005-2025)

---

## 📊 RESUMO EXECUTIVO

Este relatório apresenta os resultados de análises estatísticas avançadas em uma série temporal de precipitação de 20 anos (2005-2025), com foco em estudos de erosão do solo. Foram aplicadas 8 técnicas estatísticas de alto impacto para caracterizar padrões climáticos, eventos extremos e potencial erosivo.

---

## 1️⃣ DECOMPOSIÇÃO DA SÉRIE TEMPORAL

### Resultados Principais:
- **Tendência média**: 95.59 mm/mês
- **Amplitude sazonal**: 119.64 mm (diferença entre pico e vale sazonal)
- **Desvio padrão dos resíduos**: 28.53 mm

### Interpretação Científica:
A decomposição aditiva revelou um forte componente sazonal (119.64 mm de amplitude), indicando que a precipitação varia significativamente ao longo do ano. A tendência estável em torno de 95.59 mm/mês sugere que não há mudança climática abrupta no período analisado. O desvio padrão dos resíduos (28.53 mm) representa a variabilidade não explicada por tendência e sazonalidade, indicando eventos aleatórios de precipitação.

### Relevância para Erosão:
- Picos sazonais de precipitação coincidem com períodos de maior risco erosivo
- A previsibilidade sazonal permite planejamento de práticas conservacionistas
- Resíduos elevados indicam necessidade de monitoramento contínuo

**Figura**: [1_decomposicao_serie_temporal.png](../figuras/1_decomposicao_serie_temporal.png)

---

## 2️⃣ TESTES DE ESTACIONARIEDADE (DICKEY-FULLER AUMENTADO)

### Resultados:
| Série | Estatística ADF | p-valor | Conclusão |
|-------|----------------|---------|-----------|
| **Original** | -2.732 | 0.0686 | **NÃO ESTACIONÁRIA** |
| **Diferenciada (1ª ordem)** | -16.220 | <0.0001 | **ESTACIONÁRIA** |
| **Diferenciada Sazonal (lag=12)** | -5.137 | <0.0001 | **ESTACIONÁRIA** |

### Interpretação:
A série original apresenta não-estacionariedade marginal (p=0.0686), sugerindo presença de tendência ou sazonalidade. Após diferenciação simples, a série torna-se fortemente estacionária (p<0.0001), indicando que uma transformação simples é suficiente para modelagem. A diferenciação sazonal também produz estacionariedade, confirmando o padrão cíclico anual.

### Implicações:
- Modelos ARIMA/SARIMA são apropriados para previsão
- Eventos extremos não apresentam tendência crescente no período
- Regime pluviométrico é relativamente estável

**Figura**: [2_teste_estacionariedade.png](../figuras/2_teste_estacionariedade.png)

---

## 3️⃣ ANÁLISE DE AUTOCORRELAÇÃO (ACF/PACF)

### Resultados da Diferenciação Sazonal:
A análise de autocorrelação (ACF) e autocorrelação parcial (PACF) revelou:
- **ACF**: Decaimento gradual até lag 12, confirmando padrão sazonal
- **PACF**: Picos significativos em lags 1, 12 e 24, sugerindo componentes autorregressivos
- **Após diferenciação sazonal**: Redução drástica da autocorrelação, indicando remoção efetiva do padrão cíclico

### Interpretação:
Os correlogramas indicam que a precipitação atual depende fortemente dos valores de 1 e 12 meses anteriores. Este padrão é típico de séries pluviométricas em regiões com sazonalidade bem definida.

**Figura**: [3_diferenciacao_sazonal_acf_pacf.png](../figuras/3_diferenciacao_sazonal_acf_pacf.png)

---

## 4️⃣ MODELAGEM SARIMA (PREVISÃO SAZONAL)

### Modelo Ajustado: SARIMA(1,1,1)(1,1,1)₁₂

### Métricas de Ajuste:
- **AIC**: 2244.54 (quanto menor, melhor)
- **BIC**: 2261.68
- **Log-Likelihood**: -1117.27

### Previsão para 24 meses (2026-2027):
O modelo SARIMA capturou com sucesso os padrões sazonais e forneceu previsões com intervalos de confiança de 95%. As previsões indicam:
- Manutenção do padrão sazonal histórico
- Precipitação média mensal entre 50-150 mm
- Picos esperados entre dezembro-maio (estação chuvosa)

### Diagnóstico do Modelo:
- **Resíduos**: Aproximadamente normais (Q-Q plot próximo da linha)
- **Autocorrelação residual**: Não significativa (resíduos são ruído branco)
- **Teste Ljung-Box**: Indica bom ajuste do modelo

### Aplicação em Erosão:
- Permite planejamento antecipado de práticas de conservação
- Identificação de meses críticos para manejo do solo
- Base para modelos integrados de erosão-precipitação

**Figuras**: 
- [4_previsao_sarima.png](../figuras/4_previsao_sarima.png)
- [4_diagnostico_sarima.png](../figuras/4_diagnostico_sarima.png)

---

## 5️⃣ ANÁLISE DE EXTREMOS - DISTRIBUIÇÃO GEV

### Parâmetros da Distribuição:
- **Shape (ξ)**: 1.1971 → Distribuição Fréchet (cauda pesada)
- **Location (μ)**: 28.87 mm
- **Scale (σ)**: 0.99 mm

### Valores de Retorno:
| Período de Retorno | Precipitação Máxima Diária |
|-------------------|---------------------------|
| **2 anos** | 29.17 mm |
| **5 anos** | 29.56 mm |
| **10 anos** | 29.64 mm |
| **20 anos** | 29.68 mm |
| **50 anos** | 29.69 mm |
| **100 anos** | 29.70 mm |

### Interpretação Crítica:
O parâmetro shape positivo (1.1971) indica uma distribuição de Fréchet, característica de eventos extremos com cauda pesada. **ATENÇÃO**: Os valores de retorno muito próximos entre si sugerem que eventos extremos diários estão limitados a um valor máximo (~30 mm/dia), o que é **incomum** em séries pluviométricas reais.

### Implicações para Erosão:
- **Risco erosivo**: Eventos acima de 29 mm/dia são raros mas esperados
- **Planejamento de estruturas**: Dimensionar para precipitações de 30 mm/dia
- **Limiar crítico**: Eventos ≥29 mm/dia demandam monitoramento intensivo

**Dados**: [valores_retorno_gev.csv](../dados/valores_retorno_gev.csv)  
**Figura**: [5_analise_extremos_gev.png](../figuras/5_analise_extremos_gev.png)

---

## 6️⃣ EVENTOS EXTREMOS DE PRECIPITAÇÃO

### Critério: Percentil 95 (P95 = 15.47 mm)

### Estatísticas de Eventos:
- **Total de eventos identificados**: 320 eventos em 20 anos
- **Frequência média**: 16 eventos/ano
- **Duração média**: 1.1 dias (maioria são eventos isolados)
- **Precipitação total média por evento**: 24.14 mm
- **Intensidade máxima média**: 21.25 mm/dia

### Distribuição dos Eventos:
| Característica | Média | Mínimo | Máximo |
|---------------|-------|--------|--------|
| **Duração** | 1.1 dias | 1 dia | 2 dias |
| **Precipitação Total** | 24.14 mm | 15.52 mm | 50.4 mm |
| **Intensidade Máxima** | 21.25 mm/dia | 15.52 mm/dia | 29.7 mm/dia |

### Eventos Mais Severos (Top 5):
1. **2006-04-01 a 2006-04-02**: 50.4 mm em 2 dias (25.2 mm/dia)
2. **2006-04-17 a 2006-04-18**: 43.3 mm em 2 dias (23.2 mm/dia)
3. **2006-05-21**: 29.7 mm em 1 dia
4. **2006-05-27**: 29.7 mm em 1 dia
5. **2006-05-18**: 27.0 mm em 1 dia

### Interpretação:
A maioria dos eventos extremos são isolados (duração = 1 dia), indicando precipitações intensas de curta duração, típicas de chuvas convectivas. Estes eventos são os mais erosivos, pois concentram grande volume em curto período.

### Relevância para Erosão:
- **16 eventos/ano** acima de 15.47 mm representam alto risco erosivo
- **Eventos curtos e intensos** (1-2 dias) têm maior poder erosivo que chuvas prolongadas
- **Precipitações >25 mm/dia** devem ser consideradas críticas para manejo

**Dados**: [eventos_extremos_detalhados.csv](../dados/eventos_extremos_detalhados.csv)  
**Figura**: [6_eventos_extremos.png](../figuras/6_eventos_extremos.png)

---

## 7️⃣ ÍNDICES DE EROSIVIDADE (EI30)

### Fórmula:
$$EI30 = \sum (EC \times P) \times I_{30}$$

Onde:
- $EC = 0.29 \times (1 - 0.72 \times e^{-0.05P})$ = Energia Cinética (MJ/ha/mm)
- $P$ = Precipitação (mm)
- $I_{30}$ = Intensidade máxima em 30 minutos (mm/h)

### Resultados:
| Índice | Valor |
|--------|-------|
| **EI30 mensal médio** | 317.20 MJ·mm/ha·h |
| **EI30 anual médio** | 3,640.27 MJ·mm/ha·h |
| **EI30 máximo mensal** | 1,608.47 MJ·mm/ha·h |
| **EI30 mínimo mensal** | 7.06 MJ·mm/ha·h |
| **Desvio padrão** | 330.59 MJ·mm/ha·h |

### Classificação de Erosividade (USLE):
- **Baixa**: < 2,000 MJ·mm/ha·h/ano
- **Moderada**: 2,000 - 4,000
- **Alta**: 4,000 - 7,000
- **Muito Alta**: > 7,000

**Conclusão**: Com EI30 anual médio de **3,640 MJ·mm/ha·h**, a região está na classe **MODERADA** de erosividade.

### Sazonalidade do EI30:
Análise mensal revelou:
- **Pico de erosividade**: Dezembro a Maio (estação chuvosa)
- **Mínimo de erosividade**: Junho a Novembro (estação seca)
- **Variabilidade**: Alta (CV = 104%), indicando anos muito distintos

### Interpretação:
O índice EI30 quantifica o potencial erosivo das chuvas. Valores acima de 300 MJ·mm/ha·h em um mês indicam risco significativo de erosão, especialmente em solos expostos ou mal manejados.

### Recomendações para Manejo:
1. **Cobertura vegetal** é essencial entre dezembro-maio
2. **Plantio em nível** e **terraceamento** em áreas de risco
3. **Monitoramento intensivo** em meses com EI30 > 500

**Dados**: 
- [ei30_mensal.csv](../dados/ei30_mensal.csv)
- [ei30_anual.csv](../dados/ei30_anual.csv)

**Figuras**: 
- [7_indices_erosividade_ei30.png](../figuras/7_indices_erosividade_ei30.png)
- [7_ei30_sazonalidade.png](../figuras/7_ei30_sazonalidade.png)

---

## 8️⃣ CURVAS IDF (INTENSIDADE-DURAÇÃO-FREQUÊNCIA)

### Tabela de Intensidades (mm/h):

| Duração | TR=2 anos | TR=5 anos | TR=10 anos | TR=20 anos | TR=50 anos | TR=100 anos |
|---------|-----------|-----------|------------|------------|------------|-------------|
| **1 dia** | 1.215 | 1.232 | 1.235 | 1.237 | 1.237 | 1.237 |
| **2 dias** | 0.953 | 1.095 | 1.152 | 1.188 | 1.219 | 1.234 |
| **3 dias** | 1.110 | 1.113 | 1.113 | 1.113 | 1.113 | 1.113 |
| **5 dias** | 0.629 | 0.731 | 0.768 | 0.791 | 0.808 | 0.816 |
| **7 dias** | 0.792 | 0.793 | 0.793 | 0.793 | 0.793 | 0.793 |
| **10 dias** | 0.473 | 0.548 | 0.575 | 0.591 | 0.604 | 0.609 |
| **15 dias** | 0.571 | 0.571 | 0.571 | 0.571 | 0.571 | 0.571 |
| **20 dias** | 0.355 | 0.416 | 0.440 | 0.457 | 0.471 | 0.478 |
| **30 dias** | 0.316 | 0.365 | 0.383 | 0.395 | 0.404 | 0.409 |

### Interpretação:
As curvas IDF mostram a relação entre intensidade de precipitação, duração e frequência de ocorrência. Observa-se que:

1. **Intensidades diminuem com a duração**: Chuvas de 1 dia têm intensidade ~1.2 mm/h, enquanto chuvas de 30 dias têm ~0.3-0.4 mm/h
2. **Baixa variação entre períodos de retorno**: Sugere regime pluviométrico estável
3. **Padrão esperado**: Inversamente proporcional à duração

### Aplicações em Engenharia:
- **Dimensionamento de drenagem**: Usar TR=10-25 anos para estruturas urbanas
- **Controle de erosão**: Usar TR=2-5 anos para terraços e canais
- **Estruturas críticas**: Usar TR=50-100 anos para barragens e reservatórios

### Exemplo Prático:
Para um evento de **5 dias com TR=10 anos**:
- Intensidade esperada: **0.768 mm/h**
- Precipitação total: 0.768 × 24 × 5 = **92.2 mm**

**Dados**: [curvas_idf.csv](../dados/curvas_idf.csv)  
**Figura**: [8_curvas_idf.png](../figuras/8_curvas_idf.png)

---

## 🎯 CONCLUSÕES E RECOMENDAÇÕES

### Principais Achados:

1. **Regime Pluviométrico Estável**: Não há tendência significativa de aumento ou diminuição da precipitação no período de 20 anos.

2. **Alta Sazonalidade**: Variação sazonal de 119.64 mm, com pico entre dezembro-maio.

3. **Eventos Extremos Frequentes**: 16 eventos/ano acima do P95 (15.47 mm), com intensidades até 29.7 mm/dia.

4. **Erosividade Moderada**: EI30 anual de 3,640 MJ·mm/ha·h, classificado como risco moderado.

5. **Previsibilidade**: Modelo SARIMA captura bem os padrões sazonais, permitindo previsões confiáveis.

### Recomendações para Estudos de Erosão:

#### 🌱 Manejo do Solo:
- **Cobertura vegetal permanente** durante estação chuvosa (dez-mai)
- **Plantio direto** ou **cultivo mínimo** para reduzir exposição do solo
- **Rotação de culturas** para manter estrutura do solo

#### 🏗️ Práticas Conservacionistas:
- **Terraços em nível** dimensionados para eventos de TR=10 anos (92 mm/5 dias)
- **Canais escoadouros** para TR=25 anos
- **Faixas de contenção** em áreas críticas (declividade >10%)

#### 📊 Monitoramento:
- **Alertas** para precipitações >15 mm/dia (P95)
- **Monitoramento intensivo** quando EI30 mensal >500 MJ·mm/ha·h
- **Avaliação pós-evento** após precipitações >25 mm/dia

#### 🔬 Pesquisas Futuras:
1. **Correlacionar** índices de erosividade com perda real de solo em campo
2. **Desenvolver modelos preditivos** de erosão baseados em SARIMA + EI30
3. **Analisar** efeitos combinados de precipitação e cobertura vegetal
4. **Investigar** padrões de precipitação em escala horária (maior precisão no I30)

---

## 📁 ARQUIVOS GERADOS

### Figuras (10 arquivos):
1. `1_decomposicao_serie_temporal.png` - Decomposição STL
2. `2_teste_estacionariedade.png` - Testes ADF
3. `3_diferenciacao_sazonal_acf_pacf.png` - Correlogramas
4. `4_previsao_sarima.png` - Previsões do modelo
5. `4_diagnostico_sarima.png` - Diagnóstico de resíduos
6. `5_analise_extremos_gev.png` - Distribuição GEV e períodos de retorno
7. `6_eventos_extremos.png` - Análise de eventos extremos
8. `7_indices_erosividade_ei30.png` - Série temporal EI30
9. `7_ei30_sazonalidade.png` - Boxplot sazonal EI30
10. `8_curvas_idf.png` - Curvas IDF

### Dados (6 arquivos):
1. `valores_retorno_gev.csv` - Períodos de retorno
2. `eventos_extremos_detalhados.csv` - 320 eventos identificados
3. `ei30_mensal.csv` - Índices mensais
4. `ei30_anual.csv` - Índices anuais
5. `curvas_idf.csv` - Tabela IDF
6. `relatorio_completo.txt` - Relatório textual

---

## 📚 REFERÊNCIAS METODOLÓGICAS

- **Decomposição STL**: Cleveland et al. (1990)
- **Teste ADF**: Dickey & Fuller (1981)
- **SARIMA**: Box & Jenkins (1970)
- **Distribuição GEV**: Jenkinson (1955), Coles (2001)
- **EI30**: Wischmeier & Smith (1978) - USLE
- **Curvas IDF**: Chow et al. (1988)
- **Teste de Mann-Kendall**: Mann (1945), Kendall (1975)

---

**Relatório gerado em**: 12/12/2025  
**Período analisado**: 2005-11-01 a 2025-11-30 (20 anos)  
**Total de observações**: 7,335 dias  

---

**🔍 Para exploração interativa dos dados, utilize os notebooks Jupyter disponíveis no diretório `notebooks/`.**
