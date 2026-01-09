"""\
Analisa relações entre sedimentação fracionada e variáveis climáticas (precipitação, EI30)
usando regressão polinomial de segundo grau por segmento.

Saídas:
- Gráficos de dispersão com curvas de ajuste polinomial
- Tabela comparativa de R² por segmento e variável
- Diagnóstico estatístico completo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy import stats
import seaborn as sns
from pathlib import Path


def main():
    # Configurações
    plt.style.use("ggplot")
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["figure.dpi"] = 300
    sns.set_palette("viridis")

    # Diretórios
    BASE_DIR = Path(__file__).parent.parent.parent
    DADOS_PATH = BASE_DIR / "dados" / "dados_integrados_sedimentacao.csv"
    FIGURAS_DIR = BASE_DIR / "figuras" / "sedimentacao"
    RELATORIO_PATH = BASE_DIR / "dados" / "relatorio_regressao_polinomial.txt"

    # Carregar dados
    df = pd.read_csv(DADOS_PATH)
    df = df.dropna(subset=["RAINFALL", "EI30", "FRACIONADO", "AREA"]).copy()

    # Preparar saída
    FIGURAS_DIR.mkdir(parents=True, exist_ok=True)
    output_lines = ["ANÁLISE DE REGRESSÃO POLINOMIAL (GRAU 2) POR SEGMENTO\n"]
    output_lines.append("=" * 70)

    # Analisar cada segmento
    fig, axes = plt.subplots(3, 2, figsize=(14, 16))
    areas = ["SUP", "MED", "INF"]
    variaveis = [("RAINFALL", "Precipitação (mm)"), ("EI30", "EI30 (MJ·mm/ha·h)")]

    for i, area in enumerate(areas):
        df_area = df[df["AREA"] == area]
        output_lines.append(f"\nSEGMENTO: {area}")
        output_lines.append("-" * 30)

        for j, (var, xlabel) in enumerate(variaveis):
            # Dados
            X = df_area[var].values.reshape(-1, 1)
            y = df_area["FRACIONADO"].values

            # Regressão linear simples (grau 1)
            model_linear = LinearRegression().fit(X, y)
            y_pred_linear = model_linear.predict(X)
            r2_linear = r2_score(y, y_pred_linear)

            # Regressão polinomial (grau 2)
            poly = PolynomialFeatures(degree=2)
            X_poly = poly.fit_transform(X)
            model_poly = LinearRegression().fit(X_poly, y)
            y_pred_poly = model_poly.predict(X_poly)
            r2_poly = r2_score(y, y_pred_poly)

            # Teste F para comparar modelos
            ssr_linear = ((y - y_pred_linear) ** 2).sum()
            ssr_poly = ((y - y_pred_poly) ** 2).sum()
            df_diff = X_poly.shape[1] - 2  # graus de liberdade: poly(2) tem 3 parâmetros vs linear(1) tem 2
            f_stat = ((ssr_linear - ssr_poly) / df_diff) / (ssr_poly / (len(y) - X_poly.shape[1]))
            p_value = 1 - stats.f.cdf(f_stat, df_diff, len(y) - X_poly.shape[1])

            # Armazenar resultados
            output_lines.append(f"Variável: {xlabel}")
            output_lines.append(f"  R² linear: {r2_linear:.4f}")
            output_lines.append(f"  R² polinomial: {r2_poly:.4f}")
            output_lines.append(f"  Teste F (linear vs polinomial): F={f_stat:.2f}, p={p_value:.4f}")
            output_lines.append(f"  Coeficientes polinomiais: {model_poly.coef_}")

            # Plot
            ax = axes[i, j]
            ax.scatter(X, y, s=70, alpha=0.7, edgecolor="k", label="Dados")
            
            # Ordenar para plot da curva
            x_grid = np.linspace(X.min(), X.max(), 300)
            x_grid_poly = poly.transform(x_grid.reshape(-1, 1))
            y_grid = model_poly.predict(x_grid_poly)
            ax.plot(x_grid, y_grid, "r-", linewidth=2.5, label="Ajuste polinomial (grau 2)")
            
            ax.set_title(f"{area} - {xlabel}", fontweight="bold")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Sedimentação fracionada (cm)")
            ax.legend()
            ax.grid(True, alpha=0.25)
            
            # Anotar estatísticas
            ax.text(0.02, 0.95, f"$R^2$ = {r2_poly:.3f}", transform=ax.transAxes,
                    fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

    # Ajustar layout e salvar
    plt.tight_layout()
    plt.savefig(FIGURAS_DIR / "regressao_polinomial_por_segmento.png", dpi=300)
    plt.close()

    # Salvar relatório
    with open(RELATORIO_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    print("✓ Análise concluída!")
    print(f"  - Gráficos salvos em: {FIGURAS_DIR / 'regressao_polinomial_por_segmento.png'}")
    print(f"  - Relatório salvo em: {RELATORIO_PATH}")


if __name__ == "__main__":
    main()
