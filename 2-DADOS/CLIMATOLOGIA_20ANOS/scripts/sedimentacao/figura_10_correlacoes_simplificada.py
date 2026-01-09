"""\
Gera uma versão mais legível da Figura 10 (usada como Figura 12 no manuscrito)
com relações bivariadas entre sedimentação fracionada e (i) precipitação e (ii) EI30.

Saída
- CLIMATOLOGIA_20ANOS/figuras/sedimentacao/10_correlacoes_precipitacao_ei30_sedimentacao.png

Dados
- CLIMATOLOGIA_20ANOS/dados/dados_integrados_sedimentacao.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression


def _annotate_stats(ax, x, y, label_prefix=""):
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3:
        return

    r, p = stats.pearsonr(x, y)
    reg = LinearRegression().fit(x.reshape(-1, 1), y)
    r2 = reg.score(x.reshape(-1, 1), y)

    txt = f"{label_prefix}r = {r:.2f}  R² = {r2:.2f}  p = {p:.3f}"
    ax.text(
        0.02,
        0.98,
        txt,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="0.3"),
    )


def _plot_panel(ax, df, xcol, xlabel):
    ycol = "FRACIONADO"

    for area, g in df.groupby("AREA"):
        ax.scatter(
            g[xcol].values,
            g[ycol].values,
            s=55,
            alpha=0.70,
            edgecolors="black",
            linewidths=0.6,
            label=str(area),
        )

    x = df[xcol].to_numpy(dtype=float)
    y = df[ycol].to_numpy(dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if x.size >= 3:
        reg = LinearRegression().fit(x.reshape(-1, 1), y)
        x_grid = np.linspace(x.min(), x.max(), 200)
        y_grid = reg.predict(x_grid.reshape(-1, 1))
        ax.plot(x_grid, y_grid, "k--", linewidth=2.0)

    ax.set_xlabel(xlabel, fontweight="bold")
    ax.grid(True, alpha=0.25)

    _annotate_stats(ax, x, y)


def main():
    base_dir = Path(__file__).parent.parent.parent
    dados_path = base_dir / "dados" / "dados_integrados_sedimentacao.csv"
    out_path = base_dir / "figuras" / "sedimentacao" / "10_correlacoes_precipitacao_ei30_sedimentacao.png"

    df = pd.read_csv(dados_path)

    # Garantir colunas esperadas
    required = {"AREA", "RAINFALL", "EI30", "FRACIONADO"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes no dataset: {sorted(missing)}")

    df = df.dropna(subset=["RAINFALL", "EI30", "FRACIONADO", "AREA"]).copy()

    plt.style.use("ggplot")
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["figure.dpi"] = 300

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    _plot_panel(axes[0], df, "RAINFALL", "Precipitação (mm)")
    axes[0].set_ylabel("Sedimentação fracionada (cm)", fontweight="bold")
    axes[0].set_title("Sedimentação vs precipitação", fontweight="bold")

    _plot_panel(axes[1], df, "EI30", "EI30 (MJ·mm/ha·h)")
    axes[1].set_title("Sedimentação vs EI30", fontweight="bold")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Segmento", loc="lower center", ncol=3, frameon=False)

    fig.tight_layout(rect=(0, 0.08, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"✓ Figura salva: {out_path}")


if __name__ == "__main__":
    main()
