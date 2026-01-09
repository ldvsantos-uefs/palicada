"""\
Gera figuras essenciais para o RELATORIO_PARALELO_SEDIMENTOS.md.

Fontes de dados
- CLIMATOLOGIA_20ANOS/sediments/BD.xlsx (aba GRAFICO)
- CLIMATOLOGIA_20ANOS/dados/dados_integrados_sedimentacao.csv (inclui EI30 e DATA)

Saídas (PNG)
- RELATORIO_PARALELO_SEDIMENTOS_media/

Observação
- Não altera o artigo; apenas gera material para relatório paralelo.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression


def _prep_df() -> tuple[pd.DataFrame, pd.DataFrame]:
    repo_dir = Path(__file__).resolve().parents[3]

    xlsx_path = repo_dir / "CLIMATOLOGIA_20ANOS" / "sediments" / "BD.xlsx"
    df_xlsx = pd.read_excel(xlsx_path, sheet_name="GRAFICO")
    df_xlsx["DATA"] = pd.to_datetime(
        dict(year=df_xlsx["YEAR"], month=df_xlsx["MONTH"], day=1), errors="coerce"
    )

    csv_path = repo_dir / "CLIMATOLOGIA_20ANOS" / "dados" / "dados_integrados_sedimentacao.csv"
    df_csv = pd.read_csv(csv_path)
    df_csv["DATA"] = pd.to_datetime(df_csv["DATA"], errors="coerce")

    return df_xlsx, df_csv


def _style():
    plt.style.use("ggplot")
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["figure.dpi"] = 300
    sns.set_context("notebook")


def fig_time_series(out_dir: Path, df_csv: pd.DataFrame) -> Path:
    df = df_csv.dropna(subset=["DATA", "AREA", "FRACIONADO", "EI30"]).copy()

    # No CSV integrado, a coluna RAINFALL só está preenchida para o segmento SUP.
    # Para leitura integrada, usar a série pluviométrica (e EI30) como variável externa
    # comum aos segmentos, vinculada por DATA.
    meteo = (
        df_csv[df_csv["AREA"] == "SUP"][["DATA", "RAINFALL", "EI30"]]
        .dropna(subset=["DATA", "RAINFALL", "EI30"])
        .sort_values("DATA")
        .drop_duplicates(subset=["DATA"], keep="last")
    )

    fig, axes = plt.subplots(3, 1, figsize=(12.5, 8.5), sharex=True)

    for area, g in df.groupby("AREA"):
        g = g.sort_values("DATA")
        axes[0].plot(g["DATA"], g["FRACIONADO"], marker="o", linewidth=1.8, label=str(area))

    axes[1].plot(meteo["DATA"], meteo["RAINFALL"], marker="o", linewidth=1.8, color="black")
    axes[2].plot(meteo["DATA"], meteo["EI30"], marker="o", linewidth=1.8, color="black")

    axes[0].set_ylabel("Sedimentação fracionada (cm)", fontweight="bold")
    axes[1].set_ylabel("Precipitação (mm)", fontweight="bold")
    axes[2].set_ylabel("EI30 (MJ·mm/ha·h)", fontweight="bold")
    axes[2].set_xlabel("Data", fontweight="bold")

    axes[0].set_title("Série temporal integrada por segmento", fontweight="bold")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Segmento", loc="lower center", ncol=3, frameon=False)

    fig.tight_layout(rect=(0, 0.06, 1, 1))

    out_path = out_dir / "fig01_series_temporais_integradas.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return out_path


def fig_seasonality(out_dir: Path, df_csv: pd.DataFrame) -> Path:
    df = df_csv.dropna(subset=["MONTH", "AREA", "FRACIONADO"]).copy()
    df["MONTH"] = df["MONTH"].astype(int)

    fig, ax = plt.subplots(figsize=(12.5, 5.5))
    sns.boxplot(data=df, x="MONTH", y="FRACIONADO", hue="AREA", ax=ax)

    ax.set_xlabel("Mês", fontweight="bold")
    ax.set_ylabel("Sedimentação fracionada (cm)", fontweight="bold")
    ax.set_title("Distribuição mensal da sedimentação fracionada por segmento", fontweight="bold")
    ax.legend(title="Segmento", frameon=False)

    fig.tight_layout()
    out_path = out_dir / "fig02_sazonalidade_boxplot_fracionado.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return out_path


def fig_contributions(out_dir: Path, df_csv: pd.DataFrame) -> Path:
    df = df_csv.dropna(subset=["AREA", "FRACIONADO"]).copy()

    # contribuição por segmento no total do período
    totals = df.groupby("AREA")["FRACIONADO"].sum().sort_values(ascending=False)
    contrib = (totals / totals.sum() * 100).round(2)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(contrib.index.astype(str), contrib.values, edgecolor="black", linewidth=1.2)
    ax.set_ylabel("Contribuição para sedimentação total (%)", fontweight="bold")
    ax.set_xlabel("Segmento", fontweight="bold")
    ax.set_title("Contribuição relativa por segmento no período monitorado", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)

    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h, f"{h:.1f}%", ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    out_path = out_dir / "fig03_contribuicao_segmentos.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return out_path


def fig_extremes(out_dir: Path, df_csv: pd.DataFrame) -> Path:
    df = df_csv.dropna(subset=["FRACIONADO", "AREA"]).copy()

    # Limiares globais de extremos em FRACIONADO (P90 e P95)
    p90 = float(df["FRACIONADO"].quantile(0.90))
    p95 = float(df["FRACIONADO"].quantile(0.95))

    fig, ax = plt.subplots(figsize=(12.5, 5.5))
    sns.histplot(data=df, x="FRACIONADO", hue="AREA", element="step", stat="density", common_norm=False, ax=ax)

    ax.axvline(p90, color="black", linestyle="--", linewidth=1.8)
    ax.axvline(p95, color="black", linestyle="-", linewidth=2.2)
    ax.text(p90, ax.get_ylim()[1] * 0.95, "P90", ha="right", va="top", fontsize=10, rotation=90)
    ax.text(p95, ax.get_ylim()[1] * 0.95, "P95", ha="right", va="top", fontsize=10, rotation=90)

    ax.set_xlabel("Sedimentação fracionada (cm)", fontweight="bold")
    ax.set_ylabel("Densidade", fontweight="bold")
    ax.set_title("Distribuição e limiares de extremos da sedimentação fracionada", fontweight="bold")

    fig.tight_layout()
    out_path = out_dir / "fig04_distribuicao_extremos_fracionado.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return out_path


def _annotate_reg(ax, x: np.ndarray, y: np.ndarray) -> None:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3:
        return
    r, p = stats.pearsonr(x, y)
    model = LinearRegression().fit(x.reshape(-1, 1), y)
    r2 = model.score(x.reshape(-1, 1), y)
    ax.text(
        0.02,
        0.98,
        f"r={r:.2f}  R²={r2:.2f}  p={p:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="0.3"),
    )


def fig_regress_by_segment(out_dir: Path, df_csv: pd.DataFrame) -> Path:
    # Preparar tabela com meteo por DATA e sedimentação por segmento
    df = df_csv.dropna(subset=["DATA", "AREA", "FRACIONADO", "EI30"]).copy()
    df["DATA"] = pd.to_datetime(df["DATA"], errors="coerce")
    df["FRAC_POS"] = df["FRACIONADO"].clip(lower=0)

    meteo = (
        df_csv[df_csv["AREA"] == "SUP"][["DATA", "RAINFALL", "EI30"]]
        .dropna(subset=["DATA", "RAINFALL", "EI30"])
        .sort_values("DATA")
        .drop_duplicates(subset=["DATA"], keep="last")
    )
    df = df.merge(meteo[["DATA", "RAINFALL", "EI30"]], on="DATA", how="left", suffixes=("", "_METEO"))
    if "RAINFALL_METEO" in df.columns:
        df["RAINFALL"] = df["RAINFALL_METEO"].combine_first(df.get("RAINFALL"))
    if "EI30_METEO" in df.columns:
        df["EI30"] = df["EI30_METEO"].combine_first(df.get("EI30"))
    df = df.drop(columns=[c for c in ["RAINFALL_METEO", "EI30_METEO"] if c in df.columns])
    df = df.dropna(subset=["RAINFALL", "EI30", "FRAC_POS"]).copy()

    areas = [a for a in ["SUP", "MED", "INF"] if a in df["AREA"].unique()]
    fig, axes = plt.subplots(len(areas), 2, figsize=(12.5, 8.5), sharey=True)
    if len(areas) == 1:
        axes = np.array([axes])

    for i, area in enumerate(areas):
        g = df[df["AREA"] == area]

        # Chuva
        ax = axes[i, 0]
        ax.scatter(g["RAINFALL"], g["FRAC_POS"], s=60, alpha=0.7, edgecolors="black", linewidths=0.6)
        if len(g) >= 3:
            x = g["RAINFALL"].to_numpy(float)
            y = g["FRAC_POS"].to_numpy(float)
            model = LinearRegression().fit(x.reshape(-1, 1), y)
            xg = np.linspace(np.nanmin(x), np.nanmax(x), 200)
            ax.plot(xg, model.predict(xg.reshape(-1, 1)), "k--", linewidth=2)
            _annotate_reg(ax, x, y)
        ax.set_title(f"{area} — Chuva", fontweight="bold")
        ax.set_xlabel("Precipitação (mm)")
        ax.grid(True, alpha=0.25)

        # EI30
        ax = axes[i, 1]
        ax.scatter(g["EI30"], g["FRAC_POS"], s=60, alpha=0.7, edgecolors="black", linewidths=0.6)
        if len(g) >= 3:
            x = g["EI30"].to_numpy(float)
            y = g["FRAC_POS"].to_numpy(float)
            model = LinearRegression().fit(x.reshape(-1, 1), y)
            xg = np.linspace(np.nanmin(x), np.nanmax(x), 200)
            ax.plot(xg, model.predict(xg.reshape(-1, 1)), "k--", linewidth=2)
            _annotate_reg(ax, x, y)
        ax.set_title(f"{area} — EI30", fontweight="bold")
        ax.set_xlabel("EI30 (MJ·mm/ha·h)")
        ax.grid(True, alpha=0.25)

    for i in range(len(areas)):
        axes[i, 0].set_ylabel("Sedimentação fracionada positiva (cm)", fontweight="bold")

    fig.suptitle("Ajuste linear por segmento (FRACIONADO positivo)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path = out_dir / "fig05_regressao_por_segmento.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return out_path


def main():
    repo_dir = Path(__file__).resolve().parents[3]
    out_dir = repo_dir / "RELATORIO_PARALELO_SEDIMENTOS_media"
    out_dir.mkdir(parents=True, exist_ok=True)

    _style()
    df_xlsx, df_csv = _prep_df()

    # Preferir df_csv para ter EI30 e DATA padronizados.
    paths = []
    paths.append(fig_time_series(out_dir, df_csv))
    paths.append(fig_seasonality(out_dir, df_csv))
    paths.append(fig_contributions(out_dir, df_csv))
    paths.append(fig_extremes(out_dir, df_csv))
    paths.append(fig_regress_by_segment(out_dir, df_csv))

    print("\n✓ Figuras geradas")
    for p in paths:
        print(" -", p)


if __name__ == "__main__":
    main()
