"""\
Regenera e sincroniza figuras derivadas do dataset integrado de sedimentação
para o diretório do manuscrito.

Objetivo
- Atualizar os PNGs em 1-MANUSCRITOS/1-CONTROLE_PLITOSSOLO/media/analises_estatisticas
  após mudança em 2-DADOS/CLIMATOLOGIA_20ANOS/dados/dados_integrados_sedimentacao.csv.

Notas
- Este script NÃO reescreve o CSV integrado.
- Evita scripts que recalculam EI30 e sobrescrevem dados.

Saídas (no manuscrito)
- fig_series_temporais_integradas.png
- fig_sazonalidade_sedimentacao.png
- fig_extremos_fracionado.png
- fig_regressao_por_segmento.png
- fig_correlacoes_precipitacao_ei30_sedimentacao.png
- fig_matriz_correlacao_areas.png
- fig_distribuicao_ei30_sazonalidade.png
- fig_eficiencia_erosiva_ei30.png
- correlacao_lag_chuva.png

Uso
- python 1-MANUSCRITOS/scripts/sedimentacao/refresh_figuras_manuscrito.py
"""

from __future__ import annotations

import runpy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass(frozen=True)
class Paths:
    repo_dir: Path
    csv_integrado: Path
    out_manuscrito: Path
    out_climat_fig: Path
    out_relatorio_paralelo: Path


def _paths() -> Paths:
    repo_dir = Path(__file__).resolve().parents[3]

    csv_integrado = (
        repo_dir
        / "2-DADOS"
        / "CLIMATOLOGIA_20ANOS"
        / "dados"
        / "dados_integrados_sedimentacao.csv"
    )

    out_manuscrito = (
        repo_dir
        / "1-MANUSCRITOS"
        / "1-CONTROLE_PLITOSSOLO"
        / "media"
        / "analises_estatisticas"
    )

    out_climat_fig = repo_dir / "2-DADOS" / "CLIMATOLOGIA_20ANOS" / "figuras" / "sedimentacao"
    out_relatorio_paralelo = repo_dir / "2-DADOS" / "RELATORIO_PARALELO_SEDIMENTOS_media"

    return Paths(
        repo_dir=repo_dir,
        csv_integrado=csv_integrado,
        out_manuscrito=out_manuscrito,
        out_climat_fig=out_climat_fig,
        out_relatorio_paralelo=out_relatorio_paralelo,
    )


def _style():
    plt.style.use("ggplot")
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["figure.dpi"] = 300
    sns.set_context("notebook")


def _read_integrado(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "DATA" in df.columns:
        df["DATA"] = pd.to_datetime(df["DATA"], errors="coerce")
    return df


def _meteo_from_sup(df: pd.DataFrame) -> pd.DataFrame:
    meteo = (
        df[df["AREA"] == "SUP"][["DATA", "MONTH", "RAINFALL", "EI30"]]
        .dropna(subset=["DATA"])
        .sort_values("DATA")
    )
    if "RAINFALL" in meteo.columns:
        meteo = meteo.dropna(subset=["RAINFALL"], how="any")
    if "EI30" in meteo.columns:
        meteo = meteo.dropna(subset=["EI30"], how="any")

    meteo = meteo.drop_duplicates(subset=["DATA"], keep="last")
    return meteo


def _save_matrix_correlacao(df: pd.DataFrame, out_path: Path) -> None:
    required = {"AREA", "RAINFALL", "EI30", "SEDIMENT", "FRACIONADO"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes no CSV integrado: {sorted(missing)}")

    areas = [a for a in ["SUP", "MED", "INF"] if a in set(df["AREA"].dropna().astype(str))]
    if not areas:
        raise ValueError("Nenhuma área encontrada no CSV (SUP/MED/INF).")

    fig, axes = plt.subplots(1, len(areas), figsize=(5.6 * len(areas), 5.2))
    if len(areas) == 1:
        axes = np.array([axes])

    vars_ = ["SEDIMENT", "RAINFALL", "EI30", "FRACIONADO"]

    for idx, area in enumerate(areas):
        ax = axes[idx]
        g = df[df["AREA"] == area][vars_].dropna()
        if len(g) < 3:
            ax.text(0.5, 0.5, f"{area}\nDados insuficientes", ha="center", va="center", fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        corr = g.corr(numeric_only=True)
        sns.heatmap(
            corr,
            annot=True,
            fmt=".3f",
            cmap="coolwarm",
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=1,
            cbar=(idx == len(areas) - 1),
            cbar_kws={"shrink": 0.8},
            ax=ax,
        )
        ax.set_title(f"Matriz de correlação ({area})", fontweight="bold")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_ei30_sazonalidade(meteo: pd.DataFrame, out_path: Path) -> None:
    if meteo.empty:
        raise ValueError("Série meteorológica (SUP) vazia para EI30.")

    if "MONTH" not in meteo.columns or meteo["MONTH"].isna().all():
        meteo = meteo.copy()
        meteo["MONTH"] = meteo["DATA"].dt.month

    dfp = meteo.dropna(subset=["EI30", "MONTH"]).copy()
    dfp["MONTH"] = dfp["MONTH"].astype(int)

    fig, ax = plt.subplots(figsize=(12.5, 5.5))
    sns.boxplot(data=dfp, x="MONTH", y="EI30", ax=ax)
    ax.set_xlabel("Mês", fontweight="bold")
    ax.set_ylabel("EI30 (MJ·mm/ha·h)", fontweight="bold")
    ax.set_title("Distribuição sazonal do EI30 no período monitorado", fontweight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_eficiencia_erosiva(df: pd.DataFrame, out_path: Path) -> None:
    required = {"AREA", "RAINFALL", "EI30", "FRACIONADO"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes no CSV integrado: {sorted(missing)}")

    g = df.dropna(subset=["AREA", "RAINFALL", "EI30", "FRACIONADO"]).copy()
    if g.empty:
        raise ValueError("Sem dados suficientes para eficiência erosiva.")

    g["EFICIENCIA_EROSIVA"] = g["FRACIONADO"] / (g["RAINFALL"] + 0.01)

    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    palette = {"SUP": "tab:blue", "MED": "tab:green", "INF": "tab:red"}

    for area, gg in g.groupby("AREA"):
        ax.scatter(
            gg["EI30"].to_numpy(float),
            gg["EFICIENCIA_EROSIVA"].to_numpy(float),
            s=60,
            alpha=0.70,
            edgecolors="black",
            linewidths=0.6,
            label=str(area),
            color=palette.get(str(area), "gray"),
        )

    ax.set_xlabel("EI30 (MJ·mm/ha·h)", fontweight="bold")
    ax.set_ylabel("Eficiência erosiva (cm/mm)", fontweight="bold")
    ax.set_title("Eficiência erosiva em função do EI30", fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.legend(title="Segmento", frameon=False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _corr_at_lag(x: np.ndarray, y: np.ndarray, lag: int) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if lag < 0:
        xs = x[:lag]
        ys = y[-lag:]
    elif lag > 0:
        xs = x[lag:]
        ys = y[:-lag]
    else:
        xs = x
        ys = y

    mask = np.isfinite(xs) & np.isfinite(ys)
    xs = xs[mask]
    ys = ys[mask]
    if xs.size < 3:
        return float("nan")

    r = np.corrcoef(xs, ys)[0, 1]
    return float(r)


def _save_correlacao_lag(meteo: pd.DataFrame, df: pd.DataFrame, out_path: Path) -> None:
    """Figura de correlação por defasagem (lag) entre precipitação e sedimentação.

    - Apenas precipitação (RAINFALL) vs sedimentação incremental (FRACIONADO)
    - Série estratificada por segmento (SUP, MED, INF) em painéis separados (small multiples)
    - Lags em meses, limitados a [-6, +6] e ao tamanho amostral
    """

    areas = [a for a in ["SUP", "MED", "INF"] if a in set(df["AREA"].dropna().astype(str))]
    if not areas:
        raise ValueError("Nenhuma área encontrada no CSV (SUP/MED/INF).")

    met = meteo[["DATA", "RAINFALL"]].dropna(subset=["DATA"]).sort_values("DATA")
    met = met.drop_duplicates(subset=["DATA"], keep="last")

    merged_by_area: dict[str, pd.DataFrame] = {}
    for area in areas:
        dfa = df[df["AREA"] == area].copy()
        dfa = dfa.dropna(subset=["DATA", "FRACIONADO"]).sort_values("DATA")
        dff = dfa.merge(met, on="DATA", how="inner", suffixes=("", "_METEO"))
        if "RAINFALL" not in dff.columns and "RAINFALL_METEO" in dff.columns:
            dff["RAINFALL"] = dff["RAINFALL_METEO"]
        elif "RAINFALL_METEO" in dff.columns:
            dff["RAINFALL"] = dff["RAINFALL"].combine_first(dff["RAINFALL_METEO"])

        dff = dff.dropna(subset=["RAINFALL", "FRACIONADO"]).copy()
        merged_by_area[area] = dff

    min_n = min(len(v) for v in merged_by_area.values())
    if min_n < 10:
        raise ValueError(f"Dados insuficientes para lag (min_n={min_n}).")

    max_lag = min(6, min_n - 3)
    lags = list(range(-max_lag, max_lag + 1))

    fig, axes = plt.subplots(
        nrows=len(areas),
        ncols=1,
        figsize=(12.5, 7.6),
        sharex=True,
        sharey=True,
    )
    if len(areas) == 1:
        axes = [axes]

    # Manter padrão visual consistente com a Figura 8 (18_serie_eventos_extremos_ACUMULADA.png)
    # Cores por área (SUP/MED/INF)
    cores_areas = {"SUP": "saddlebrown", "MED": "darkolivegreen", "INF": "indigo"}

    # Diferenciar por traço para reforçar a identificação visual
    linestyles = {
        "SUP": "--",  # tracejado
        "MED": ":",   # pontilhado
        "INF": "-.",  # traço-ponto
    }

    def _lag_tick_label(v: float) -> str:
        i = int(v)
        unidade = "mês" if abs(i) == 1 else "meses"
        return f"{i} {unidade}"

    legend_handles = []
    legend_labels = []

    for ax, area in zip(axes, areas, strict=False):
        dff = merged_by_area[area]
        precip = dff["RAINFALL"].to_numpy(float)
        sed = dff["FRACIONADO"].to_numpy(float)
        corr = np.array([_corr_at_lag(precip, sed, lag) for lag in lags], dtype=float)

        lag_opt = lags[int(np.nanargmax(np.abs(corr)))]
        r_opt = float(corr[lags.index(lag_opt)])

        color = cores_areas.get(str(area), "gray")
        linestyle = linestyles.get(str(area), "-")

        ax.plot(
            lags,
            corr,
            marker="o",
            linewidth=2.4,
            color=color,
            linestyle=linestyle,
        )
        ax.axhline(0, color="black", linewidth=1, alpha=0.55)
        ax.axvline(x=lag_opt, color=color, linewidth=1.8, alpha=0.85, linestyle=linestyle)
        ax.scatter(
            [lag_opt],
            [r_opt],
            s=55,
            color=color,
            edgecolors="black",
            linewidths=0.5,
            zorder=5,
        )

        if area in cores_areas:
            from matplotlib.lines import Line2D

            legend_handles.append(
                Line2D([0], [0], color=color, lw=2.4, linestyle=linestyle, marker="o", markersize=5)
            )
            legend_labels.append(str(area))

        ax.text(
            0.02,
            0.92,
            f"{area}  lag ótimo {lag_opt} m  r={r_opt:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            fontweight="bold",
        )
        ax.set_ylabel("Correlação (r)", fontweight="bold")
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Defasagem (meses)", fontweight="bold", labelpad=14)
    axes[-1].set_xticks(lags)
    axes[-1].set_xticklabels([_lag_tick_label(v) for v in lags])
    for ax in axes:
        ax.set_ylim(-1.0, 1.0)

    fig.suptitle(
        "Defasagem temporal entre precipitação e sedimentação",
        fontweight="bold",
    )

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            ncol=min(3, len(legend_handles)),
            frameon=False,
            bbox_to_anchor=(0.5, -0.02),
            title="Segmento",
        )

    fig.tight_layout(rect=(0, 0.10, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _run_script(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    runpy.run_path(str(path), run_name="__main__")


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())


def main() -> None:
    p = _paths()
    _style()

    if not p.csv_integrado.exists():
        raise FileNotFoundError(f"CSV integrado não encontrado: {p.csv_integrado}")

    df = _read_integrado(p.csv_integrado)
    meteo = _meteo_from_sup(df)

    # 1) Gerar pacote do relatório paralelo (fig01/02/04/05) e copiar para o manuscrito
    script_relatorio = (
        p.repo_dir
        / "2-DADOS"
        / "CLIMATOLOGIA_20ANOS"
        / "scripts"
        / "sedimentacao"
        / "gerar_figuras_relatorio_paralelo.py"
    )
    _run_script(script_relatorio)

    map_relatorio = {
        "fig01_series_temporais_integradas.png": "fig_series_temporais_integradas.png",
        "fig02_sazonalidade_boxplot_fracionado.png": "fig_sazonalidade_sedimentacao.png",
        "fig04_distribuicao_extremos_fracionado.png": "fig_extremos_fracionado.png",
        "fig05_regressao_por_segmento.png": "fig_regressao_por_segmento.png",
    }
    for src_name, dst_name in map_relatorio.items():
        src = p.out_relatorio_paralelo / src_name
        dst = p.out_manuscrito / dst_name
        if src.exists():
            _copy(src, dst)
            print(f"✓ Atualizado: {dst.name}")
        else:
            print(f"⚠ Não encontrado (relatório paralelo): {src}")

    # 2) Correlações (Figura 10 simplificada) e copiar
    script_corr = (
        p.repo_dir
        / "2-DADOS"
        / "CLIMATOLOGIA_20ANOS"
        / "scripts"
        / "sedimentacao"
        / "figura_10_correlacoes_simplificada.py"
    )
    _run_script(script_corr)

    src_corr = p.out_climat_fig / "10_correlacoes_precipitacao_ei30_sedimentacao.png"
    if src_corr.exists():
        _copy(src_corr, p.out_manuscrito / "fig_correlacoes_precipitacao_ei30_sedimentacao.png")
        print("✓ Atualizado: fig_correlacoes_precipitacao_ei30_sedimentacao.png")
    else:
        print(f"⚠ Não encontrado: {src_corr}")

    # 3) Figuras complementares geradas aqui (sem sobrescrever CSV)
    _save_matrix_correlacao(df, p.out_manuscrito / "fig_matriz_correlacao_areas.png")
    print("✓ Atualizado: fig_matriz_correlacao_areas.png")

    _save_ei30_sazonalidade(meteo, p.out_manuscrito / "fig_distribuicao_ei30_sazonalidade.png")
    print("✓ Atualizado: fig_distribuicao_ei30_sazonalidade.png")

    _save_eficiencia_erosiva(df, p.out_manuscrito / "fig_eficiencia_erosiva_ei30.png")
    print("✓ Atualizado: fig_eficiencia_erosiva_ei30.png")

    _save_correlacao_lag(meteo, df, p.out_manuscrito / "correlacao_lag_chuva.png")
    print("✓ Atualizado: correlacao_lag_chuva.png")

    # 4) (Opcional) Regerar IDF apenas, mantendo padrão do manuscrito
    script_idf = p.repo_dir / "1-MANUSCRITOS" / "scripts" / "sedimentacao" / "figura_idf_apenas.py"
    _run_script(script_idf)

    print("\n✓ Refresh concluído")


if __name__ == "__main__":
    main()
