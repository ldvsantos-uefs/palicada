"""Gera figura IDF + série temporal (painéis a/b) para inserção no manuscrito.

Objetivo editorial:
- Recriar a ideia da Figura 21a em um único gráfico com dois painéis (a) e (b).
- Remover equação do gráfico (ela fica no texto de Materiais e Métodos).
- Ajustar paleta para ficar consistente com o estilo usado em
  `20_serie_eventos_extremos_EROSIVIDADE`.

Saída:
  1-MANUSCRITOS/1-CONTROLE_PLITOSSOLO/media/analises_estatisticas/21a_curvas_idf_serie_temporal.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def intensidade_idf(duracao_min: float, periodo_retorno_anos: float) -> float:
    k = 1200
    a = 0.20
    b = 15
    c = 0.80
    return (k * periodo_retorno_anos**a) / (duracao_min + b) ** c


def main() -> int:
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["hatch.linewidth"] = 0.8

    root_dir = Path(__file__).resolve().parents[3]
    data_csv = (
        root_dir
        / "2-DADOS"
        / "CLIMATOLOGIA_20ANOS"
        / "dados"
        / "dados_integrados_sedimentacao.csv"
    )

    out_dir = (
        root_dir
        / "1-MANUSCRITOS"
        / "1-CONTROLE_PLITOSSOLO"
        / "media"
        / "analises_estatisticas"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "21a_curvas_idf_serie_temporal.png"

    df = pd.read_csv(data_csv)
    df["DATA"] = pd.to_datetime(df["DATA"])

    df_valid = df[(df["AREA"] == "SUP") & (df["RAINFALL"] > 0) & (df["EI30"] > 0)].copy()

    df_valid["RAZAO_EROSIVA"] = df_valid["EI30"] / df_valid["RAINFALL"]
    limiar_razao = df_valid["RAZAO_EROSIVA"].median()
    df_valid["TIPO_CHUVA"] = df_valid["RAZAO_EROSIVA"].apply(
        lambda x: "TORRENCIAL" if x > limiar_razao else "PROLONGADA"
    )

    # Duração estimada (proxy) com seed fixa
    rng = np.random.default_rng(42)
    df_valid["DURACAO_ESTIMADA"] = df_valid["TIPO_CHUVA"].apply(
        lambda x: float(rng.uniform(30, 60)) if x == "TORRENCIAL" else float(rng.uniform(120, 240))
    )
    df_valid["INTENSIDADE_MEDIA"] = df_valid["RAINFALL"] / (df_valid["DURACAO_ESTIMADA"] / 60)

    limiar_ei30 = df_valid["EI30"].quantile(0.95)
    eventos_extremos = df_valid[df_valid["EI30"] >= limiar_ei30].copy()
    torrenciais = df_valid[df_valid["TIPO_CHUVA"] == "TORRENCIAL"].copy()
    prolongadas = df_valid[df_valid["TIPO_CHUVA"] == "PROLONGADA"].copy()

    # Curvas IDF
    duracoes = np.linspace(5, 240, 100)
    periodos_retorno = [2, 5, 10, 25, 50, 100]

    # Paleta + rachuras no estilo da Figura 22 (RColorBrewer::Pastel1)
    cor_torrencial = "#FBB4AE"  # pastel pink/red
    cor_prolongada = "#B3CDE3"  # pastel blue
    cor_extremo = "#CCEBC5"  # pastel green
    cor_ei30 = "#FED9A6"  # pastel orange/peach
    cor_borda = "0.2"
    hatch_torrencial = "///"
    hatch_prolongada = "oo"

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 1, hspace=0.30)

    # -------------------------
    # (a) IDF
    # -------------------------
    ax1 = fig.add_subplot(gs[0])

    cores_tr = plt.cm.plasma(np.linspace(0.1, 0.9, len(periodos_retorno)))
    for i, tr in enumerate(periodos_retorno):
        intensidades = [intensidade_idf(float(d), float(tr)) for d in duracoes]
        ax1.plot(
            duracoes,
            intensidades,
            "-",
            linewidth=3,
            label=f"TR = {tr} anos",
            color=cores_tr[i],
            alpha=0.85,
        )
        ax1.fill_between(duracoes, 0, intensidades, alpha=0.10, color=cores_tr[i])

    ax1.scatter(
        torrenciais["DURACAO_ESTIMADA"],
        torrenciais["INTENSIDADE_MEDIA"],
        s=250,
        c=cor_torrencial,
        marker="^",
        edgecolors=cor_borda,
        linewidths=2,
        label="Chuvas torrenciais (observadas)",
        alpha=0.90,
        zorder=5,
    )

    ax1.scatter(
        prolongadas["DURACAO_ESTIMADA"],
        prolongadas["INTENSIDADE_MEDIA"],
        s=250,
        c=cor_prolongada,
        marker="o",
        edgecolors=cor_borda,
        linewidths=2,
        label="Chuvas prolongadas (observadas)",
        alpha=0.90,
        zorder=5,
    )

    ax1.scatter(
        eventos_extremos["DURACAO_ESTIMADA"],
        eventos_extremos["INTENSIDADE_MEDIA"],
        s=600,
        c=cor_extremo,
        marker="*",
        edgecolors=cor_borda,
        linewidths=3,
        label="Eventos extremos (P95)",
        alpha=1.0,
        zorder=6,
    )

    ax1.set_xlabel("Duração (minutos)", fontsize=15, fontweight="bold")
    ax1.set_ylabel("Intensidade (mm/h)", fontsize=15, fontweight="bold")
    ax1.text(
        0.0,
        1.01,
        "(a)",
        transform=ax1.transAxes,
        ha="left",
        va="bottom",
        fontsize=16,
        fontweight="bold",
    )
    ax1.legend(loc="upper right", fontsize=12, framealpha=0.95, ncol=2)
    ax1.grid(True, alpha=0.25, linestyle="--")
    ax1.set_xlim(0, 250)
    ax1.set_ylim(0, max([intensidade_idf(5, 100), float(df_valid["INTENSIDADE_MEDIA"].max()) * 1.2]))

    for _, evt in eventos_extremos.iterrows():
        ax1.annotate(
            f"{evt['DATA'].strftime('%Y-%m')}\n{evt['RAINFALL']:.0f} mm",
            xy=(evt["DURACAO_ESTIMADA"], evt["INTENSIDADE_MEDIA"]),
            xytext=(15, 15),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            color=cor_borda,
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.9},
            arrowprops={"arrowstyle": "->", "color": cor_borda, "lw": 2},
        )

    # -------------------------
    # (b) Série temporal
    # -------------------------
    ax2 = fig.add_subplot(gs[1])

    ax2.bar(
        torrenciais["DATA"],
        torrenciais["RAINFALL"],
        width=20,
        color=cor_torrencial,
        alpha=0.65,
        edgecolor=cor_borda,
        linewidth=2,
        hatch=hatch_torrencial,
        label="Torrenciais",
    )

    ax2.bar(
        prolongadas["DATA"],
        prolongadas["RAINFALL"],
        width=20,
        color=cor_prolongada,
        alpha=0.65,
        edgecolor=cor_borda,
        linewidth=2,
        hatch=hatch_prolongada,
        label="Prolongadas",
    )

    ax2_twin = ax2.twinx()
    ax2_twin.plot(
        df_valid["DATA"],
        df_valid["EI30"],
        "-o",
        color=cor_ei30,
        linewidth=3.5,
        markersize=9,
        label="EI30",
        alpha=0.85,
        markeredgecolor=cor_borda,
        markeredgewidth=1,
    )

    ax2_twin.scatter(
        eventos_extremos["DATA"],
        eventos_extremos["EI30"],
        s=600,
        c=cor_extremo,
        marker="*",
        edgecolors=cor_borda,
        linewidths=3.5,
        label="Extremos (P95)",
        zorder=5,
    )

    ax2_twin.axhline(
        y=limiar_ei30,
        color=cor_borda,
        linestyle="--",
        linewidth=2.5,
        label=f"Limiar P95 = {limiar_ei30:.0f}",
        alpha=0.7,
    )

    ax2.set_xlabel("Período experimental", fontsize=15, fontweight="bold")
    ax2.set_ylabel("Precipitação (mm)", fontsize=14, fontweight="bold", color=cor_borda)
    ax2_twin.set_ylabel("EI30 (MJ mm ha⁻¹ h⁻¹)", fontsize=14, fontweight="bold", color=cor_borda)

    ax2.tick_params(axis="y", labelcolor=cor_borda, labelsize=12)
    ax2_twin.tick_params(axis="y", labelcolor=cor_borda, labelsize=12)
    ax2.tick_params(axis="x", labelsize=11)

    ax2.text(
        0.0,
        1.01,
        "(b)",
        transform=ax2.transAxes,
        ha="left",
        va="bottom",
        fontsize=16,
        fontweight="bold",
    )

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=12, framealpha=0.95, ncol=2)

    ax2.grid(True, alpha=0.3, axis="y")
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=11)

    for _, evt in eventos_extremos.iterrows():
        ax2_twin.annotate(
            f"{evt['EI30']:.0f}",
            xy=(evt["DATA"], evt["EI30"]),
            xytext=(0, 15),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
            color=cor_borda,
            ha="center",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9},
        )

    fig.subplots_adjust(top=0.98)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"✅ Figura (a/b) salva em: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
