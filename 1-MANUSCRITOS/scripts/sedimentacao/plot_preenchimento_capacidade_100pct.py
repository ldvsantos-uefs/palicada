from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> int:
    root = Path(__file__).resolve().parents[3]

    data_csv = (
        root
        / "2-DADOS"
        / "CLIMATOLOGIA_20ANOS"
        / "dados"
        / "dados_integrados_sedimentacao.csv"
    )

    out_png = (
        root
        / "1-MANUSCRITOS"
        / "1-CONTROLE_PLITOSSOLO"
        / "media"
        / "analises_estatisticas"
        / "22_composicao_deposicional_100pct.png"
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_csv)
    df["DATA"] = pd.to_datetime(df["DATA"], errors="coerce")

    # FRACIONADO no CSV integrado normalmente está em metros; como o gráfico é percentual,
    # a unidade se cancela. Ainda assim, padronizamos para cm para evitar confusão.
    df["FRAC_CM"] = pd.to_numeric(df["FRACIONADO"], errors="coerce").fillna(0.0) * 100.0
    df["FRAC_CM"] = df["FRAC_CM"].clip(lower=0.0)

    # Ordem lógica (para filtros/colunas)
    areas = ["SUP", "MED", "INF"]
    # Ordem de plotagem em barras empilhadas (bottom -> top).
    # Para aparecer visualmente de cima para baixo: SUP, MED, INF,
    # precisamos empilhar INF (base) -> MED -> SUP (topo).
    plot_order = ["INF", "MED", "SUP"]
    df = df[df["AREA"].isin(areas)].copy()
    df = df.dropna(subset=["DATA"])

    # Trabalhar em escala mensal e alinhar todas as barras ao início do mês.
    # Isso evita desalinhamento quando a coluna DATA não cai exatamente no 1º dia.
    df["MES"] = df["DATA"].dt.to_period("M").dt.to_timestamp(how="start")

    # Matriz MÊS x AREA com deposição incremental (cm)
    pivot = (
        df.pivot_table(index="MES", columns="AREA", values="FRAC_CM", aggfunc="sum")
        .fillna(0.0)
        .sort_index()
    )
    for a in areas:
        if a not in pivot.columns:
            pivot[a] = 0.0
    pivot = pivot[areas]

    row_sum = pivot.sum(axis=1)

    # Remove meses sem deposição (evita barras “fantasmas”/vazias)
    pivot = pivot.loc[row_sum > 0.0]
    row_sum = pivot.sum(axis=1)

    shares = pivot.div(row_sum.replace(0.0, pd.NA), axis=0).fillna(0.0) * 100.0

    # Plot 100% empilhado por mês
    # Estilo (paleta + rachuras) alinhado aos gráficos de referência (R/ggpattern).
    plt.rcParams["hatch.linewidth"] = 0.8

    fig, ax = plt.subplots(figsize=(14, 6.0), constrained_layout=True)

    # Eixo x categórico: 1 barra por mês com dados, sempre alinhada ao rótulo.
    months = shares.index.to_list()
    x = np.arange(len(months))
    x_labels = [pd.Timestamp(m).strftime("%b/%Y") for m in months]

    bottoms = pd.Series(0.0, index=shares.index)
    # RColorBrewer::brewer.pal(name = "Pastel1")
    colors = {
        "SUP": "#FBB4AE",  # pastel pink/red
        "MED": "#CCEBC5",  # pastel green
        "INF": "#B3CDE3",  # pastel blue
    }
    hatches = {
        "SUP": "///",  # stripe
        "MED": "xx",   # crosshatch
        "INF": "oo",   # circle
    }
    labels = {"SUP": "Superior", "MED": "Intermediário", "INF": "Inferior"}

    for area in plot_order:
        ax.bar(
            x,
            shares[area],
            bottom=bottoms,
            width=0.85,
            color=colors.get(area, "0.5"),
            edgecolor="0.2",
            linewidth=0.8,
            hatch=hatches.get(area, ""),
            label=labels.get(area, area),
            align="center",
        )
        bottoms = bottoms + shares[area]

    ax.set_ylim(0, 100)
    ax.set_ylabel("Composição mensal da deposição (%)")
    ax.grid(True, axis="y", alpha=0.25)

    # Legenda fora do quadro, abaixo e horizontal
    handles, legend_labels = ax.get_legend_handles_labels()
    desired = [labels[a] for a in areas]
    order_idx = [legend_labels.index(d) for d in desired if d in legend_labels]
    ax.legend(
        [handles[i] for i in order_idx],
        [legend_labels[i] for i in order_idx],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        frameon=False,
        ncol=3,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)

    ax.set_title(
        "Composição mensal da deposição incremental retida (método dos pinos)",
        fontsize=13,
    )

    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"OK: figura gerada em: {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
