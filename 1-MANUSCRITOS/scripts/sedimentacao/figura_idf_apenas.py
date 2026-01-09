"""Gera figura IDF (apenas curvas) para inserção no manuscrito.

- Remove painel de série temporal.
- Não escreve equação no título do gráfico.

Saída:
  1-MANUSCRITOS/1-CONTROLE_PLITOSSOLO/media/analises_estatisticas/21_curvas_idf_apenas.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def intensidade_idf(duracao_min: float, periodo_retorno_anos: float) -> float:
    # Mesmos parâmetros usados nas rotinas anteriores do projeto
    k = 1200
    a = 0.20
    b = 15
    c = 0.80
    return (k * periodo_retorno_anos**a) / (duracao_min + b) ** c


def main() -> int:
    plt.style.use("ggplot")
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["figure.dpi"] = 300

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
    out_path = out_dir / "21_curvas_idf_apenas.png"

    df = pd.read_csv(data_csv)
    df["DATA"] = pd.to_datetime(df["DATA"])

    df_valid = df[(df["AREA"] == "SUP") & (df["RAINFALL"] > 0) & (df["EI30"] > 0)].copy()
    df_valid["RAZAO_EROSIVA"] = df_valid["EI30"] / df_valid["RAINFALL"]
    limiar_razao = df_valid["RAZAO_EROSIVA"].median()
    df_valid["TIPO_CHUVA"] = df_valid["RAZAO_EROSIVA"].apply(
        lambda x: "TORRENCIAL" if x > limiar_razao else "PROLONGADA"
    )

    # Mantém a mesma lógica de duração estimada (proxy) do script anterior
    # (seed fixa para reprodutibilidade)
    rng = np.random.default_rng(42)
    df_valid["DURACAO_ESTIMADA"] = df_valid["TIPO_CHUVA"].apply(
        lambda x: float(rng.uniform(30, 60)) if x == "TORRENCIAL" else float(rng.uniform(120, 240))
    )
    df_valid["INTENSIDADE_MEDIA"] = df_valid["RAINFALL"] / (df_valid["DURACAO_ESTIMADA"] / 60)

    limiar_ei30 = df_valid["EI30"].quantile(0.95)
    eventos_extremos = df_valid[df_valid["EI30"] >= limiar_ei30].copy()
    torrenciais = df_valid[df_valid["TIPO_CHUVA"] == "TORRENCIAL"]
    prolongadas = df_valid[df_valid["TIPO_CHUVA"] == "PROLONGADA"]

    duracoes = np.linspace(5, 240, 100)
    periodos_retorno = [2, 5, 10, 25, 50, 100]

    # Mantém o "estilo" da Figura 21a (painel IDF) porém sem o painel temporal
    # e sem equação/texto no título da figura.
    fig, ax = plt.subplots(figsize=(18, 9))

    cores_tr = plt.cm.plasma(np.linspace(0.1, 0.9, len(periodos_retorno)))
    for i, tr in enumerate(periodos_retorno):
        intensidades = [intensidade_idf(float(d), float(tr)) for d in duracoes]
        ax.plot(
            duracoes,
            intensidades,
            "-",
            linewidth=3,
            label=f"TR = {tr} anos",
            color=cores_tr[i],
            alpha=0.85,
        )
        ax.fill_between(duracoes, 0, intensidades, alpha=0.10, color=cores_tr[i])

    ax.scatter(
        torrenciais["DURACAO_ESTIMADA"],
        torrenciais["INTENSIDADE_MEDIA"],
        s=250,
        c="darkorange",
        marker="^",
        edgecolors="saddlebrown",
        linewidths=2,
        label="Chuvas TORRENCIAIS (observadas)",
        alpha=0.90,
        zorder=5,
    )
    ax.scatter(
        prolongadas["DURACAO_ESTIMADA"],
        prolongadas["INTENSIDADE_MEDIA"],
        s=250,
        c="steelblue",
        marker="o",
        edgecolors="navy",
        linewidths=2,
        label="Chuvas PROLONGADAS (observadas)",
        alpha=0.90,
        zorder=5,
    )
    ax.scatter(
        eventos_extremos["DURACAO_ESTIMADA"],
        eventos_extremos["INTENSIDADE_MEDIA"],
        s=600,
        c="gold",
        marker="*",
        edgecolors="darkorange",
        linewidths=3.0,
        label="Eventos EXTREMOS P95",
        alpha=1.0,
        zorder=6,
    )

    ax.set_xlabel("Duração (minutos)", fontsize=15, fontweight="bold")
    ax.set_ylabel("Intensidade (mm/h)", fontsize=15, fontweight="bold")
    ax.set_title(
        "Curvas IDF regionais e eventos observados no experimento\n"
        "Classificação: chuvas torrenciais vs prolongadas",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )

    ax.legend(loc="upper right", fontsize=12, framealpha=0.95, ncol=2)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlim(0, 250)
    ax.set_ylim(0, max([intensidade_idf(5, 100), float(df_valid["INTENSIDADE_MEDIA"].max()) * 1.2]))

    # Anotações para eventos extremos (mesmo espírito da 21a)
    for _, evt in eventos_extremos.iterrows():
        ax.annotate(
            f"{evt['DATA'].strftime('%Y-%m')}\n{evt['RAINFALL']:.0f} mm",
            xy=(evt["DURACAO_ESTIMADA"], evt["INTENSIDADE_MEDIA"]),
            xytext=(15, 15),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            color="darkorange",
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "yellow", "alpha": 0.8},
            arrowprops={"arrowstyle": "->", "color": "darkorange", "lw": 2},
        )

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")

    print(f"✅ Figura IDF (apenas) salva em: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
