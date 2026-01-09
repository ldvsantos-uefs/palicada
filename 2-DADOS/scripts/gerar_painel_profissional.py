import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
import logging

# Configurar backend e logging
matplotlib.use("Agg")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURAÇÕES DE ESTILO (GGPLOT)
# =============================================================================

def configure_style():
    """Configura estilo ggplot profissional"""
    plt.rcdefaults()
    plt.style.use('ggplot')
    
    font_name = "DejaVu Sans"
    try:
        if any("Poppins" in f.name for f in fm.fontManager.ttflist):
            font_name = "Poppins"
    except:
        pass

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [font_name, "Arial", "sans-serif"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.labelweight": "bold",
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.title_fontsize": 10,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        
        "axes.facecolor": "#EBEBEB",
        "grid.color": "white",
        "grid.linewidth": 1.2,
        "grid.alpha": 1.0,
        "axes.edgecolor": "white",
        "axes.linewidth": 0,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "text.color": "#333333",
        "axes.labelcolor": "#333333",
        "legend.frameon": False,
    })
    return font_name

# Cores Ajustadas
COLORS = {
    "optimistic": "#00BA38",  # Verde (Bom)
    "baseline":   "#C77CFF",  # Lilás (Referência)
    "pessimistic":"#F8766D",  # Vermelho (Ruim)
    "p90":        "#FBC15E",  # Laranja
    "p95":        "#619CFF",  # Azul
    "veg":        "#444444",  # Cinza Escuro (Significativo)
}

SCENARIO_CFG = {
    "optimistic":   {"label": "Otimista (k=0.03)", "color": COLORS["optimistic"], "marker": "o"},
    "baseline":     {"label": "Ref. (k=0.06)",     "color": COLORS["baseline"],   "marker": "s"},
    "pessimistic":  {"label": "Pessimista (k=0.10)","color": COLORS["pessimistic"],"marker": "^"},
}

CAP_CFG = {
    "p90": {"label": "Capacidade P90", "color": COLORS["p90"], "marker": "o"},
    "p95": {"label": "Capacidade P95", "color": COLORS["p95"], "marker": "s"},
    "veg": {"label": "Fator Veg.",     "color": COLORS["veg"], "marker": "D"},
}

# =============================================================================
# PLOTAGEM
# =============================================================================

def plot_panel(ax, df, x_col, y_cols, configs, title, ylabel, is_panel_b=False):
    years = df[x_col].unique()
    mark_every = [i for i, y in enumerate(years) if y % 2 == 0]

    handles = []
    labels = []

    for col_name, cfg_key in y_cols:
        cfg = configs[cfg_key]
        
        if is_panel_b:
            y_data = df[col_name]
            label = cfg["label"]
            color = cfg["color"]
            marker = cfg["marker"]
            linestyle = "--" if cfg_key == "veg" else "-"
            
            line, = ax.plot(
                years, y_data,
                label=label,
                color=color,
                marker=marker,
                markevery=mark_every,
                linewidth=2.5,
                markersize=7,
                linestyle=linestyle,
                markeredgecolor="white",
                markeredgewidth=1.0
            )
            handles.append(line)
            labels.append(label)

    ax.set_title(title, pad=15)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Tempo (anos)")
    ax.set_xticks(years[::2])
    
    # Margem extra no final (até 10.5)
    ax.set_xlim(left=0, right=10.5)
    ax.set_ylim(bottom=0)
    
    return handles, labels

def main():
    font = configure_style()
    logger.info(f"Fonte configurada: {font}")

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "1-MANUSCRITOS" / "1-CONTROLE_PLITOSSOLO" / "media" / "figuras"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        df_res = pd.read_csv(repo_root / "2-DADOS" / "simulacoes" / "resistencia_temporal.csv")
        df_cap = pd.read_csv(repo_root / "2-DADOS" / "simulacoes" / "capacidade_temporal.csv")
        
        # Tentativa de ler dados de literatura (opcional)
        try:
            df_lit_res = pd.read_csv(repo_root / "2-DADOS" / "simulacoes" / "dados_literatura_resistencia.csv")
            df_lit_cap = pd.read_csv(repo_root / "2-DADOS" / "simulacoes" / "dados_literatura_capacidade.csv")
        except FileNotFoundError:
            df_lit_res = None
            df_lit_cap = None
            logger.warning("Arquivos de literatura não encontrados. Plotando apenas simulações.")

    except Exception as e:
        logger.error(f"Erro ao ler dados obrigatórios: {e}")
        return 1

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- PAINEL A: Resistência ---
    ax_a = axes[0]
    years = df_res["year"].unique()
    mark_every = [i for i, y in enumerate(years) if y % 2 == 0]
    
    handles_a = []
    labels_a = []
    
    # Ordem explícita: Otimista (Topo) -> Baseline (Meio) -> Pessimista (Fundo)
    plot_order = ["optimistic", "baseline", "pessimistic"]
    
    for scenario in plot_order:
        grp = df_res[df_res["scenario"] == scenario]
        if grp.empty: continue
        
        cfg = SCENARIO_CFG.get(scenario, {})
        line, = ax_a.plot(
            grp["year"], grp["tensile_strength_mpa"],
            label=cfg["label"], color=cfg["color"], marker=cfg["marker"],
            markevery=mark_every, linewidth=2.5, markersize=7,
            markeredgecolor="white", markeredgewidth=1.0
        )
        handles_a.append(line)
        labels_a.append(cfg["label"])

    # Adicionar Scatter de Literatura se existir
    if df_lit_res is not None:
        # Escalar % para MPa (Assumindo 180 MPa como base inicial típica)
        df_lit_res["mpa_equiv"] = (df_lit_res["integrity_pct"] / 100.0) * 180.0
        scatter_a = ax_a.scatter(
            df_lit_res["year"], df_lit_res["mpa_equiv"],
            color="#333333", alpha=0.6, s=50, zorder=5, 
            label="Lit. (Sintetizado)", marker="x", linewidths=1.5
        )
        handles_a.append(scatter_a)
        labels_a.append("Lit. (Sintetizado)")

    ax_a.set_title("(a) Degradação da Resistência", pad=15)
    ax_a.set_ylabel("Resistência (MPa)")
    ax_a.set_xlabel("Tempo (anos)")
    ax_a.set_xticks(years[::2])
    
    # Margem extra no final
    ax_a.set_xlim(0, 10.5)
    ax_a.set_ylim(bottom=0)

    # Legenda A (Ordem visual: Otimista -> Ref -> Pessimista)
    ax_a.legend(
        handles_a, labels_a,
        loc="upper center", bbox_to_anchor=(0.5, -0.15),
        ncol=3, frameon=False
    )

    # --- PAINEL B: Capacidade ---
    ax_b = axes[1]
    cols_b = [
        ("capacity_rel_p90", "p90"),
        ("capacity_rel_p95", "p95"),
        ("veg_factor", "veg")
    ]
    
    handles_b, labels_b = plot_panel(
        ax_b, df_cap, "year", cols_b, CAP_CFG,
        "(b) Capacidade e Vegetação", "Capacidade Relativa (%)", is_panel_b=True
    )
    
    # Adicionar Scatter de Literatura se existir (Painel B)
    if df_lit_cap is not None:
        scatter_b = ax_b.scatter(
            df_lit_cap["year"], df_lit_cap["capacity_filled_pct"],
            color="#333333", alpha=0.6, s=50, zorder=5, 
            label="Lit. (Assoreamento)", marker="x", linewidths=1.5
        )
        handles_b.append(scatter_b)
        labels_b.append("Lit. (Assoreamento)")

    ax_b.legend(
        handles_b, labels_b,
        loc="upper center", bbox_to_anchor=(0.5, -0.15),
        ncol=3, frameon=False
    )

    fig.tight_layout(rect=[0, 0.05, 1, 1])

    png_path = out_dir / "painel_resistencia_capacidade.png"
    pdf_path = out_dir / "painel_resistencia_capacidade.pdf"
    
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    
    logger.info(f"Salvo: {png_path}")
    return 0

if __name__ == "__main__":
    main()