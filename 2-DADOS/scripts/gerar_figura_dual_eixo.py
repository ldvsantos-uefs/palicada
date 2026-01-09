import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.font_manager as fm
from pathlib import Path
import logging

# Configurar backend e logging
matplotlib.use("Agg")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURAÇÕES DE ESTILO
# =============================================================================

def configure_style():
    """Configura estilo unificado"""
    plt.rcdefaults()
    plt.style.use('ggplot')
    
    font_name = "DejaVu Sans"
    try:
        if any("Poppins" in f.name for f in fm.fontManager.ttflist):
            font_name = "Poppins"
    except:
        pass

    plt.rcParams.update({
        "font.family": font_name,
        "figure.dpi": 300,
        "axes.labelsize": 12,
        "axes.labelweight": "bold",
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        # "axes.facecolor": "white",  <-- REMOVIDO PARA MANTER O CINZA DO GGPLOT
        # "grid.color": "#E5E5E5",    <-- REMOVIDO
        "grid.alpha": 0.6,
        "grid.linestyle": "--",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.0,
    })
    return font_name

# Cores
COLORS = {
    "lit_res":    "steelblue",    
    "p90":        "orangered",    
    "p95":        "gold",         
    "veg":        "darkolivegreen" 
}

def main():
    font_name = configure_style()
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "1-MANUSCRITOS" / "1-CONTROLE_PLITOSSOLO" / "media" / "figuras"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Caminho da imagem (A)
    # Ajustar para o caminho correto relativo ao repo_root
    img_path = repo_root / "1-MANUSCRITOS" / "1-CONTROLE_PLITOSSOLO" / "media" / "figuras_nao_usadas" / "image13.jpg"

    # Carregar Dados CSV
    try:
        df_lit = pd.read_csv(repo_root / "2-DADOS" / "simulacoes" / "dados_literatura_resistencia.csv")
        df_sim = pd.read_csv(repo_root / "2-DADOS" / "simulacoes" / "capacidade_temporal.csv")
        
        INITIAL_MPA = 180.0
        df_lit["mpa_calc"] = (df_lit["integrity_pct"] / 100.0) * INITIAL_MPA
        
        # Carregar Imagem
        img = None
        if img_path.exists():
            img = mpimg.imread(str(img_path))
        else:
            logger.warning(f"Imagem não encontrada em: {img_path}")
        
    except Exception as e:
        logger.error(f"Erro ao ler recursos: {e}")
        return

    # Criar Figura (Painel Horizontal: Lado a Lado)
    # Aumentando largura para acomodar layout horizontal
    fig = plt.figure(figsize=(16, 7))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.8, 1.2], wspace=0.25)

    # --- PAINE A: FOTO ---
    ax_img = fig.add_subplot(gs[0])
    if img is not None:
        ax_img.imshow(img)
    else:
        ax_img.text(0.5, 0.5, "Imagem Indisponível", ha='center', va='center')
        
    ax_img.set_title("(a) Estabelecimento da Barreira", loc='left', pad=10, fontsize=12)
    ax_img.axis('off')

    # --- PAINEL B: GRÁFICO ---
    ax1 = fig.add_subplot(gs[1])

    # Plotar Eixo 1 (Resistência)
    color_ax1 = COLORS["lit_res"]
    line1, = ax1.plot(
        df_lit["year"], df_lit["mpa_calc"],
        '--o', color=color_ax1,
        linewidth=2.5, markersize=8, label=f"Resistência do Bambu (Lit.)",
        alpha=0.8, markeredgecolor='navy', markeredgewidth=0.8, dashes=(5, 3)
    )

    ax1.set_xlabel('Tempo (anos)', fontweight='bold')
    ax1.set_ylabel('Resistência do Bambu [MPa]', fontweight='bold', color=color_ax1)
    ax1.tick_params(axis='y', labelcolor=color_ax1)
    ax1.set_ylim(0, 200)
    ax1.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
    
    # Formatar Eixo X
    years_ticks = [y for y in df_lit["year"].unique() if y % 2 == 0]
    labels_ticks = [f"{int(y)} ano" if y <= 1 else f"{int(y)} anos" for y in years_ticks]
    ax1.set_xticks(years_ticks)
    ax1.set_xticklabels(labels_ticks)

    # Plotar Eixo 2 (Capacidade)
    ax2 = ax1.twinx()
    
    line2, = ax2.plot(
        df_sim["year"], df_sim["capacity_rel_p90"] * 100,
        '-s', color=COLORS["p90"], label="Preenchimento Paliçadas (P90)",
        linewidth=2.5, markersize=7, alpha=0.85, markeredgecolor='black', markeredgewidth=0.5
    )
    line3, = ax2.plot(
        df_sim["year"], df_sim["capacity_rel_p95"] * 100,
        '-^', color=COLORS["p95"], label="Preenchimento Paliçadas (P95)",
        linewidth=2.5, markersize=7, alpha=0.85, markeredgecolor='black', markeredgewidth=0.5
    )
    line4, = ax2.plot(
        df_sim["year"], df_sim["veg_factor"] * 100,
        '-D', color=COLORS["veg"], label="Cobertura Vegetal",
        linewidth=2.5, markersize=7, alpha=0.85, markeredgecolor='black', markeredgewidth=0.5
    )
    
    color_ax2 = "#444444"
    ax2.set_ylabel('Taxa de Preenchimento / Cobertura [%]', fontweight='bold', color=color_ax2)
    ax2.tick_params(axis='y', labelcolor=color_ax2)
    ax2.set_ylim(0, 120)

    # Título do Painel B
    ax1.set_title("(b) Resistência e Capacidade", loc='left', pad=10, fontsize=12)

    # Legenda (Dentro, canto superior direito)
    lines = [line1, line2, line3, line4]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', ncol=1,
               framealpha=0.95, edgecolor='black', fancybox=True, shadow=True)

    # Layout Ajustado
    plt.tight_layout()

    # Salvar
    out_path = out_dir / "painel_integrado_foto_grafico.png"
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    logger.info(f"Salvo: {out_path}")

if __name__ == "__main__":
    main()