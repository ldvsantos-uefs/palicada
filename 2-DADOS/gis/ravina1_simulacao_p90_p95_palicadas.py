from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import rasterio
except Exception:  # pragma: no cover
    rasterio = None

def vegetation_factor(year: np.ndarray, k: float = 0.7, t50: float = 2.0) -> np.ndarray:
    """Fator [0..1] de reforço por vegetação (logística)."""
    year = np.asarray(year, dtype=float)
    return 1.0 / (1.0 + np.exp(-k * (year - t50)))


def bamboo_strength_factor(year: np.ndarray, k_year: float) -> np.ndarray:
    """Fator [0..1] de degradação do bambu (exp)."""
    year = np.asarray(year, dtype=float)
    return np.exp(-k_year * year)


def simulate_capacity(years: np.ndarray) -> pd.DataFrame:
    """Capacidade relativa ao longo do tempo.

    - A estrutura perde capacidade pela degradação do bambu
    - Ganha capacidade por ancoragem/coesão via enraizamento (vegetação)
    """

    years = np.asarray(years, dtype=float)

    veg = vegetation_factor(years)

    # p90/p95 como cenários de maior solicitação (mais severos) => maior k
    k_p90 = 0.08
    k_p95 = 0.11

    sf_p90 = bamboo_strength_factor(years, k_year=k_p90)
    sf_p95 = bamboo_strength_factor(years, k_year=k_p95)

    # Combinação simples: capacidade_relativa = (bambu) * (1 + alpha*veg)
    alpha = 0.6
    cap_p90 = sf_p90 * (1.0 + alpha * veg)
    cap_p95 = sf_p95 * (1.0 + alpha * veg)

    return pd.DataFrame(
        {
            "year": years.astype(int),
            "veg_factor": veg,
            "bamboo_factor_p90": sf_p90,
            "bamboo_factor_p95": sf_p95,
            "capacity_rel_p90": cap_p90,
            "capacity_rel_p95": cap_p95,
        }
    )


def plot_capacity(df: pd.DataFrame, out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(df["year"], df["capacity_rel_p90"], label="Capacidade relativa (p90)")
    plt.plot(df["year"], df["capacity_rel_p95"], label="Capacidade relativa (p95)")
    plt.plot(df["year"], df["veg_factor"], "--", label="Fator de vegetação")
    plt.xlabel("Ano")
    plt.ylabel("Fator relativo")
    plt.title("Evolução temporal: degradação do bambu vs. reforço por vegetação")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def try_plot_raster_background(raster_path: Path, out_png: Path) -> None:
    if rasterio is None:
        return
    if not raster_path.exists():
        return

    out_png.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(raster_path) as src:
        arr = src.read(1, masked=True)

    plt.figure(figsize=(8, 6))
    plt.imshow(arr, cmap="gray")
    plt.title("Raster de referência (RAVINA_1_RECORTE)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

# Exemplo de execução
if __name__ == '__main__':
    repo_root = Path(__file__).resolve().parents[1]

    years = np.arange(0, 11)
    df = simulate_capacity(years)

    out_dir = repo_root / "2-DADOS" / "simulacoes"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "capacidade_temporal.csv"
    df.to_csv(out_csv, index=False)
    print(f"✓ Salvo: {out_csv}")

    figs_dir = repo_root / "1-MANUSCRITOS" / "1-CONTROLE_PLITOSSOLO" / "media" / "figuras"
    plot_capacity(df, figs_dir / "capacidade_temporal.png")
    print(f"✓ Figura salva: {figs_dir / 'capacidade_temporal.png'}")

    raster_path = repo_root / "2-DADOS" / "DADOS" / "RAVINA_1_RECORTE.tif"
    try_plot_raster_background(raster_path, figs_dir / "ravina1_raster_referencia.png")