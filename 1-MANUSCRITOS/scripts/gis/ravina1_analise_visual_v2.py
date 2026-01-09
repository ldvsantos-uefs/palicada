#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ravina1_analise_visual_v2.py

Versão corrigida do script de análise visual da Ravina 1 (Pseudo-DEM).

Correções principais (v2):
- Centraliza definição das paliçadas em uma estrutura única (PALICADAS).
- Evita propagação de NaN no blur (preenche antes e remasca depois).
- Desenha as paliçadas (diagonais) nos painéis (a) e (b) para checagem visual,
  mantendo o sistema origin="upper".

Observação metodológica:
- A "elevação/profundidade" em cm aqui é uma escala RELATIVA imposta via
  --max-depth-cm. Não é profundidade observada diretamente no GeoTIFF RGB,
  a menos que haja calibração com medidas reais.

Uso:
  python ravina1_analise_visual_v2.py --geotiff "..." --out "..." --layout ab
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from PIL import Image


# Paliçadas na Figura (a), em coordenadas normalizadas do frame (0–1)
# IMPORTANTE: estes valores precisam corresponder ao RECORTE usado.
PALICADAS = [
    # (id, x0, y0, x1, y1, largura_rel)
    (1, 0.1824, 0.3077, 0.3027, 0.2901, 0.035),
    (2, 0.4390, 0.4623, 0.3739, 0.6015, 0.035),
    (3, 0.6796, 0.4656, 0.7469, 0.6068, 0.035),
    (4, 0.7456, 0.6607, 0.6290, 0.7983, 0.040),
]


def pct_to_px(x: float, y: float, w: int, h: int) -> tuple[float, float]:
    # x,y em 0–1 -> pixel (com origin="upper")
    return x * (w - 1), y * (h - 1)


def plot_palicadas_2d(
    ax,
    w: int,
    h: int,
    *,
    color: str = "deepskyblue",
    text_color: str | None = None,
    lw: float = 2.5,
    alpha: float = 0.9,
    annotate: bool = True,
):
    if text_color is None:
        text_color = color
    for pid, x0, y0, x1, y1, _ in PALICADAS:
        X0, Y0 = pct_to_px(x0, y0, w, h)
        X1, Y1 = pct_to_px(x1, y1, w, h)
        ax.plot([X0, X1], [Y0, Y1], color=color, linewidth=lw, alpha=alpha)
        if annotate:
            ax.text(
                (X0 + X1) / 2,
                (Y0 + Y1) / 2,
                f"P{pid}",
                color=text_color,
                fontsize=9,
                ha="center",
                va="center",
                path_effects=[pe.withStroke(linewidth=2, foreground="black")],
            )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ravina 1: Análise Visual (Pseudo-DEM) - v2.")
    p.add_argument("--geotiff", required=True, help="Imagem base (GeoTIFF/PNG) da Ravina 1.")
    p.add_argument("--out", required=True, help="Arquivo PNG de saída.")
    p.add_argument("--title", default="Topografia visual (Pseudo-DEM) - Ravina 1")
    p.add_argument(
        "--layout",
        choices=["stack", "ab"],
        default="ab",
        help="Layout da figura. 'stack' gera empilhamento 3D + mapa 2D. 'ab' gera apenas painéis 2D (A recorte, B elevação).",
    )
    p.add_argument(
        "--debug-grid",
        action="store_true",
        help="Sobrepõe grade de referência em quadrantes (0–100%) para facilitar marcação de coordenadas.",
    )
    p.add_argument(
        "--grid-n",
        type=int,
        default=10,
        help="Número de divisões da grade por eixo quando --debug-grid estiver ativo.",
    )

    # Parâmetros visuais
    p.add_argument("--max-grid", type=int, default=500, help="Resolução do grid de trabalho (ex: 500x500).")
    p.add_argument("--blur-radius", type=float, default=2.0, help="Suavização inicial do Pseudo-DEM.")
    p.add_argument("--layer-gap", type=float, default=20.0, help="Espaçamento vertical entre camadas (stack).")
    p.add_argument("--contour-interval", type=float, default=15.0, help="Intervalo de curvas em cm (escala relativa).")
    p.add_argument(
        "--max-depth-cm",
        type=float,
        default=100.0,
        help="Escala máxima em cm para a topografia visual (representação relativa; requer calibração para ser métrica real).",
    )
    p.add_argument("--base-max-px", type=int, default=3000, help="Limite de pixels para leitura da imagem.")

    return p.parse_args()


def _resample_to(img: np.ndarray, h: int, w: int) -> np.ndarray:
    ih, iw = img.shape[:2]
    yy = np.linspace(0, ih - 1, h).astype(np.int32)
    xx = np.linspace(0, iw - 1, w).astype(np.int32)
    if img.ndim == 2:
        return img[yy][:, xx]
    return img[yy][:, xx, :]


def _gaussian_blur_separable(a: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return a

    radius = int(max(1, round(3 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma))
    k /= k.sum()

    out_x = np.zeros_like(a)
    for i in range(a.shape[0]):
        out_x[i, :] = np.convolve(a[i, :], k, mode="same")

    out = np.zeros_like(a)
    for j in range(a.shape[1]):
        out[:, j] = np.convolve(out_x[:, j], k, mode="same")

    return out


def _hillshade(z: np.ndarray, cellsize: float = 1.0, az_deg: float = 315, alt_deg: float = 45) -> np.ndarray:
    gy, gx = np.gradient(z, cellsize, cellsize)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)

    az = math.radians(az_deg)
    alt = math.radians(alt_deg)

    hs = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    return np.clip(hs, 0, 1)


def _slope_degrees(z: np.ndarray, cellsize: float = 1.0) -> np.ndarray:
    gy, gx = np.gradient(z, cellsize, cellsize)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def ridge_mask(grid: np.ndarray, x0_pct: float, y0_pct: float, x1_pct: float, y1_pct: float, width_pct: float) -> np.ndarray:
    h, w = grid.shape
    p0 = np.array([x0_pct * w, y0_pct * h])
    p1 = np.array([x1_pct * w, y1_pct * h])

    y_idx, x_idx = np.ogrid[:h, :w]
    x_grid = np.broadcast_to(x_idx, (h, w))
    y_grid = np.broadcast_to(y_idx, (h, w))
    coords = np.stack((x_grid, y_grid), axis=-1)

    line_vec = p1 - p0
    line_len_sq = np.sum(line_vec**2)
    if line_len_sq == 0:
        return np.zeros_like(grid)

    t = np.sum((coords - p0) * line_vec, axis=-1) / line_len_sq
    t = np.clip(t, 0, 1)
    nearest = p0 + t[..., np.newaxis] * line_vec
    dist = np.linalg.norm(coords - nearest, axis=-1)

    radius = width_pct * max(h, w)
    denom = 2 * (radius / 2) ** 2
    denom = denom if denom > 0 else 1.0
    mask = np.exp(-dist**2 / denom)
    return np.clip(mask, 0, 1)


def main() -> int:
    args = parse_args()
    geotiff_path = Path(args.geotiff)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not geotiff_path.exists():
        raise FileNotFoundError(f"Imagem não encontrada: {geotiff_path}")

    print(f"Processando: {geotiff_path.name}")

    Image.MAX_IMAGE_PIXELS = None
    im = Image.open(geotiff_path)

    if max(im.size) > args.base_max_px:
        im.thumbnail((args.base_max_px, args.base_max_px))

    im = im.convert("RGBA")
    arr_rgba = np.asarray(im, dtype=np.float32) / 255.0

    base_rgb = np.clip(arr_rgba[:, :, :3], 0, 1)
    base_alpha = np.clip(arr_rgba[:, :, 3], 0, 1)

    H, W = base_alpha.shape
    scale_factor = max(H / args.max_grid, W / args.max_grid, 1.0)
    hh = int(round(H / scale_factor))
    ww = int(round(W / scale_factor))

    print(f"Grid: {ww}x{hh} (Scale: {scale_factor:.2f})")

    base_s = _resample_to(base_rgb, hh, ww)
    alpha_s = _resample_to(base_alpha, hh, ww)

    nodata = alpha_s < 0.1

    lum = 0.299 * base_s[:, :, 0] + 0.587 * base_s[:, :, 1] + 0.114 * base_s[:, :, 2]

    lum_filled = lum.copy()
    if nodata.any():
        mean_val = lum[~nodata].mean() if (~nodata).any() else 0.5
        lum_filled[nodata] = mean_val

    dem = _gaussian_blur_separable(lum_filled, sigma=args.blur_radius)

    dmin, dmax = dem.min(), dem.max()
    if dmax > dmin:
        dem_norm = (dem - dmin) / (dmax - dmin)
    else:
        dem_norm = np.zeros_like(dem)

    dem_norm[nodata] = np.nan

    dem_calc = dem_norm.copy()
    dem_calc[np.isnan(dem_calc)] = 0.5

    hs = _hillshade(dem_calc, cellsize=1.0)
    slope = _slope_degrees(dem_calc * 100, cellsize=1.0)
    slope_norm = (slope - slope.min()) / (slope.max() - slope.min() + 1e-6)

    # --- Layout ---
    if args.layout == "ab":
        fig = plt.figure(figsize=(13, 6.5))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.2])

        ax_a = fig.add_subplot(gs[0])
        ax_a.set_axis_off()
        ax_a.set_title("(a) Recorte ortomosaico", fontsize=12, y=0.95)

        base_rgba = np.dstack([base_s, alpha_s])
        base_rgba[nodata, 3] = 0.0
        ax_a.imshow(base_rgba, origin="upper")
        try:
            ax_a.contour((~nodata).astype(np.float32), levels=[0.5], colors="black", linewidths=0.9, alpha=0.8)
        except Exception:
            pass

        # Desenha paliçadas para checagem visual
        plot_palicadas_2d(ax_a, w=alpha_s.shape[1], h=alpha_s.shape[0], annotate=True)

        if args.debug_grid:
            h, w = alpha_s.shape
            n = max(2, int(args.grid_n))
            for i in range(1, n):
                x = w * i / n
                y = h * i / n
                ax_a.axvline(x, color="white", linewidth=0.6, alpha=0.7)
                ax_a.axhline(y, color="white", linewidth=0.6, alpha=0.7)

            for i in range(0, n + 1):
                x = w * i / n
                y = h * i / n
                px = int(round(100 * i / n))
                ax_a.text(
                    x,
                    0,
                    f"{px}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="white",
                    clip_on=True,
                    path_effects=[pe.withStroke(linewidth=2, foreground="black")],
                )
                ax_a.text(
                    0,
                    y,
                    f"{px}%",
                    ha="left",
                    va="center",
                    fontsize=8,
                    color="white",
                    clip_on=True,
                    path_effects=[pe.withStroke(linewidth=2, foreground="black")],
                )
            ax_a.text(
                0.01,
                0.99,
                "Grade 0–100%",
                transform=ax_a.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                color="white",
                path_effects=[pe.withStroke(linewidth=2, foreground="black")],
            )

    else:
        # Mantém compatibilidade com empilhamento 3D (legado)
        fig = plt.figure(figsize=(14, 7))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.2])

        ax3d = fig.add_subplot(gs[0], projection="3d")
        ax3d.set_axis_off()
        ax3d.view_init(elev=30, azim=-60)
        ax3d.set_title("(a) Camadas de Análise (Stack)", fontsize=12, y=0.95)

        cmap_elev = plt.get_cmap("terrain")
        cmap_slope = plt.get_cmap("magma")
        cmap_hs = plt.get_cmap("gray")

        fc0 = np.dstack([base_s, alpha_s])
        elev_color = cmap_elev(dem_norm)[:, :, :3]
        fc1 = np.dstack([elev_color, alpha_s])
        slope_color = cmap_slope(slope_norm)[:, :, :3]
        fc2 = np.dstack([slope_color, alpha_s])
        hs_color = cmap_hs(hs)[:, :, :3]
        fc3 = np.dstack([hs_color, alpha_s])

        xx, yy = np.meshgrid(np.arange(ww), np.arange(hh))
        z0 = 0.0
        z1 = z0 + args.layer_gap
        z2 = z1 + args.layer_gap
        z3 = z2 + args.layer_gap

        def plot_layer(fc, z, shift_x, shift_y, alpha):
            Z = np.full_like(xx, z)
            ax3d.plot_surface(
                xx + shift_x,
                yy + shift_y,
                Z,
                facecolors=fc,
                rstride=1,
                cstride=1,
                shade=False,
                antialiased=False,
                linewidth=0,
                alpha=alpha,
            )

        plot_layer(fc0, z0, 0, 0, 0.95)
        plot_layer(fc1, z1, 10, -10, 0.90)
        plot_layer(fc2, z2, 20, -20, 0.85)
        plot_layer(fc3, z3, 30, -30, 0.80)

    # --- Painel (b) ---
    ax_b = fig.add_subplot(gs[1])
    ax_b.set_axis_off()
    ax_b.set_title("(b) Mapa de Elevação Relativa (cm)", fontsize=12, y=1.0)

    # --- Efeito "Kriging" (Suavização) ---
    sigma_krig = max(4.0, args.blur_radius * 3.0)

    # Preenche NaNs antes do blur para não propagar NaN
    dem_fill = dem_norm.copy()
    if np.isnan(dem_fill).any():
        mean_dem = np.nanmean(dem_fill)
        if not np.isfinite(mean_dem):
            mean_dem = 0.5
        dem_fill[np.isnan(dem_fill)] = mean_dem

    dem_smooth = _gaussian_blur_separable(dem_fill, sigma=sigma_krig)

    # Reaplica máscara nodata
    dem_smooth[nodata] = np.nan

    # --- Ajuste Manual de Topografia (Paliçadas) ---
    # Paliçadas representam acúmulo máximo de sedimento: forçar topo da escala.
    if args.debug_grid:
        for pid, x0, y0, x1, y1, w_pct in PALICADAS:
            print(
                f"Paliçada {pid} endpoints em %  (x0={x0*100:.1f}, y0={y0*100:.1f})  (x1={x1*100:.1f}, y1={y1*100:.1f})  largura={w_pct*100:.1f}%"
            )

    m_all = np.zeros_like(dem_smooth, dtype=np.float32)
    for pid, x0, y0, x1, y1, w_pct in PALICADAS:
        m = ridge_mask(dem_smooth, x0, y0, x1, y1, w_pct)
        m_all = np.clip(m_all + m, 0, 1)

    target_norm = 1.0
    dem_smooth = dem_smooth * (1 - m_all) + target_norm * m_all
    dem_smooth = np.clip(dem_smooth, 0, 1)

    # Ancora a escala em 0..1 após o forçamento
    finite = np.isfinite(dem_smooth)
    if finite.any():
        vmin = float(np.nanmin(dem_smooth[finite]))
        vmax = float(np.nanmax(dem_smooth[finite]))
        if vmax > vmin:
            dem_smooth = (dem_smooth - vmin) / (vmax - vmin)
            dem_smooth = np.clip(dem_smooth, 0, 1)

    elevation_cm = dem_smooth * args.max_depth_cm

    elev_masked = np.ma.masked_invalid(elevation_cm)
    elev_masked[nodata] = np.ma.masked

    cmap_elev_final = plt.get_cmap("Spectral_r")
    im_plot = ax_b.imshow(elev_masked, cmap=cmap_elev_final, origin="upper", vmin=0, vmax=args.max_depth_cm)

    # Desenha paliçadas (preto) para validação no painel B
    plot_palicadas_2d(
        ax_b,
        w=alpha_s.shape[1],
        h=alpha_s.shape[0],
        annotate=True,
        color="black",
        text_color="white",
        lw=2.0,
        alpha=0.7,
    )

    if args.layout == "ab" and args.debug_grid:
        h, w = alpha_s.shape
        n = max(2, int(args.grid_n))
        for i in range(1, n):
            x = w * i / n
            y = h * i / n
            ax_b.axvline(x, color="black", linewidth=0.5, alpha=0.35)
            ax_b.axhline(y, color="black", linewidth=0.5, alpha=0.35)

        for i in range(0, n + 1):
            x = w * i / n
            y = h * i / n
            px = int(round(100 * i / n))
            ax_b.text(
                x,
                0,
                f"{px}%",
                ha="center",
                va="bottom",
                fontsize=7,
                color="black",
                alpha=0.75,
                clip_on=True,
                path_effects=[pe.withStroke(linewidth=2, foreground="white")],
            )
            ax_b.text(
                0,
                y,
                f"{px}%",
                ha="left",
                va="center",
                fontsize=7,
                color="black",
                alpha=0.75,
                clip_on=True,
                path_effects=[pe.withStroke(linewidth=2, foreground="white")],
            )

    if elev_masked.count() > 0:
        levels_elev = np.arange(0, args.max_depth_cm + 1, args.contour_interval)
        cs = ax_b.contour(elev_masked, levels=levels_elev, colors="black", linewidths=0.6, alpha=0.6)
        ax_b.clabel(cs, inline=True, fontsize=8, fmt="%d", colors="black")

    cbar = plt.colorbar(im_plot, ax=ax_b, fraction=0.046, pad=0.04, shrink=0.7)
    cbar.set_label("Elevação relativa (escala imposta) (cm)", rotation=270, labelpad=15)

    plt.tight_layout()

    print(f"Salvando figura em: {out_path}")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
