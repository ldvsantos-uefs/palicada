#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ravina1_full_analise.py

Script adaptado para processar a imagem completa da Ravina 1 (RAVINA 1.tif).
Mantém a metodologia de Pseudo-DEM e visualização 3D, mas remove as cristas manuais
pois as coordenadas mudaram.

Funcionalidades:
- Suporta imagens grandes (resampling).
- Gera visualização 3D empilhada.
- Gera mapa de elevação relativa (sem cristas manuais).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import matplotlib

# Configura backend para não precisar de X11/Display
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from PIL import Image, ImageFilter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ravina 1 Full: Análise Visual 3D.")
    p.add_argument("--geotiff", required=True, help="Imagem base (GeoTIFF/PNG).")
    p.add_argument("--out", required=True, help="Arquivo PNG de saída.")
    p.add_argument("--title", default="Expressão SIG por camadas (Ravina 1 - Completa)")
    
    # Parâmetros visuais
    p.add_argument("--max-grid", type=int, default=800, help="Resolução do grid 3D (aumentado para imagem full).")
    p.add_argument("--blur-radius", type=float, default=3.0, help="Suavização do Pseudo-DEM.")
    p.add_argument("--layer-gap", type=float, default=20.0, help="Espaçamento vertical entre camadas.")
    p.add_argument("--contour-interval", type=float, default=25.0, help="Intervalo de curvas em cm.")
    p.add_argument("--max-depth-cm", type=float, default=300.0, help="Profundidade máxima estimada em cm.")
    p.add_argument("--base-max-px", type=int, default=4000, help="Limite de pixels para leitura da imagem.")
    
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
        print(f"Redimensionando de {im.size} para max {args.base_max_px}px...")
        im.thumbnail((args.base_max_px, args.base_max_px))
    
    im = im.convert("RGBA")
    arr_rgba = np.asarray(im, dtype=np.float32) / 255.0
    
    base_rgb = np.clip(arr_rgba[:, :, :3], 0, 1)
    base_alpha = np.clip(arr_rgba[:, :, 3], 0, 1)
    
    H, W = base_alpha.shape
    scale_factor = max(H / args.max_grid, W / args.max_grid, 1.0)
    hh = int(round(H / scale_factor))
    ww = int(round(W / scale_factor))
    
    print(f"Grid 3D: {ww}x{hh} (Scale: {scale_factor:.2f})")

    base_s = _resample_to(base_rgb, hh, ww)
    alpha_s = _resample_to(base_alpha, hh, ww)
    nodata = alpha_s < 0.1

    # Pseudo-DEM: Luminância invertida (áreas escuras = fundo)
    lum = 0.299 * base_s[:, :, 0] + 0.587 * base_s[:, :, 1] + 0.114 * base_s[:, :, 2]
    dem_raw = 1.0 - lum
    dem_raw[nodata] = 0
    
    # Suavização
    dem_smooth = _gaussian_blur_separable(dem_raw, sigma=args.blur_radius)
    dem_norm = (dem_smooth - dem_smooth.min()) / (dem_smooth.max() - dem_smooth.min() + 1e-6)
    
    # --- Visualização ---
    fig = plt.figure(figsize=(12, 10))
    
    # Painel A: 3D Stack
    ax_3d = fig.add_subplot(1, 2, 1, projection='3d')
    ax_3d.set_axis_off()
    ax_3d.set_title("(a) Integração de Camadas", fontsize=12, y=1.05)
    
    xx, yy = np.meshgrid(np.arange(ww), np.arange(hh))
    
    # Camada 1: Ortomosaico (Base)
    z_base = np.zeros_like(dem_norm)
    stride = max(1, int(min(hh, ww) / 100))
    ax_3d.plot_surface(xx, yy, z_base, rstride=stride, cstride=stride,
                       facecolors=base_s, shade=False, alpha=0.9)
    
    # Camada 2: Modelo Digital (Meio)
    z_dem = z_base + args.layer_gap
    hs = _hillshade(dem_norm * 50, az_deg=315, alt_deg=45)
    hs_rgb = np.dstack([hs, hs, hs])
    # Blend com colormap terrain
    cmap_dem = plt.get_cmap("terrain")
    dem_color = cmap_dem(dem_norm)[:, :, :3]
    blend = 0.6 * dem_color + 0.4 * hs_rgb
    ax_3d.plot_surface(xx, yy, z_dem, rstride=stride, cstride=stride,
                       facecolors=blend, shade=False, alpha=0.8)
    
    # Camada 3: Declividade (Topo)
    z_slope = z_dem + args.layer_gap
    slp = _slope_degrees(dem_norm * 100)
    slp_norm = np.clip(slp / 45.0, 0, 1)
    cmap_slope = plt.get_cmap("RdYlGn_r")
    slope_color = cmap_slope(slp_norm)[:, :, :3]
    ax_3d.plot_surface(xx, yy, z_slope, rstride=stride, cstride=stride,
                       facecolors=slope_color, shade=False, alpha=0.7)
    
    # Ajuste de câmera
    ax_3d.view_init(elev=35, azim=-60)
    ax_3d.set_zlim(0, z_slope.max() + 10)
    
    # Painel B: Mapa de Elevação Relativa
    ax_b = fig.add_subplot(1, 2, 2)
    ax_b.set_axis_off()
    ax_b.set_title("(b) Mapa de Elevação Relativa (cm)", fontsize=12, y=1.0)
    
    # Suavização extra para contornos
    sigma_krig = max(4.0, args.blur_radius * 3.0)
    dem_final = _gaussian_blur_separable(dem_norm, sigma=sigma_krig)
    
    # NOTA: Cristas manuais removidas pois as coordenadas mudaram na imagem completa.
    
    dem_final = np.clip(dem_final, 0, 1)
    elevation_cm = dem_final * args.max_depth_cm
    elev_masked = np.ma.masked_invalid(elevation_cm)
    elev_masked[nodata] = np.ma.masked
    
    # Colormap invertido (Spectral_r)
    cmap_elev_final = plt.get_cmap("Spectral_r")
    
    im_plot = ax_b.imshow(elev_masked, cmap=cmap_elev_final, origin="upper", vmin=0, vmax=args.max_depth_cm)
    
    if elev_masked.count() > 0:
        levels_elev = np.arange(0, args.max_depth_cm + 1, args.contour_interval)
        cs = ax_b.contour(elev_masked, levels=levels_elev, colors="black", linewidths=0.6, alpha=0.6)
        ax_b.clabel(cs, inline=True, fontsize=8, fmt="%d", colors="black")

    cbar = plt.colorbar(im_plot, ax=ax_b, fraction=0.046, pad=0.04, shrink=0.7)
    cbar.set_label("Elevação Relativa / Sedimentos (cm)", rotation=270, labelpad=15)
    
    plt.tight_layout()
    print(f"Salvando figura em: {out_path}")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
