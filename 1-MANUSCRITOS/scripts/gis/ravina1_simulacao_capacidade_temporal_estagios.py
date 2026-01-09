#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ravina1_simulacao_capacidade_temporal_estagios.py

Gera mapas de evolução temporal do preenchimento (25/50/75/100%) "do zero",
reaproveitando exatamente o mesmo pipeline de base (pseudo-DEM + suavização tipo
"krig" + colormap contínuo + contornos + paliçadas) usado nas figuras aceitas.

O mapa é um campo contínuo (elevação relativa em cm, escala imposta) e as
etiquetas nas paliçadas mostram o tempo (anos) até atingir o estágio.

Saída: 4 PNGs separados (25%, 50%, 75% e 100%) para montagem posterior.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from PIL import Image

# Reuso explícito do pipeline de dados (chuva/sedimento) do script canônico.
# Mantém consistência sem "inventar" parâmetros.
from ravina1_simulacao_p90_p95_capacity_em_anos import (
    _fit_sediment_ratio_by_area,
    _runoff_coeff_from_vib,
    _stat_monthly_totals,
    _fmt_years,
)


# (id, x0, y0, x1, y1, width_pct)
PALICADAS = [
    (1, 0.1824, 0.3077, 0.3027, 0.2901, 0.035),
    (2, 0.4390, 0.4623, 0.3739, 0.6015, 0.035),
    (3, 0.6796, 0.4656, 0.7469, 0.6068, 0.035),
    (4, 0.7456, 0.6607, 0.6290, 0.7983, 0.040),
]


@dataclass(frozen=True)
class PalicadaCfg:
    id: int
    area_group: str
    crest_height_cm: float
    current_fill_cm: float
    deposit_length_pct: float | None


DEFAULT_PRECIP_SERIES = Path("2-DADOS/CLIMATOLOGIA_20ANOS/dados/serie_precipitacao_20anos.csv")
DEFAULT_SEDIMENT_DATA = Path("2-DADOS/CLIMATOLOGIA_20ANOS/dados/dados_integrados_sedimentacao.csv")


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
    """Gera sombreamento de relevo (hillshade) para dar textura 3D."""
    gy, gx = np.gradient(z, cellsize, cellsize)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)

    az = math.radians(az_deg)
    alt = math.radians(alt_deg)

    hs = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    return np.clip(hs, 0, 1)


def pct_to_px(x: float, y: float, w: int, h: int) -> tuple[float, float]:
    return x * (w - 1), y * (h - 1)


def _load_palicadas_config(path: Path) -> list[PalicadaCfg]:
    with path.open("r", encoding="utf-8") as f:
        obj: Mapping[str, Any] = json.load(f)

    pals = obj.get("palicadas")
    if not isinstance(pals, list) or not pals:
        raise ValueError("JSON inválido: esperado campo 'palicadas' como lista não vazia")

    cfgs: list[PalicadaCfg] = []
    for p_raw in pals:
        if not isinstance(p_raw, dict):
            raise ValueError("JSON inválido: cada item em 'palicadas' deve ser um objeto")
        p: Mapping[str, Any] = p_raw

        dep_len = p.get("deposit_length_pct")
        cfgs.append(
            PalicadaCfg(
                id=int(p["id"]),
                area_group=str(p["area_group"]),
                crest_height_cm=float(p["crest_height_cm"]),
                current_fill_cm=float(p.get("current_fill_cm", 0.0)),
                deposit_length_pct=(None if dep_len in (None, "") else float(dep_len)),
            )
        )

    return cfgs


def _deposit_mask(*, shape_hw: tuple[int, int], pal: tuple[int, float, float, float, float, float], length_pct: float) -> np.ndarray:
    """Máscara de deposição em forma de cunha.

    Produz uma função peso (0..1) com:
    - decaimento longitudinal a montante
    - suavização transversal com gaussiana

    Nota: o sentido "a montante" é aproximado e consistente com os scripts GIS.
    """

    h, w = shape_hw
    _, x0, y0, x1, y1, width_pct = pal

    p0 = np.array([x0 * (w - 1), y0 * (h - 1)], dtype=np.float32)
    p1 = np.array([x1 * (w - 1), y1 * (h - 1)], dtype=np.float32)

    v = p1 - p0
    v_norm = float(np.linalg.norm(v))
    if v_norm <= 1e-6:
        return np.zeros((h, w), dtype=np.float32)

    v_unit = v / v_norm
    n_unit = np.array([-v_unit[1], v_unit[0]], dtype=np.float32)

    c = 0.5 * (p0 + p1)
    test = c + np.array([0.0, -1.0], dtype=np.float32)

    def side(pt: np.ndarray) -> float:
        return float((p1[0] - p0[0]) * (pt[1] - p0[1]) - (p1[1] - p0[1]) * (pt[0] - p0[0]))

    upstream_sign = 1.0 if side(test) >= 0 else -1.0

    yy, xx = np.mgrid[0:h, 0:w]
    P = np.stack([xx.astype(np.float32), yy.astype(np.float32)], axis=-1)

    signed_dist = (P[..., 0] - p0[0]) * n_unit[0] + (P[..., 1] - p0[1]) * n_unit[1]
    is_upstream = (signed_dist * upstream_sign) >= 0

    dist_perp = np.abs(signed_dist)
    width_px = float(width_pct * max(h, w))

    # eixo "a montante" aproximado para cima na imagem
    up_axis = np.array([0.0, -1.0], dtype=np.float32)
    along_up = (P[..., 0] - c[0]) * up_axis[0] + (P[..., 1] - c[1]) * up_axis[1]

    length_px = float(length_pct * max(h, w))
    within_len = (along_up >= 0) & (along_up <= length_px)

    if length_px <= 1e-6:
        along_w = np.zeros((h, w), dtype=np.float32)
    else:
        along_w = 1.0 - np.clip(along_up / length_px, 0.0, 1.0)

    sigma = max(1.0, width_px)
    cross_w = np.exp(-(dist_perp * dist_perp) / (2.0 * sigma * sigma)).astype(np.float32)

    wgt = (along_w.astype(np.float32) * cross_w).astype(np.float32)
    wgt[~(is_upstream & within_len)] = 0.0
    return np.clip(wgt, 0.0, 1.0)


def _apply_deposition_stage(
    base_elev_cm: np.ndarray,
    nodata: np.ndarray,
    pal_cfgs: list[PalicadaCfg],
    *,
    stage_frac: float,
    sed_inc_by_pid_cm: dict[int, float | None],
    default_length_pct: float,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    out = base_elev_cm.copy()
    metrics: list[dict[str, object]] = []

    h, w = base_elev_cm.shape
    cfg_by_id = {c.id: c for c in pal_cfgs}

    for pal in PALICADAS:
        pid = pal[0]
        cfg = cfg_by_id.get(pid)
        if cfg is None:
            continue

        remaining = max(0.0, float(cfg.crest_height_cm) - float(cfg.current_fill_cm))
        target_fill_cm = float(cfg.current_fill_cm) + float(stage_frac) * float(remaining)

        sed_inc = sed_inc_by_pid_cm.get(pid)
        years_to_stage: float | None
        if sed_inc is None or not np.isfinite(float(sed_inc)) or float(sed_inc) <= 0:
            years_to_stage = None
        else:
            months = (float(stage_frac) * float(remaining) / float(sed_inc)) if remaining > 0 else 0.0
            years_to_stage = float(months) / 12.0

        length_pct = cfg.deposit_length_pct if cfg.deposit_length_pct is not None else float(default_length_pct)
        wgt = _deposit_mask(shape_hw=(h, w), pal=pal, length_pct=float(length_pct))
        wgt[nodata] = 0.0

        deposit_area_px = int(np.count_nonzero(wgt > 0))
        deposit_mean_cm: float | None = None
        if target_fill_cm > 0 and deposit_area_px > 0:
            delta = (wgt * float(target_fill_cm)).astype(np.float32)
            out = out + delta
            deposit_mean_cm = float(delta[wgt > 0].mean())

        metrics.append(
            {
                "palicada": f"P{pid}",
                "area_group": cfg.area_group,
                "crest_height_cm": cfg.crest_height_cm,
                "current_fill_cm": cfg.current_fill_cm,
                "remaining_cm": remaining,
                "target_fill_cm": target_fill_cm,
                "sed_inc_cm": (None if sed_inc is None else float(sed_inc)),
                # compatível com o renderizador do script canônico
                "years_to_fill": years_to_stage,
                "deposit_mean_cm": deposit_mean_cm,
                "deposit_area_px": deposit_area_px,
                "deposit_length_pct": float(length_pct),
            }
        )

    return out, metrics


def _render_stage(
    *,
    out_path: Path,
    elev_cm: np.ndarray,
    metrics: list[dict[str, object]],
    nodata: np.ndarray,
    hh: int,
    ww: int,
    stage_frac: float,
    medio_mm: float,
    vmax_plot: float,
    contour_interval: float,
    dpi: int,
) -> None:
    # Mapa de cores contínuo (Spectral_r) para representar a elevação/preenchimento
    cmap = plt.get_cmap("Spectral_r")

    fig = plt.figure(figsize=(6.6, 6.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.06], wspace=0.0, hspace=0.10)
    ax = fig.add_subplot(gs[0, 0])
    ax.set_axis_off()

    # Gera Hillshade para textura (baseado na elevação atual do estágio)
    # Normaliza elevação para hillshade (exagero vertical ajuda visualização)
    z_for_hs = elev_cm.copy()
    z_for_hs[nodata] = np.nanmean(z_for_hs[~nodata]) if (~nodata).any() else 0.0
    hs = _hillshade(z_for_hs, cellsize=1.0, az_deg=315, alt_deg=45)

    # Composição: Cor (Elevação) modulada pelo Hillshade (Intensidade)
    elev_masked = np.ma.masked_invalid(elev_cm)
    elev_masked[nodata] = np.ma.masked

    # 1. Plota Hillshade em escala de cinza como base (textura)
    hs_masked = np.ma.masked_where(nodata, hs)
    ax.imshow(hs_masked, cmap="gray", origin="upper", vmin=0, vmax=1, alpha=1.0)

    # 2. Plota Elevação (Cor) com transparência para fundir com hillshade
    # Usamos alpha=0.65 para deixar o hillshade aparecer por baixo
    im = ax.imshow(elev_masked, cmap=cmap, origin="upper", vmin=0, vmax=float(vmax_plot), alpha=0.65)

    if elev_masked.count() > 0 and contour_interval > 0:
        # Contornos baseados na elevação total
        levels = np.arange(0, float(vmax_plot) + 1, float(contour_interval))
        levels = levels[levels > 0]  # remove nível 0 se desejar
        cs = ax.contour(elev_masked, levels=levels, colors="black", linewidths=0.55, alpha=0.6)
        ax.clabel(cs, inline=True, fontsize=8, fmt="%d", colors="black")

    # Paliçadas + etiqueta em ANOS (tempo até atingir o estágio)
    for pid, x0, y0, x1, y1, _ in PALICADAS:
        X0, Y0 = pct_to_px(x0, y0, ww, hh)
        X1, Y1 = pct_to_px(x1, y1, ww, hh)
        ax.plot([X0, X1], [Y0, Y1], color="black", linewidth=2.0, alpha=0.75)

        m = next((mm for mm in metrics if mm.get("palicada") == f"P{pid}"), None)
        yrs: float | None
        try:
            yrs = float(m.get("years_to_fill")) if m is not None and m.get("years_to_fill") is not None else None
        except Exception:
            yrs = None

        yrs_s = _fmt_years(yrs)
        label = f"{yrs_s} anos" if yrs_s != "—" else "—"
        ax.text(
            (X0 + X1) / 2,
            (Y0 + Y1) / 2,
            label,
            color="white",
            fontsize=16,
            ha="center",
            va="center",
            path_effects=[pe.withStroke(linewidth=4.0, foreground="black")],
        )

    pct = int(round(100 * float(stage_frac)))
    ax.text(
        0.02,
        0.02,
        f"Médio (mediana) mensal = {medio_mm:.1f} mm\nestágio={pct}%\n(rótulos em anos)",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="black",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75, edgecolor="none"),
    )

    ax_cb = fig.add_subplot(gs[1, 0])
    norm = mcolors.Normalize(vmin=0.0, vmax=float(vmax_plot))
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, cax=ax_cb, orientation="horizontal")
    cbar.set_label("Elevação Relativa Total (Base + Sedimento) [cm]")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.10, wspace=0.0, hspace=0.10)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ravina 1: simulação espacial temporal por estágios (25/50/75/100%).")
    p.add_argument("--geotiff", required=True, help="Imagem base do recorte (GeoTIFF/PNG com alpha).")
    p.add_argument("--palicadas-config", required=True, help="JSON com alturas úteis e parâmetros por paliçada.")

    p.add_argument(
        "--precip-series",
        default=str(DEFAULT_PRECIP_SERIES),
        help="CSV diário com Data e Precipitacao_mm (default: série 20 anos).",
    )
    p.add_argument(
        "--sediment-data",
        default=str(DEFAULT_SEDIMENT_DATA),
        help="CSV integrado com AREA, RAINFALL e FRACIONADO (default: dados integrados).",
    )
    p.add_argument("--fracionado-unit", choices=["m", "cm"], default="m")
    p.add_argument(
        "--medio-stat",
        choices=["median", "mean"],
        default="median",
        help="Como definir o cenário médio a partir dos totais mensais (default: median).",
    )
    p.add_argument(
        "--runoff-coeff",
        type=float,
        default=None,
        help="Override do coeficiente de escoamento (0–1). Se omitido, usa defaults do JSON e/ou regra VIB→runoff.",
    )

    p.add_argument("--out-dir", required=True, help="Diretório de saída para as 4 figuras.")
    p.add_argument("--prefix", default="fig_espacial_temporal_capacidade_", help="Prefixo dos arquivos de saída.")

    p.add_argument("--max-grid", type=int, default=500)
    p.add_argument("--blur-radius", type=float, default=2.0)
    p.add_argument("--contour-interval", type=float, default=15.0)
    p.add_argument("--max-depth-cm", type=float, default=100.0)
    p.add_argument("--base-max-px", type=int, default=3000, help="Reduz a imagem para acelerar (maior dimensão em px).")
    p.add_argument("--deposit-length-pct", type=float, default=0.12)
    p.add_argument("--dpi", type=int, default=300)

    return p.parse_args()


def main() -> int:
    args = parse_args()

    geotiff_path = Path(args.geotiff)
    cfg_path = Path(args.palicadas_config)
    out_dir = Path(args.out_dir)

    if not geotiff_path.exists():
        raise FileNotFoundError(f"Imagem não encontrada: {geotiff_path}")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config não encontrada: {cfg_path}")

    pal_cfgs = _load_palicadas_config(cfg_path)

    # Ler VIB e defaults do JSON (mesma lógica do script canônico)
    vib_by_area: dict[str, float] = {}
    cfg_obj: dict[str, Any] = {}
    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            cfg_obj = loaded
            vib_raw = cfg_obj.get("vib_cm_h_by_area_group")
            if isinstance(vib_raw, dict):
                for k, v in vib_raw.items():
                    try:
                        vib_by_area[str(k).strip().upper()] = float(v)
                    except Exception:
                        continue
    except Exception:
        vib_by_area = {}
        cfg_obj = {}

    runoff_coeff_default = 0.6
    defaults = cfg_obj.get("defaults")
    if isinstance(defaults, dict):
        try:
            runoff_coeff_default = float(defaults.get("runoff_coeff", runoff_coeff_default))
        except Exception:
            pass
    runoff_coeff_default = float(args.runoff_coeff) if args.runoff_coeff is not None else float(runoff_coeff_default)

    runoff_coeff_ref_for_sed = float(runoff_coeff_default)
    model_obj = cfg_obj.get("vib_to_runoff_model")
    if isinstance(model_obj, dict):
        try:
            runoff_coeff_ref_for_sed = float(model_obj.get("coeff_at_ref", runoff_coeff_ref_for_sed))
        except Exception:
            pass

    precip_series = Path(args.precip_series)
    sediment_data = Path(args.sediment_data)
    if not precip_series.exists():
        raise FileNotFoundError(f"precip-series não encontrado: {precip_series}")
    if not sediment_data.exists():
        raise FileNotFoundError(f"sediment-data não encontrado: {sediment_data}")

    medio_mm = _stat_monthly_totals(precip_series, stat=str(args.medio_stat))
    ratio_by_area = _fit_sediment_ratio_by_area(sediment_data, fracionado_unit=str(args.fracionado_unit))

    sed_medio_by_pid: dict[int, float | None] = {}
    for cfg in pal_cfgs:
        ratio = ratio_by_area.get(str(cfg.area_group).strip().upper())
        if ratio is None:
            sed_medio_by_pid[cfg.id] = None
            continue

        sed_medio_base = float(max(0.0, float(ratio) * float(medio_mm)))

        vib_cm_h = vib_by_area.get(str(cfg.area_group).strip().upper())
        if args.runoff_coeff is not None:
            runoff_coeff_used = float(args.runoff_coeff)
        else:
            runoff_coeff_used, _, _ = _runoff_coeff_from_vib(
                vib_cm_h,
                cfg_obj=cfg_obj,
                fallback_coeff=runoff_coeff_default,
            )

        sed_scale = 1.0
        if runoff_coeff_ref_for_sed > 0:
            sed_scale = float(runoff_coeff_used) / float(runoff_coeff_ref_for_sed)

        sed_medio_by_pid[cfg.id] = float(max(0.0, float(sed_medio_base) * float(sed_scale)))

    # Pseudo-DEM a partir do recorte (mesmo pipeline do script canônico)
    Image.MAX_IMAGE_PIXELS = None
    im = Image.open(geotiff_path)
    if max(im.size) > int(args.base_max_px):
        im.thumbnail((int(args.base_max_px), int(args.base_max_px)))

    im = im.convert("RGBA")
    arr_rgba = np.asarray(im, dtype=np.float32) / 255.0
    base_rgb = np.clip(arr_rgba[:, :, :3], 0, 1)
    base_alpha = np.clip(arr_rgba[:, :, 3], 0, 1)

    H, W = base_alpha.shape
    scale_factor = max(H / int(args.max_grid), W / int(args.max_grid), 1.0)
    hh = int(round(H / scale_factor))
    ww = int(round(W / scale_factor))

    base_s = _resample_to(base_rgb, hh, ww)
    alpha_s = _resample_to(base_alpha, hh, ww)
    nodata = alpha_s < 0.1

    lum = 0.299 * base_s[:, :, 0] + 0.587 * base_s[:, :, 1] + 0.114 * base_s[:, :, 2]

    lum_filled = lum.copy()
    if nodata.any():
        mean_val = lum[~nodata].mean() if (~nodata).any() else 0.5
        lum_filled[nodata] = mean_val

    dem = _gaussian_blur_separable(lum_filled, sigma=float(args.blur_radius))
    dmin, dmax = float(dem.min()), float(dem.max())
    if dmax > dmin:
        dem_norm = (dem - dmin) / (dmax - dmin)
    else:
        dem_norm = np.zeros_like(dem)

    dem_norm[nodata] = np.nan

    sigma_krig = max(4.0, float(args.blur_radius) * 3.0)
    dem_fill = dem_norm.copy()
    if np.isnan(dem_fill).any():
        mean_dem = float(np.nanmean(dem_fill))
        if not np.isfinite(mean_dem):
            mean_dem = 0.5
        dem_fill[np.isnan(dem_fill)] = mean_dem

    dem_smooth = _gaussian_blur_separable(dem_fill, sigma=float(sigma_krig))
    dem_smooth[nodata] = np.nan

    finite = np.isfinite(dem_smooth)
    if finite.any():
        vmin = float(np.nanmin(dem_smooth[finite]))
        vmax = float(np.nanmax(dem_smooth[finite]))
        if vmax > vmin:
            dem_smooth = (dem_smooth - vmin) / (vmax - vmin)
            dem_smooth = np.clip(dem_smooth, 0, 1)

    base_elev_cm = dem_smooth * float(args.max_depth_cm)

    # Computa 4 estágios e plota a ELEVAÇÃO TOTAL (Base + Sedimento)
    # O Hillshade dará a textura de terreno, e a cor mudará conforme o preenchimento sobe.
    stages = [0.25, 0.50, 0.75, 1.00]
    elev_by_stage: dict[float, np.ndarray] = {}
    met_by_stage: dict[float, list[dict[str, object]]] = {}
    pools: list[np.ndarray] = []
    for f in stages:
        elev_cm, met = _apply_deposition_stage(
            base_elev_cm,
            nodata,
            pal_cfgs,
            stage_frac=float(f),
            sed_inc_by_pid_cm=sed_medio_by_pid,
            default_length_pct=float(args.deposit_length_pct),
        )
        elev_by_stage[float(f)] = elev_cm
        met_by_stage[float(f)] = met
        pools.append(elev_cm[np.isfinite(elev_cm)].ravel())

    vmax_data = float(np.nanmax(np.concatenate([p if p.size else np.array([0.0]) for p in pools]))) if pools else 0.0
    vmax_plot = max(float(args.max_depth_cm), vmax_data)

    for f in stages:
        pct = int(round(100 * float(f)))
        out_path = out_dir / f"{args.prefix}{pct:03d}pct.png"
        _render_stage(
            out_path=out_path,
            elev_cm=elev_by_stage[float(f)],
            metrics=met_by_stage[float(f)],
            nodata=nodata,
            hh=hh,
            ww=ww,
            stage_frac=float(f),
            medio_mm=float(medio_mm),
            vmax_plot=float(vmax_plot),
            contour_interval=float(args.contour_interval),
            dpi=int(args.dpi),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
