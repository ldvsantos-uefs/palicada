#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownParameterType=false
"""ravina1_simulacao_p90_p95_palicadas.py

Gera uma figura no estilo "recorte grid" da Ravina 1, incorporando uma simulação
simplificada de eventos P90 e P95 de precipitação e um balanço de capacidade de
acúmulo a montante das paliçadas P1–P4.

Modelo adotado (intencionalmente parsimonioso):
- P90 e P95 são extraídos de uma série diária (20 anos) agregada em totais mensais.
- A resposta sedimentar incremental (cm por mês) sob um total de chuva é estimada
  por um modelo empírico por área (SUP/MED/INF) ajustado a partir de
  dados_integrados_sedimentacao.csv.
    Importante: no CSV integrado, FRACIONADO normalmente está em *metros* (ex.: 0.05 ≈ 5 cm).
    Este script permite escolher a unidade via --fracionado-unit.
- A verificação "não extravasar por cima" é tratada como um limite geométrico de
  altura acumulável (crest_height_cm), parametrizado por paliçada no JSON.

Limitações conscientes:
- O painel (b) usa Pseudo-DEM (escala imposta em cm) derivado de luminância do
  ortomosaico, seguindo ravina1_analise_visual_v2.py. A estimativa de volume em m³
  só é emitida se a área de depósito (pool_area_m2) estiver informada no JSON.

Uso típico:
  python ravina1_simulacao_p90_p95_palicadas.py \
    --geotiff ".../corte_ravina_1.tif" \
    --out ".../fig_ravina1_sim_p90_p95.png" \
    --palicadas-config ".../palicadas_ravina1_config.json" \
    --out-csv ".../sim_p90_p95_palicadas.csv"

Se o arquivo JSON não existir, o script gera um template e interrompe.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from PIL import Image


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

    pool_area_m2: float | None
    catchment_area_m2: float | None


def pct_to_px(x: float, y: float, w: int, h: int) -> tuple[float, float]:
    return x * (w - 1), y * (h - 1)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _runoff_coeff_from_vib(
    vib_cm_h: float | None,
    *,
    cfg_obj: Mapping[str, Any],
    fallback_coeff: float,
) -> tuple[float, str, str]:
    """Converte VIB em coeficiente de escoamento superficial (0–1).

    Regra (quando habilitada no JSON):
      runoff_coeff_eff = clamp(coeff_at_ref * (ref_vib_cm_h / vib_cm_h)^alpha, min_coeff, max_coeff)
    """

    model = cfg_obj.get("vib_to_runoff_model")
    if not isinstance(model, dict):
        return fallback_coeff, "defaults", ""

    enabled = bool(model.get("enabled", True))
    method = str(model.get("method", "")).strip().lower()
    if not enabled or method != "power_inverse":
        return fallback_coeff, "defaults", ""

    if vib_cm_h is None or vib_cm_h <= 0:
        return fallback_coeff, "defaults", ""

    try:
        ref_vib = float(model.get("ref_vib_cm_h", 2.0))
        alpha = float(model.get("alpha", 1.0))
        coeff_at_ref = float(model.get("coeff_at_ref", fallback_coeff))
        min_coeff = float(model.get("min_coeff", 0.0))
        max_coeff = float(model.get("max_coeff", 1.0))
    except Exception:
        return fallback_coeff, "defaults", ""

    raw = coeff_at_ref * (ref_vib / vib_cm_h) ** alpha
    eff = _clamp(raw, min_coeff, max_coeff)
    rule = f"power_inverse(ref_vib={ref_vib:g}, alpha={alpha:g}, coeff@ref={coeff_at_ref:g}, clamp=[{min_coeff:g},{max_coeff:g}])"
    return eff, "vib_model", rule


def plot_palicadas_2d(
    ax: Any,
    w: int,
    h: int,
    *,
    color: str,
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


def ridge_mask(
    grid: np.ndarray,
    x0_pct: float,
    y0_pct: float,
    x1_pct: float,
    y1_pct: float,
    width_pct: float,
) -> np.ndarray:
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


def _read_csv_simple(path: Path) -> tuple[list[str], list[list[str]]]:
    import csv

    with path.open("r", encoding="utf-8") as f:
        r = csv.reader(f)
        rows = list(r)

    if not rows:
        return [], []

    header = rows[0]
    data = rows[1:]
    return header, data


def _percentile_monthly_totals(series_csv: Path, q: float) -> float:
    header, rows = _read_csv_simple(series_csv)
    if not header:
        raise ValueError(f"CSV vazio ou inválido: {series_csv}")

    try:
        i_date = header.index("Data")
        i_p = header.index("Precipitacao_mm")
    except ValueError as e:
        raise ValueError(f"CSV sem colunas esperadas (Data, Precipitacao_mm): {series_csv}") from e

    month_sum: dict[tuple[int, int], float] = {}
    for r in rows:
        if len(r) <= max(i_date, i_p):
            continue
        sdate = r[i_date].strip()
        sval = r[i_p].strip()
        if not sdate:
            continue
        try:
            year = int(sdate[0:4])
            month = int(sdate[5:7])
        except Exception:
            continue
        try:
            p = float(sval)
        except Exception:
            p = 0.0
        key = (year, month)
        month_sum[key] = month_sum.get(key, 0.0) + p

    vals = np.array([v for v in month_sum.values() if np.isfinite(v)], dtype=float)
    if vals.size == 0:
        raise ValueError(f"Sem totais mensais válidos em: {series_csv}")

    return float(np.quantile(vals, q))


def _fit_sediment_ratio_by_area(sed_csv: Path, *, fracionado_unit: str) -> dict[str, float]:
    header, rows = _read_csv_simple(sed_csv)
    if not header:
        raise ValueError(f"CSV vazio ou inválido: {sed_csv}")

    required = ["AREA", "RAINFALL", "FRACIONADO"]
    for c in required:
        if c not in header:
            raise ValueError(f"CSV sem coluna {c}: {sed_csv}")

    i_area = header.index("AREA")
    i_r = header.index("RAINFALL")
    i_f = header.index("FRACIONADO")

    i_year = header.index("YEAR") if "YEAR" in header else None
    i_month = header.index("MONTH") if "MONTH" in header else None
    i_data = header.index("DATA") if "DATA" in header else None

    # Alguns arquivos deixam RAINFALL em branco para MED/INF.
    # Faz uma imputação conservadora por mês usando as linhas que possuem chuva.
    rainfall_by_ym: dict[tuple[int, int], float] = {}
    rainfall_by_data: dict[str, float] = {}
    for r in rows:
        if len(r) <= i_r:
            continue
        try:
            rainfall = float(r[i_r])
        except Exception:
            continue
        if not np.isfinite(rainfall) or rainfall <= 0:
            continue

        if i_year is not None and i_month is not None and len(r) > max(i_year, i_month):
            try:
                yy = int(float(r[i_year]))
                mm = int(float(r[i_month]))
                rainfall_by_ym[(yy, mm)] = rainfall
            except Exception:
                pass

        if i_data is not None and len(r) > i_data:
            sdata = (r[i_data] or "").strip()
            if sdata:
                rainfall_by_data[sdata] = rainfall

    ratios: dict[str, list[float]] = {}
    for r in rows:
        if len(r) <= max(i_area, i_r, i_f):
            continue

        area = (r[i_area] or "").strip()
        if not area:
            continue

        rainfall: float | None
        try:
            rainfall = float(r[i_r])
        except Exception:
            rainfall = None

        if rainfall is None or not np.isfinite(rainfall) or rainfall <= 0:
            rainfall = None

        if rainfall is None:
            if i_data is not None and len(r) > i_data:
                sdata = (r[i_data] or "").strip()
                rainfall = rainfall_by_data.get(sdata)

        if rainfall is None:
            if i_year is not None and i_month is not None and len(r) > max(i_year, i_month):
                try:
                    yy = int(float(r[i_year]))
                    mm = int(float(r[i_month]))
                    rainfall = rainfall_by_ym.get((yy, mm))
                except Exception:
                    rainfall = None

        if rainfall is None or not np.isfinite(rainfall) or rainfall <= 0:
            continue

        try:
            frac_raw = float(r[i_f])
        except Exception:
            continue

        if not np.isfinite(frac_raw) or frac_raw <= 0:
            continue

        frac_cm: float
        if fracionado_unit.lower() == "m":
            frac_cm = float(frac_raw) * 100.0
        elif fracionado_unit.lower() == "cm":
            frac_cm = float(frac_raw)
        else:
            raise ValueError("--fracionado-unit deve ser 'm' ou 'cm'")

        ratios.setdefault(area, []).append(frac_cm / rainfall)

    out: dict[str, float] = {}
    for area, rr in ratios.items():
        if len(rr) == 0:
            continue
        out[area] = float(np.median(np.array(rr, dtype=float)))

    return out


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
        cfgs.append(
            PalicadaCfg(
                id=int(p["id"]),
                area_group=str(p["area_group"]),
                crest_height_cm=float(p["crest_height_cm"]),
                current_fill_cm=float(p.get("current_fill_cm", 0.0)),
                pool_area_m2=(None if p.get("pool_area_m2") in (None, "") else float(p["pool_area_m2"])),
                catchment_area_m2=(None if p.get("catchment_area_m2") in (None, "") else float(p["catchment_area_m2"])),
            )
        )

    return cfgs


def _write_config_template(path: Path) -> None:
    template = {
        "vib_cm_h_by_area_group": {
            "SUP": 3.13,
            "MED": 1.53,
            "INF": 1.59,
        },
        "vib_notes": "VIB (cm/h) por terço da ravina; média das 5 últimas leituras em 2-DADOS/Infiltracao.xlsx.",
        "vib_to_runoff_model": {
            "enabled": True,
            "method": "power_inverse",
            "ref_vib_cm_h": 2.0,
            "alpha": 1.0,
            "coeff_at_ref": 0.6,
            "min_coeff": 0.2,
            "max_coeff": 0.95,
        },
        "vib_to_runoff_notes": "Regra: runoff_coeff_eff = clamp(coeff_at_ref * (ref_vib_cm_h / vib_cm_h)^alpha, min_coeff, max_coeff). Se --runoff-coeff for passado no CLI, ele sobrescreve tudo.",
        "palicadas": [
            {
                "id": 1,
                "area_group": "SUP",
                "crest_height_cm": 100.0,
                "current_fill_cm": 0.0,
                "pool_area_m2": None,
                "catchment_area_m2": None,
            },
            {
                "id": 2,
                "area_group": "MED",
                "crest_height_cm": 100.0,
                "current_fill_cm": 0.0,
                "pool_area_m2": None,
                "catchment_area_m2": None,
            },
            {
                "id": 3,
                "area_group": "MED",
                "crest_height_cm": 100.0,
                "current_fill_cm": 0.0,
                "pool_area_m2": None,
                "catchment_area_m2": None,
            },
            {
                "id": 4,
                "area_group": "INF",
                "crest_height_cm": 100.0,
                "current_fill_cm": 0.0,
                "pool_area_m2": None,
                "catchment_area_m2": None,
            },
        ],
        "defaults": {
            "runoff_coeff": 0.6
        },
        "notes": "Preencha pool_area_m2 para obter volumes. crest_height_cm representa a altura útil para acumulação sem extravasamento. current_fill_cm pode ser usado para descontar o preenchimento já ocorrido.",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(template, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ravina 1: simulação P90/P95 e capacidade de paliçadas.")
    p.add_argument("--geotiff", required=True, help="Imagem base (GeoTIFF/PNG) da Ravina 1 usada no recorte.")
    p.add_argument("--out", required=True, help="Arquivo PNG de saída.")

    p.add_argument(
        "--palicadas-config",
        required=False,
        help="JSON com parâmetros por paliçada (área, altura útil, preenchimento atual).",
    )
    p.add_argument("--out-csv", required=False, help="CSV de saída com métricas por paliçada.")

    p.add_argument(
        "--precip-series",
        default=str(Path("CLIMATOLOGIA_20ANOS") / "dados" / "serie_precipitacao_20anos.csv"),
        help="CSV diário com Data e Precipitacao_mm.",
    )
    p.add_argument(
        "--sediment-data",
        default=str(Path("CLIMATOLOGIA_20ANOS") / "dados" / "dados_integrados_sedimentacao.csv"),
        help="CSV integrado com AREA, RAINFALL e FRACIONADO.",
    )

    p.add_argument(
        "--fracionado-unit",
        choices=["m", "cm"],
        default="m",
        help="Unidade da coluna FRACIONADO no CSV integrado (default: m).",
    )

    p.add_argument("--p90", type=float, default=0.90)
    p.add_argument("--p95", type=float, default=0.95)

    p.add_argument("--runoff-coeff", type=float, default=None, help="Coeficiente de escoamento superficial (0–1).")

    p.add_argument("--max-grid", type=int, default=500)
    p.add_argument("--blur-radius", type=float, default=2.0)
    p.add_argument("--contour-interval", type=float, default=15.0)
    p.add_argument("--max-depth-cm", type=float, default=100.0)
    p.add_argument("--base-max-px", type=int, default=3000)

    p.add_argument("--debug-grid", action="store_true")
    p.add_argument("--grid-n", type=int, default=10)

    return p.parse_args()


def main() -> int:
    args = parse_args()

    geotiff_path = Path(args.geotiff)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    precip_series = Path(args.precip_series)
    sediment_data = Path(args.sediment_data)

    cfg_path = Path(args.palicadas_config) if args.palicadas_config else None
    if cfg_path is None or not cfg_path.exists():
        template_path = Path(__file__).with_name("palicadas_ravina1_config.template.json")
        _write_config_template(template_path)
        raise FileNotFoundError(
            "Arquivo de configuração não encontrado. Um template foi criado em "
            f"{template_path}. Preencha e passe via --palicadas-config."
        )

    pal_cfgs = _load_palicadas_config(cfg_path)

    ratio_by_area = _fit_sediment_ratio_by_area(sediment_data, fracionado_unit=str(args.fracionado_unit))

    p90_mm = _percentile_monthly_totals(precip_series, args.p90)
    p95_mm = _percentile_monthly_totals(precip_series, args.p95)

    # Leitura e Pseudo-DEM conforme v2
    if not geotiff_path.exists():
        raise FileNotFoundError(f"Imagem não encontrada: {geotiff_path}")

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

    # Suavização adicional e forçamento de cristas na linha das paliçadas
    sigma_krig = max(4.0, args.blur_radius * 3.0)

    dem_fill = dem_norm.copy()
    if np.isnan(dem_fill).any():
        mean_dem = np.nanmean(dem_fill)
        if not np.isfinite(mean_dem):
            mean_dem = 0.5
        dem_fill[np.isnan(dem_fill)] = mean_dem

    dem_smooth = _gaussian_blur_separable(dem_fill, sigma=sigma_krig)
    dem_smooth[nodata] = np.nan

    m_all = np.zeros_like(dem_smooth, dtype=np.float32)
    for pid, x0, y0, x1, y1, w_pct in PALICADAS:
        m = ridge_mask(dem_smooth, x0, y0, x1, y1, w_pct)
        m_all = np.clip(m_all + m, 0, 1)

    dem_smooth = dem_smooth * (1 - m_all) + 1.0 * m_all
    dem_smooth = np.clip(dem_smooth, 0, 1)

    finite = np.isfinite(dem_smooth)
    if finite.any():
        vmin = float(np.nanmin(dem_smooth[finite]))
        vmax = float(np.nanmax(dem_smooth[finite]))
        if vmax > vmin:
            dem_smooth = (dem_smooth - vmin) / (vmax - vmin)

    # Escala imposta (cm) para visualização estilo Pseudo-DEM
    elevation_cm = dem_smooth * float(args.max_depth_cm)

    # Figura base no estilo ab
    fig = plt.figure(figsize=(13, 6.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.2])

    ax_a = fig.add_subplot(gs[0])
    ax_a.set_axis_off()
    ax_a.set_title("(a) Recorte ortomosaico", fontsize=12, y=0.95)

    base_rgba = np.dstack([base_s, alpha_s])
    base_rgba[nodata, 3] = 0.0
    ax_a.imshow(base_rgba, origin="upper")
    plot_palicadas_2d(ax_a, w=alpha_s.shape[1], h=alpha_s.shape[0], annotate=True, color="deepskyblue")

    ax_b = fig.add_subplot(gs[1])
    ax_b.set_axis_off()
    ax_b.set_title("(b) Mapa de Elevação Relativa (cm)", fontsize=12, y=1.0)

    elev_masked = np.ma.masked_invalid(elevation_cm)
    elev_masked[nodata] = np.ma.masked

    cmap_elev_final = plt.get_cmap("Spectral_r")
    im_plot = ax_b.imshow(elev_masked, cmap=cmap_elev_final, origin="upper", vmin=0, vmax=args.max_depth_cm)

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

    if elev_masked.count() > 0:
        levels_elev = np.arange(0, args.max_depth_cm + 1, args.contour_interval)
        cs = ax_b.contour(elev_masked, levels=levels_elev, colors="black", linewidths=0.6, alpha=0.6)
        ax_b.clabel(cs, inline=True, fontsize=8, fmt="%d", colors="black")

    cbar = plt.colorbar(im_plot, ax=ax_b, fraction=0.046, pad=0.04, shrink=0.7)
    cbar.set_label("Elevação relativa (escala imposta) (cm)", rotation=270, labelpad=15)

    # Texto de simulação no painel (b)
    txt_lines = [
        f"P90 mensal (20 anos) = {p90_mm:.1f} mm",
        f"P95 mensal (20 anos) = {p95_mm:.1f} mm",
    ]
    ax_b.text(
        0.02,
        0.98,
        "\n".join(txt_lines),
        transform=ax_b.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="black",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75, edgecolor="none"),
    )

    # Métricas por paliçada
    out_rows: list[dict[str, object]] = []

    # Parâmetros opcionais vindos do JSON (runoff_coeff e VIB por segmento)
    cfg_obj: dict[str, Any] = {}
    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            cfg_obj = loaded
    except Exception:
        cfg_obj = {}

    runoff_coeff_cli = args.runoff_coeff
    if runoff_coeff_cli is None:
        try:
            runoff_coeff_default = float(cfg_obj.get("defaults", {}).get("runoff_coeff", 0.6))
        except Exception:
            runoff_coeff_default = 0.6
    else:
        runoff_coeff_default = float(runoff_coeff_cli)

    vib_by_area: dict[str, float] = {}
    vib_raw = cfg_obj.get("vib_cm_h_by_area_group")
    if isinstance(vib_raw, dict):
        for k, v in vib_raw.items():
            try:
                vib_by_area[str(k).strip().upper()] = float(v)
            except Exception:
                continue

    # Referência para modular sedimentação: assume que a calibração (razão FRACIONADO/RAINFALL)
    # corresponde, em média, ao coeff_at_ref do modelo VIB→runoff (ou ao default se não existir).
    runoff_coeff_ref_for_sed = runoff_coeff_default
    model_obj = cfg_obj.get("vib_to_runoff_model")
    if isinstance(model_obj, dict):
        try:
            runoff_coeff_ref_for_sed = float(model_obj.get("coeff_at_ref", runoff_coeff_ref_for_sed))
        except Exception:
            pass

    for cfg in pal_cfgs:
        area = cfg.area_group
        ratio = ratio_by_area.get(area)
        if ratio is None:
            # Sem ajuste para a área, mantém None
            sed_p90_cm = None
            sed_p95_cm = None
        else:
            sed_p90_cm = max(0.0, float(ratio) * p90_mm)
            sed_p95_cm = max(0.0, float(ratio) * p95_mm)

        # Acoplamento (mais forte): VIB altera runoff_coeff e isso modula a sedimentação incremental.
        # Implementação parsimoniosa: sed_inc_eff = sed_inc_base * (runoff_coeff_eff / runoff_coeff_ref).
        sed_scale_runoff = 1.0
        sed_p90_cm_base = sed_p90_cm
        sed_p95_cm_base = sed_p95_cm

        remaining_cm = max(0.0, cfg.crest_height_cm - cfg.current_fill_cm)

        n_p90 = None
        n_p95 = None
        if sed_p90_cm is not None and sed_p90_cm > 0:
            n_p90 = remaining_cm / sed_p90_cm
        if sed_p95_cm is not None and sed_p95_cm > 0:
            n_p95 = remaining_cm / sed_p95_cm

        vib_cm_h = vib_by_area.get(str(area).strip().upper())
        runoff_index_1_over_vib = (1.0 / vib_cm_h) if (vib_cm_h is not None and vib_cm_h > 0) else None

        runoff_coeff_used = runoff_coeff_default
        runoff_coeff_source = "args" if runoff_coeff_cli is not None else "defaults"
        runoff_coeff_rule = ""
        if runoff_coeff_cli is None:
            runoff_coeff_used, runoff_coeff_source, runoff_coeff_rule = _runoff_coeff_from_vib(
                vib_cm_h,
                cfg_obj=cfg_obj,
                fallback_coeff=runoff_coeff_default,
            )

        if runoff_coeff_ref_for_sed > 0:
            sed_scale_runoff = float(runoff_coeff_used) / float(runoff_coeff_ref_for_sed)

        if sed_p90_cm is not None:
            sed_p90_cm = float(max(0.0, float(sed_p90_cm) * sed_scale_runoff))
        if sed_p95_cm is not None:
            sed_p95_cm = float(max(0.0, float(sed_p95_cm) * sed_scale_runoff))

        runoff_p90_m3 = None
        runoff_p95_m3 = None
        if cfg.catchment_area_m2 is not None:
            runoff_p90_m3 = (p90_mm / 1000.0) * cfg.catchment_area_m2 * runoff_coeff_used
            runoff_p95_m3 = (p95_mm / 1000.0) * cfg.catchment_area_m2 * runoff_coeff_used

        cap_m3 = None
        if cfg.pool_area_m2 is not None:
            cap_m3 = (remaining_cm / 100.0) * cfg.pool_area_m2

        out_rows.append(
            {
                "palicada": f"P{cfg.id}",
                "area_group": area,
                "vib_cm_h": vib_cm_h,
                "runoff_index_1_over_vib": runoff_index_1_over_vib,
                "crest_height_cm": cfg.crest_height_cm,
                "current_fill_cm": cfg.current_fill_cm,
                "remaining_cm": remaining_cm,
                "p90_mm": p90_mm,
                "p95_mm": p95_mm,
                "sed_scale_runoff": sed_scale_runoff,
                "sed_inc_p90_cm_base": sed_p90_cm_base,
                "sed_inc_p95_cm_base": sed_p95_cm_base,
                "sed_inc_p90_cm": sed_p90_cm,
                "sed_inc_p95_cm": sed_p95_cm,
                "n_events_p90": n_p90,
                "n_events_p95": n_p95,
                "pool_area_m2": cfg.pool_area_m2,
                "catchment_area_m2": cfg.catchment_area_m2,
                "runoff_coeff": runoff_coeff_used,
                "runoff_coeff_source": runoff_coeff_source,
                "runoff_coeff_rule": runoff_coeff_rule,
                "runoff_p90_m3": runoff_p90_m3,
                "runoff_p95_m3": runoff_p95_m3,
                "capacity_remaining_m3": cap_m3,
            }
        )

        # Anotação compacta no painel A, próxima ao centro da paliçada
        for pid, x0, y0, x1, y1, _ in PALICADAS:
            if pid != cfg.id:
                continue
            cx = (x0 + x1) / 2
            cy = (y0 + y1) / 2
            px, py = pct_to_px(cx, cy, alpha_s.shape[1], alpha_s.shape[0])

            label = f"P{cfg.id}"
            if n_p90 is not None and n_p95 is not None:
                label = f"P{cfg.id} n90={n_p90:.1f} n95={n_p95:.1f}"
            elif n_p90 is not None:
                label = f"P{cfg.id} n90={n_p90:.1f}"
            elif n_p95 is not None:
                label = f"P{cfg.id} n95={n_p95:.1f}"

            ax_a.text(
                px,
                py + 14,
                label,
                color="white",
                fontsize=8,
                ha="center",
                va="top",
                path_effects=[pe.withStroke(linewidth=2.5, foreground="black")],
            )

    # Opcional, grid de debug
    if args.debug_grid:
        h, w = alpha_s.shape
        n = max(2, int(args.grid_n))
        for i in range(1, n):
            x = w * i / n
            y = h * i / n
            ax_a.axvline(x, color="white", linewidth=0.6, alpha=0.7)
            ax_a.axhline(y, color="white", linewidth=0.6, alpha=0.7)
            ax_b.axvline(x, color="black", linewidth=0.5, alpha=0.35)
            ax_b.axhline(y, color="black", linewidth=0.5, alpha=0.35)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    if args.out_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)

        import csv

        if out_rows:
            fieldnames = list(out_rows[0].keys())
            with out_csv.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(out_rows)

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
