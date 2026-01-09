#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownParameterType=false
"""ravina1_simulacao_p90_p95_capacity_em_anos.py

Objetivo
- Reproduzir a figura "capacity" (cunha máxima antes de extravasar) usando a MESMA
  metodologia do script ravina1_simulacao_p90_p95_mapas.py.
- Trocar os rótulos internos de cada paliçada: em vez de "Pico – X cm",
  exibir o TEMPO (anos) para saturar (restante/ sed_inc_mensal / 12).
- Gerar também um gráfico com as curvas de crescimento (preenchimento acumulado)
  por paliçada para P90 e P95.

Entradas
- GeoTIFF/PNG do recorte real (com alpha).
- JSON palicadas_ravina1_config.json.
- série diária (Data, Precipitacao_mm) e CSV integrado de sedimentação.

Saídas
- PNG triptico (a) recorte, (b) P90 capacity, (c) P95 capacity.
- PNG curvas de crescimento (P90 vs P95) para P1–P4.

NOTA IMPORTANTE
- Não usa PNG pronto como base para o mapa.
- O mapa continua sendo "elevação relativa (cm)" (Pseudo-DEM + deposição),
  mas as etiquetas nas paliçadas passam a ser em ANOS.

Autor: GitHub Copilot
Data: 2026-01-09
"""

from __future__ import annotations

import argparse
import csv
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
    rule = (
        f"power_inverse(ref_vib={ref_vib:g}, alpha={alpha:g}, coeff@ref={coeff_at_ref:g}, "
        f"clamp=[{min_coeff:g},{max_coeff:g}])"
    )
    return eff, "vib_model", rule


def _read_csv_simple(path: Path) -> tuple[list[str], list[list[str]]]:
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


def _stat_monthly_totals(series_csv: Path, *, stat: str) -> float:
    """Retorna um resumo típico da chuva mensal (mm/mês) a partir de série diária.

    stat:
      - 'mean': média dos totais mensais
      - 'median': mediana dos totais mensais
    """

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
        sdate = (r[i_date] or "").strip()
        sval = (r[i_p] or "").strip()
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

    stat_norm = str(stat).strip().lower()
    if stat_norm == "mean":
        return float(np.mean(vals))
    if stat_norm == "median":
        return float(np.median(vals))
    raise ValueError("--typical-stat deve ser 'mean' ou 'median'")


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


def _deposit_mask(
    *,
    shape_hw: tuple[int, int],
    pal: tuple[int, float, float, float, float, float],
    length_pct: float,
) -> np.ndarray:
    h, w = shape_hw
    pid, x0, y0, x1, y1, width_pct = pal

    p0 = np.array([x0 * (w - 1), y0 * (h - 1)], dtype=np.float32)
    p1 = np.array([x1 * (w - 1), y1 * (h - 1)], dtype=np.float32)

    v = p1 - p0
    v_norm = np.linalg.norm(v)
    if v_norm <= 1e-6:
        return np.zeros((h, w), dtype=bool)

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


def _apply_deposition_capacity(
    base_elev_cm: np.ndarray,
    nodata: np.ndarray,
    pal_cfgs: list[PalicadaCfg],
    *,
    sed_inc_by_pid_cm: dict[int, float | None],
    default_length_pct: float,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    out = base_elev_cm.copy()
    metrics: list[dict[str, object]] = []

    h, w = base_elev_cm.shape

    for pal in PALICADAS:
        pid = pal[0]
        cfg = next((c for c in pal_cfgs if c.id == pid), None)
        if cfg is None:
            continue

        sed_inc = sed_inc_by_pid_cm.get(pid)
        remaining = max(0.0, cfg.crest_height_cm - cfg.current_fill_cm)

        applied = float(remaining)

        n_events_to_fill: float | None
        if sed_inc is None or not np.isfinite(float(sed_inc)) or float(sed_inc) <= 0:
            n_events_to_fill = None
        else:
            n_events_to_fill = float(remaining / float(sed_inc)) if remaining > 0 else 0.0

        length_pct = cfg.deposit_length_pct if cfg.deposit_length_pct is not None else default_length_pct
        wgt = _deposit_mask(shape_hw=(h, w), pal=pal, length_pct=length_pct)
        wgt[nodata] = 0.0

        deposit_area_px = int(np.count_nonzero(wgt > 0))
        deposit_mean_cm: float | None = None

        if applied > 0 and deposit_area_px > 0:
            delta = (wgt * float(applied)).astype(np.float32)
            out = out + delta
            deposit_mean_cm = float(delta[wgt > 0].mean())

        metrics.append(
            {
                "palicada": f"P{pid}",
                "area_group": cfg.area_group,
                "crest_height_cm": cfg.crest_height_cm,
                "current_fill_cm": cfg.current_fill_cm,
                "remaining_cm": remaining,
                "sed_inc_cm": (None if sed_inc is None else float(sed_inc)),
                "applied_capacity_cm": applied,
                "deposit_mean_cm": deposit_mean_cm,
                "deposit_area_px": deposit_area_px,
                "n_events_to_fill": n_events_to_fill,
                "years_to_fill": (None if n_events_to_fill is None else float(n_events_to_fill) / 12.0),
                "deposit_length_pct": length_pct,
            }
        )

    return out, metrics


def _open_rgba(path: Path) -> np.ndarray:
    im = Image.open(path).convert("RGBA")
    return np.asarray(im)


def _crop_half(arr: np.ndarray, *, side: str) -> np.ndarray:
    h, w = arr.shape[:2]
    mid = w // 2
    if side == "left":
        return arr[:, :mid, :]
    if side == "right":
        return arr[:, mid:, :]
    raise ValueError("side deve ser 'left' ou 'right'")


def _trim_whitespace(arr: np.ndarray, *, tol: int = 245, pad: int = 2) -> np.ndarray:
    if arr.ndim != 3 or arr.shape[2] < 4:
        return arr

    rgb = arr[:, :, :3].astype(np.int16)
    alpha = arr[:, :, 3].astype(np.int16)
    not_transparent = alpha > 5
    not_white = (rgb.mean(axis=2) < tol)
    mask = not_transparent & not_white

    if not mask.any():
        return arr

    ys, xs = np.where(mask)
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, arr.shape[0])
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, arr.shape[1])
    return arr[y0:y1, x0:x1, :]


def _fmt_years(years: float | None) -> str:
    if years is None or not np.isfinite(float(years)):
        return "—"
    y = float(years)
    if y >= 10:
        return f"{y:.0f}".replace(".", ",")
    if y >= 1:
        return f"{y:.1f}".replace(".", ",")
    return f"{y:.2f}".replace(".", ",")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ravina 1: capacity em anos (P90/P95).")
    p.add_argument("--geotiff", required=True, help="Imagem base do RECORTE (GeoTIFF/PNG com alpha).")
    p.add_argument(
        "--recorte-ab",
        required=False,
        help="(Opcional) PNG fig_ravina1_recorte_ab_v2.png. Não é mais usado nas saídas atuais.",
    )
    p.add_argument("--out", required=False, help="PNG de saída (3 painéis: recorte + P90 + P95) OU (2 painéis no modo típico).")
    p.add_argument("--out-curvas", required=False, help="PNG com curvas de crescimento (conforme o(s) cenário(s) gerado(s)).")

    # Saídas independentes (para montagem editorial melhor)
    p.add_argument("--out-medio", required=False, help="PNG de saída: recorte + cenário médio (anos).")
    p.add_argument("--out-p90", required=False, help="PNG de saída: recorte + cenário P90 (anos).")
    p.add_argument("--out-p95", required=False, help="PNG de saída: recorte + cenário P95 (anos).")
    p.add_argument("--out-csv", required=False, help="CSV de saída com as métricas calculadas (anos para preencher).")
    p.add_argument(
        "--medio-stat",
        choices=["median", "mean"],
        default="median",
        help="Como definir o cenário médio a partir dos totais mensais (default: median).",
    )

    p.add_argument("--palicadas-config", required=True, help="JSON com parâmetros por paliçada.")

    p.add_argument(
        "--runoff-coeff",
        type=float,
        default=None,
        help="Override do coeficiente de escoamento superficial (0–1). Se omitido, usa defaults do JSON e/ou regra VIB→runoff.",
    )

    p.add_argument(
        "--precip-series",
        required=True,
        help="CSV diário com Data e Precipitacao_mm.",
    )
    p.add_argument(
        "--sediment-data",
        required=True,
        help="CSV integrado com AREA, RAINFALL e FRACIONADO.",
    )

    p.add_argument("--fracionado-unit", choices=["m", "cm"], default="m")
    p.add_argument("--p90", type=float, default=0.90)
    p.add_argument("--p95", type=float, default=0.95)

    p.add_argument(
        "--typical-stat",
        choices=["mean", "median"],
        default=None,
        help="Se definido, gera apenas um cenário típico baseado na estatística dos totais MENSAIS (mm/mês), em vez de P90/P95.",
    )

    # Visual / DEM
    p.add_argument("--max-grid", type=int, default=500)
    p.add_argument("--blur-radius", type=float, default=2.0)
    p.add_argument("--contour-interval", type=float, default=15.0)
    p.add_argument("--max-depth-cm", type=float, default=100.0)
    p.add_argument("--base-max-px", type=int, default=3000)
    p.add_argument("--deposit-length-pct", type=float, default=0.12)

    return p.parse_args()


def main() -> int:
    args = parse_args()

    geotiff_path = Path(args.geotiff)
    recorte_ab = Path(args.recorte_ab) if args.recorte_ab else None

    out_path = Path(args.out) if args.out else None
    out_curvas = Path(args.out_curvas) if args.out_curvas else None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_curvas is not None:
        out_curvas.parent.mkdir(parents=True, exist_ok=True)

    out_medio = Path(args.out_medio) if args.out_medio else None
    out_p90_only = Path(args.out_p90) if args.out_p90 else None
    out_p95_only = Path(args.out_p95) if args.out_p95 else None
    for p in [out_medio, out_p90_only, out_p95_only]:
        if p is not None:
            p.parent.mkdir(parents=True, exist_ok=True)

    cfg_path = Path(args.palicadas_config)
    pal_cfgs = _load_palicadas_config(cfg_path)

    # Ler VIB e defaults do JSON (mesma lógica do script original)
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

    # Referência do acoplamento: calibração da razão FRACIONADO/RAINFALL corresponde ao coeff_at_ref
    runoff_coeff_ref_for_sed = float(runoff_coeff_default)
    model_obj = cfg_obj.get("vib_to_runoff_model")
    if isinstance(model_obj, dict):
        try:
            runoff_coeff_ref_for_sed = float(model_obj.get("coeff_at_ref", runoff_coeff_ref_for_sed))
        except Exception:
            pass

    precip_series = Path(args.precip_series)
    sediment_data = Path(args.sediment_data)

    typical_stat = args.typical_stat

    # Chuva mensal (P90/P95) ou típica (média/mediana)
    if typical_stat is None:
        p90_mm = _percentile_monthly_totals(precip_series, args.p90)
        p95_mm = _percentile_monthly_totals(precip_series, args.p95)
        typical_mm = None
    else:
        p90_mm = None
        p95_mm = None
        typical_mm = _stat_monthly_totals(precip_series, stat=str(typical_stat))

    # Cenário médio SEM depender de typical_stat (para gerar 3 figuras separadas)
    medio_mm = _stat_monthly_totals(precip_series, stat=str(args.medio_stat))

    # Razão empírica (cm/mm) por área
    ratio_by_area = _fit_sediment_ratio_by_area(sediment_data, fracionado_unit=str(args.fracionado_unit))

    # sed_inc (cm/mês) por paliçada COM acoplamento VIB→runoff (igual ao original)
    sed_p90_by_pid: dict[int, float | None] = {}
    sed_p95_by_pid: dict[int, float | None] = {}
    sed_typical_by_pid: dict[int, float | None] = {}
    sed_medio_by_pid: dict[int, float | None] = {}
    for cfg in pal_cfgs:
        ratio = ratio_by_area.get(cfg.area_group)
        if ratio is None:
            sed_p90_by_pid[cfg.id] = None
            sed_p95_by_pid[cfg.id] = None
            sed_typical_by_pid[cfg.id] = None
            sed_medio_by_pid[cfg.id] = None
            continue

        # Bases por cenário (sem escala VIB→runoff)
        if typical_stat is None:
            assert p90_mm is not None and p95_mm is not None
            sed_p90_base = float(max(0.0, float(ratio) * float(p90_mm)))
            sed_p95_base = float(max(0.0, float(ratio) * float(p95_mm)))
            sed_typ_base = None
        else:
            sed_p90_base = None
            sed_p95_base = None
            sed_typ_base = float(max(0.0, float(ratio) * float(typical_mm)))

        # Base médio (sempre calculado)
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

        if sed_p90_base is not None:
            sed_p90_by_pid[cfg.id] = float(max(0.0, float(sed_p90_base) * float(sed_scale)))
        if sed_p95_base is not None:
            sed_p95_by_pid[cfg.id] = float(max(0.0, float(sed_p95_base) * float(sed_scale)))
        if sed_typ_base is not None:
            sed_typical_by_pid[cfg.id] = float(max(0.0, float(sed_typ_base) * float(sed_scale)))

        sed_medio_by_pid[cfg.id] = float(max(0.0, float(sed_medio_base) * float(sed_scale)))

    # Pseudo-DEM a partir do recorte
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

    sigma_krig = max(4.0, args.blur_radius * 3.0)
    dem_fill = dem_norm.copy()
    if np.isnan(dem_fill).any():
        mean_dem = np.nanmean(dem_fill)
        if not np.isfinite(mean_dem):
            mean_dem = 0.5
        dem_fill[np.isnan(dem_fill)] = mean_dem

    dem_smooth = _gaussian_blur_separable(dem_fill, sigma=sigma_krig)
    dem_smooth[nodata] = np.nan

    finite = np.isfinite(dem_smooth)
    if finite.any():
        vmin = float(np.nanmin(dem_smooth[finite]))
        vmax = float(np.nanmax(dem_smooth[finite]))
        if vmax > vmin:
            dem_smooth = (dem_smooth - vmin) / (vmax - vmin)
            dem_smooth = np.clip(dem_smooth, 0, 1)

    base_elev_cm = dem_smooth * float(args.max_depth_cm)

    elev_p90_cm = None
    elev_p95_cm = None
    elev_typ_cm = None
    elev_medio_cm = None
    met_p90 = None
    met_p95 = None
    met_typ = None
    met_medio = None

    # Computa cenários necessários
    if typical_stat is None:
        elev_p90_cm, met_p90 = _apply_deposition_capacity(
            base_elev_cm,
            nodata,
            pal_cfgs,
            sed_inc_by_pid_cm=sed_p90_by_pid,
            default_length_pct=float(args.deposit_length_pct),
        )
        elev_p95_cm, met_p95 = _apply_deposition_capacity(
            base_elev_cm,
            nodata,
            pal_cfgs,
            sed_inc_by_pid_cm=sed_p95_by_pid,
            default_length_pct=float(args.deposit_length_pct),
        )
    else:
        elev_typ_cm, met_typ = _apply_deposition_capacity(
            base_elev_cm,
            nodata,
            pal_cfgs,
            sed_inc_by_pid_cm=sed_typical_by_pid,
            default_length_pct=float(args.deposit_length_pct),
        )

    # Médio (para figura separada)
    elev_medio_cm, met_medio = _apply_deposition_capacity(
        base_elev_cm,
        nodata,
        pal_cfgs,
        sed_inc_by_pid_cm=sed_medio_by_pid,
        default_length_pct=float(args.deposit_length_pct),
    )

    # Escala (unificada quando gerando múltiplos cenários)
    pools: list[np.ndarray] = []
    if elev_p90_cm is not None:
        pools.append(elev_p90_cm[np.isfinite(elev_p90_cm)].ravel())
    if elev_p95_cm is not None:
        pools.append(elev_p95_cm[np.isfinite(elev_p95_cm)].ravel())
    if elev_typ_cm is not None:
        pools.append(elev_typ_cm[np.isfinite(elev_typ_cm)].ravel())
    if elev_medio_cm is not None:
        pools.append(elev_medio_cm[np.isfinite(elev_medio_cm)].ravel())

    if len(pools) == 0:
        vmax_data = 0.0
    else:
        vmax_data = float(np.nanmax(np.concatenate([p if p.size else np.array([0.0]) for p in pools])))

    vmax_plot = max(float(args.max_depth_cm), vmax_data)

    cmap_name = "Spectral_r"
    cmap = plt.get_cmap(cmap_name)

    def draw_map_panel(
        ax: Any,
        elev_cm: np.ndarray,
        metrics: list[dict[str, object]],
        *,
        scenario: str,
        scenario_mm: float,
    ) -> Any:
        ax.set_axis_off()

        # Gera Hillshade para textura (baseado na elevação atual)
        z_for_hs = elev_cm.copy()
        z_for_hs[nodata] = np.nanmean(z_for_hs[~nodata]) if (~nodata).any() else 0.0
        hs = _hillshade(z_for_hs, cellsize=1.0, az_deg=315, alt_deg=45)

        elev_masked = np.ma.masked_invalid(elev_cm)
        elev_masked[nodata] = np.ma.masked

        # 1. Plota Hillshade em escala de cinza como base (textura)
        hs_masked = np.ma.masked_where(nodata, hs)
        ax.imshow(hs_masked, cmap="gray", origin="upper", vmin=0, vmax=1, alpha=1.0)

        # 2. Plota Elevação (Cor) com transparência para fundir com hillshade
        im_ = ax.imshow(elev_masked, cmap=cmap, origin="upper", vmin=0, vmax=vmax_plot, alpha=0.65)

        if elev_masked.count() > 0:
            levels = np.arange(0, vmax_plot + 1, float(args.contour_interval))
            levels = levels[levels > 0]
            cs = ax.contour(elev_masked, levels=levels, colors="black", linewidths=0.55, alpha=0.6)
            ax.clabel(cs, inline=True, fontsize=8, fmt="%d", colors="black")

        # Paliçadas + etiqueta em ANOS
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

        # caixa de contexto
        ax.text(
            0.02,
            0.02,
            f"{scenario} mensal = {scenario_mm:.1f} mm\nmodo=capacity\n(rótulos em anos)",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            color="black",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75, edgecolor="none"),
        )

        return im_

    if typical_stat is None:
        if out_path is not None:
            # Figura: apenas mapas gerados (P90 e P95) + barra de cores
            fig = plt.figure(figsize=(12.5, 6.6))
            gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.06], wspace=0.0, hspace=0.10)

            ax_b = fig.add_subplot(gs[0, 0])
            ax_c = fig.add_subplot(gs[0, 1])

            _ = draw_map_panel(ax_b, elev_p90_cm, met_p90, scenario="P90", scenario_mm=float(p90_mm))
            _ = draw_map_panel(ax_c, elev_p95_cm, met_p95, scenario="P95", scenario_mm=float(p95_mm))

            ax_cb_blank = fig.add_subplot(gs[1, 0])
            ax_cb_blank.set_axis_off()
            ax_cb = fig.add_subplot(gs[1, 1])

            norm = mcolors.Normalize(vmin=0.0, vmax=float(vmax_plot))
            sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            cbar = fig.colorbar(sm, cax=ax_cb, orientation="horizontal")
            cbar.set_label("Elevação Relativa Total (Base + Sedimento) [cm]")

            fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.08, wspace=0.0, hspace=0.10)
            fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
            plt.close(fig)
    else:
        if out_path is not None:
            # Figura: apenas mapa gerado do cenário típico + barra de cores
            fig = plt.figure(figsize=(6.6, 6.6))
            gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.06], wspace=0.0, hspace=0.10)

            ax_b = fig.add_subplot(gs[0, 0])
            scen_label = "Típico (média)" if str(typical_stat).lower() == "mean" else "Típico (mediana)"
            _ = draw_map_panel(ax_b, elev_typ_cm, met_typ, scenario=scen_label, scenario_mm=float(typical_mm))

            ax_cb = fig.add_subplot(gs[1, 0])

            norm = mcolors.Normalize(vmin=0.0, vmax=float(vmax_plot))
            sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            cbar = fig.colorbar(sm, cax=ax_cb, orientation="horizontal")
            cbar.set_label("Elevação Relativa Total (Base + Sedimento) [cm]")

            fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.10, wspace=0.0, hspace=0.10)
            fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
            plt.close(fig)

    # Saídas separadas (apenas mapa gerado)
    def _save_single_panel(*, out_file: Path, elev_cm: np.ndarray, metrics: list[dict[str, object]], scenario: str, scenario_mm: float) -> None:
        fig = plt.figure(figsize=(6.6, 6.6))
        gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.06], wspace=0.0, hspace=0.10)
        ax_b = fig.add_subplot(gs[0, 0])

        _ = draw_map_panel(ax_b, elev_cm, metrics, scenario=scenario, scenario_mm=float(scenario_mm))

        ax_cb = fig.add_subplot(gs[1, 0])
        norm = mcolors.Normalize(vmin=0.0, vmax=float(vmax_plot))
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = fig.colorbar(sm, cax=ax_cb, orientation="horizontal")
        cbar.set_label("Elevação Relativa Total (Base + Sedimento) [cm]")

        fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.10, wspace=0.0, hspace=0.10)
        fig.savefig(out_file, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    if out_medio is not None:
        medio_label = "Médio (média)" if str(args.medio_stat).lower() == "mean" else "Médio (mediana)"
        _save_single_panel(out_file=out_medio, elev_cm=elev_medio_cm, metrics=met_medio, scenario=medio_label, scenario_mm=float(medio_mm))

    if out_p90_only is not None and elev_p90_cm is not None and met_p90 is not None and p90_mm is not None:
        _save_single_panel(out_file=out_p90_only, elev_cm=elev_p90_cm, metrics=met_p90, scenario="P90", scenario_mm=float(p90_mm))

    if out_p95_only is not None and elev_p95_cm is not None and met_p95 is not None and p95_mm is not None:
        _save_single_panel(out_file=out_p95_only, elev_cm=elev_p95_cm, metrics=met_p95, scenario="P95", scenario_mm=float(p95_mm))

    # Export CSV if requested
    if args.out_csv:
        out_csv_path = Path(args.out_csv)
        out_csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        rows = []
        def get_years(metrics_list, pid):
            if metrics_list is None: return None
            m = next((x for x in metrics_list if x.get("palicada") == f"P{pid}"), None)
            return m.get("years_to_fill") if m else None

        for cfg in pal_cfgs:
            pid = cfg.id
            row = {
                "palicada": f"P{pid}",
                "area_group": cfg.area_group,
                "crest_height_cm": cfg.crest_height_cm,
                "current_fill_cm": cfg.current_fill_cm,
            }
            
            # Medio is always calculated
            row["years_medio"] = get_years(met_medio, pid)

            if typical_stat is None:
                row["years_p90"] = get_years(met_p90, pid)
                row["years_p95"] = get_years(met_p95, pid)
            else:
                row["years_typical"] = get_years(met_typ, pid)
            
            rows.append(row)
            
        with out_csv_path.open("w", encoding="utf-8", newline="") as f:
            if rows:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

    if out_curvas is None:
        return 0

    f2 = plt.figure(figsize=(10, 6))
    ax = f2.add_subplot(1, 1, 1)
    ax.set_title("Curvas de crescimento do preenchimento (modo capacity)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Tempo (anos)")
    ax.set_ylabel("Preenchimento acumulado (cm)")

    colors = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c", 4: "#d62728"}

    if typical_stat is None:
        for cfg in pal_cfgs:
            pid = int(cfg.id)
            c = colors.get(pid, "black")

            x90, y90 = _curve_for(cfg, sed_p90_by_pid.get(pid))
            x95, y95 = _curve_for(cfg, sed_p95_by_pid.get(pid))

            ax.plot(x90, y90, color=c, linewidth=2.0, label=f"P{pid} (P90)")
            ax.plot(x95, y95, color=c, linewidth=2.0, linestyle="--", label=f"P{pid} (P95)")
    else:
        for cfg in pal_cfgs:
            pid = int(cfg.id)
            c = colors.get(pid, "black")
            xt, yt = _curve_for(cfg, sed_typical_by_pid.get(pid))
            ax.plot(xt, yt, color=c, linewidth=2.0, label=f"P{pid} ({scen_label})")

    ax.grid(True, alpha=0.25)
    ax.legend(ncols=2, fontsize=9, frameon=False)
    f2.tight_layout()
    f2.savefig(out_curvas, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(f2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
