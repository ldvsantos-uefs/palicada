#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownParameterType=false
"""ravina1_simulacao_p90_p95_mapas.py

Gera UMA figura com DOIS mapas de elevação relativa (Pseudo-DEM) no recorte da Ravina 1:
(a) cenário P90 e (b) cenário P95.

O que é simulado
- P90 e P95: percentis dos totais MENSAIS derivados de uma série DIÁRIA de precipitação.
- Sedimentação incremental (cm/evento mensal): estimada a partir de dados observados
    (FRACIONADO vs RAINFALL) por área (SUP/MED/INF), usando uma razão robusta
    (mediana de FRACIONADO/RAINFALL para registros com valores positivos).
    Importante: no CSV integrado, FRACIONADO normalmente está em *metros* (ex.: 0.05 ≈ 5 cm).
    Este script permite escolher a unidade via --fracionado-unit.
- Depósito a montante das paliçadas: aplicado como um incremento de elevação (cm)
  em uma região "a montante" de cada paliçada (máscara geométrica simples), com
  limite de capacidade (crest_height_cm - current_fill_cm).

Entradas
- --geotiff: imagem do recorte da ravina (GeoTIFF/PNG com alpha); idealmente o recorte real.
- --palicadas-config: JSON com parâmetros por paliçada (área, altura útil, preenchimento).

Saídas
- --out: PNG com painéis (a) P90 e (b) P95.
- --out-csv (opcional): tabela com sed_inc estimada, incremento aplicado e flag de extravasamento.

Observação
Este script NÃO tenta estimar hidráulica dentro do canal; ele entrega uma simulação
coerente com seus dados (chuva↔sedimentação) e visualização no padrão "mapa".
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
import matplotlib.patheffects as pe
from PIL import Image


# Paliçadas em coordenadas normalizadas (0–1) do FRAME do RECORTE
PALICADAS = [
    # (id, x0, y0, x1, y1, largura_rel)
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

    # Imputação por mês/dia para MED/INF quando RAINFALL vier vazio
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

        # Converte para cm antes de calcular razão (cm/mm)
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


def _deposit_mask(
    *,
    shape_hw: tuple[int, int],
    pal: tuple[int, float, float, float, float, float],
    length_pct: float,
) -> np.ndarray:
    """Retorna um campo de pesos (0–1) para depositar uma cunha a montante.

    Regras:
    - Região é um buffer em torno da linha da paliçada (largura definida por largura_rel),
      restringido a uma faixa a montante (lado da linha determinado por um ponto de teste
      deslocado para cima na imagem).
    - Também limita ao comprimento (a montante) via projeção ao longo do vetor de escoamento
      (assumido para cima na imagem).
    """

    h, w = shape_hw
    pid, x0, y0, x1, y1, width_pct = pal

    # Endpoints em pixels
    p0 = np.array([x0 * (w - 1), y0 * (h - 1)], dtype=np.float32)
    p1 = np.array([x1 * (w - 1), y1 * (h - 1)], dtype=np.float32)

    # Vetor da linha e normal
    v = p1 - p0
    v_norm = np.linalg.norm(v)
    if v_norm <= 1e-6:
        return np.zeros((h, w), dtype=bool)

    v_unit = v / v_norm
    n_unit = np.array([-v_unit[1], v_unit[0]], dtype=np.float32)

    # Centro da linha
    c = 0.5 * (p0 + p1)

    # Define "montante" como o lado onde um ponto deslocado para cima (y-)
    # cairia em relação à linha.
    test = c + np.array([0.0, -1.0], dtype=np.float32)

    # Função de lado via produto vetorial 2D (sinal)
    def side(pt: np.ndarray) -> float:
        return float((p1[0] - p0[0]) * (pt[1] - p0[1]) - (p1[1] - p0[1]) * (pt[0] - p0[0]))

    upstream_sign = 1.0 if side(test) >= 0 else -1.0

    # Geração de grade
    yy, xx = np.mgrid[0:h, 0:w]
    P = np.stack([xx.astype(np.float32), yy.astype(np.float32)], axis=-1)

    # Distância assinada ao eixo da linha (normal)
    # signed_dist = dot((P - p0), n_unit)
    signed_dist = (P[..., 0] - p0[0]) * n_unit[0] + (P[..., 1] - p0[1]) * n_unit[1]

    # Mantém apenas o lado montante
    is_upstream = (signed_dist * upstream_sign) >= 0

    # Distância perpendicular (buffer)
    dist_perp = np.abs(signed_dist)
    width_px = float(width_pct * max(h, w))

    # Limite de comprimento a montante usando eixo "para cima" na imagem
    # eixo de escoamento assumido: para baixo (y+), então montante = y-.
    # Portanto, mede quanto cada ponto está acima do centro.
    up_axis = np.array([0.0, -1.0], dtype=np.float32)
    along_up = (P[..., 0] - c[0]) * up_axis[0] + (P[..., 1] - c[1]) * up_axis[1]

    length_px = float(length_pct * max(h, w))
    within_len = (along_up >= 0) & (along_up <= length_px)

    # Peso ao longo (cunha): 1 na paliçada -> 0 no fim do comprimento
    if length_px <= 1e-6:
        along_w = np.zeros((h, w), dtype=np.float32)
    else:
        along_w = 1.0 - np.clip(along_up / length_px, 0.0, 1.0)

    # Peso transversal: gaussiano (mais concentrado no eixo da paliçada)
    sigma = max(1.0, width_px)
    cross_w = np.exp(-(dist_perp * dist_perp) / (2.0 * sigma * sigma)).astype(np.float32)

    wgt = (along_w.astype(np.float32) * cross_w).astype(np.float32)
    wgt[~(is_upstream & within_len)] = 0.0
    return np.clip(wgt, 0.0, 1.0)


def _apply_deposition(
    elevation_cm: np.ndarray,
    nodata: np.ndarray,
    pal_cfgs: list[PalicadaCfg],
    sed_inc_by_pid_cm: dict[int, float | None],
    *,
    default_length_pct: float,
    mode: str,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    out = elevation_cm.copy()
    metrics: list[dict[str, object]] = []

    h, w = elevation_cm.shape

    for pal in PALICADAS:
        pid = pal[0]
        cfg = next((c for c in pal_cfgs if c.id == pid), None)
        if cfg is None:
            continue

        sed_inc = sed_inc_by_pid_cm.get(pid)
        remaining = max(0.0, cfg.crest_height_cm - cfg.current_fill_cm)

        applied = None
        overflow = None
        n_events_to_fill = None

        if sed_inc is not None:
            sed_inc = float(max(0.0, sed_inc))

            if sed_inc > 0:
                n_events_to_fill = float(remaining / sed_inc) if remaining > 0 else 0.0

            if mode == "capacity":
                # Simula a CUNHA MÁXIMA antes de extravasar
                applied = float(remaining)
                overflow = False
            else:
                # Um único evento com magnitude P90/P95
                applied = float(min(sed_inc, remaining))
                overflow = bool(sed_inc > remaining + 1e-9)

        length_pct = cfg.deposit_length_pct if cfg.deposit_length_pct is not None else default_length_pct
        wgt = _deposit_mask(shape_hw=(h, w), pal=pal, length_pct=length_pct)
        wgt[nodata] = 0.0

        deposit_area_px = int(np.count_nonzero(wgt > 0))
        deposit_mean_cm: float | None = None

        if applied is not None and applied > 0 and deposit_area_px > 0:
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
                "sed_inc_cm": sed_inc,
                "applied_peak_cm": applied,
                "deposit_mean_cm": deposit_mean_cm,
                "deposit_area_px": deposit_area_px,
                "overflow": overflow,
                "n_events_to_fill": n_events_to_fill,
                "deposit_length_pct": length_pct,
            }
        )

    return out, metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ravina 1: mapas simulados P90/P95.")
    p.add_argument("--geotiff", required=True, help="Imagem base do RECORTE (GeoTIFF/PNG com alpha).")
    p.add_argument("--out", required=True, help="PNG de saída (painéis a=P90, b=P95).")

    # Saídas opcionais para montagem editorial (painéis separados + escala)
    p.add_argument("--out-p90", required=False, help="PNG do painel P90 isolado (para montagem externa).")
    p.add_argument("--out-p95", required=False, help="PNG do painel P95 isolado (para montagem externa).")
    p.add_argument("--out-scale-json", required=False, help="JSON com vmin/vmax/cmap para colorbar externa.")

    p.add_argument("--palicadas-config", required=True, help="JSON com parâmetros por paliçada.")
    p.add_argument("--out-csv", required=False, help="CSV de métricas da simulação.")

    p.add_argument(
        "--runoff-coeff",
        type=float,
        default=None,
        help="Override do coeficiente de escoamento superficial (0–1). Se omitido, usa defaults do JSON e/ou regra VIB→runoff.",
    )

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

    # Visual
    p.add_argument("--max-grid", type=int, default=500)
    p.add_argument("--blur-radius", type=float, default=2.0)
    p.add_argument("--contour-interval", type=float, default=15.0)
    p.add_argument("--max-depth-cm", type=float, default=100.0)
    p.add_argument("--base-max-px", type=int, default=3000)
    p.add_argument("--deposit-length-pct", type=float, default=0.12, help="Comprimento a montante (fração 0–1).")
    p.add_argument(
        "--mode",
        choices=["capacity", "one-event"],
        default="one-event",
        help="capacity: preenche até a crista (capacidade máxima). one-event: aplica apenas um evento P90/P95.",
    )

    # Saída limpa (útil para compor figuras com painéis externos)
    p.add_argument(
        "--no-panel-titles",
        action="store_true",
        help="Remove títulos (a)/(b) dos painéis do PNG.",
    )
    p.add_argument(
        "--no-overlay-box",
        action="store_true",
        help="Remove a caixa de texto com P90/P95 e modo (ex.: capacity) sobre o mapa.",
    )
    p.add_argument(
        "--no-colorbar",
        action="store_true",
        help="Remove a barra de cores (recomendado se você vai montar painéis em outra figura).",
    )

    p.add_argument(
        "--colorbar-orientation",
        choices=["vertical", "horizontal"],
        default="vertical",
        help="Orientação da barra de cores quando habilitada.",
    )

    return p.parse_args()


def main() -> int:
    args = parse_args()

    geotiff_path = Path(args.geotiff)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg_path = Path(args.palicadas_config)
    pal_cfgs = _load_palicadas_config(cfg_path)

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
    runoff_coeff_default = float(args.runoff_coeff) if args.runoff_coeff is not None else runoff_coeff_default

    # Referência para modular sedimentação: assume que a calibração (razão FRACIONADO/RAINFALL)
    # corresponde, em média, ao coeff_at_ref do modelo VIB→runoff (ou ao default se não existir).
    runoff_coeff_ref_for_sed = runoff_coeff_default
    model_obj = cfg_obj.get("vib_to_runoff_model")
    if isinstance(model_obj, dict):
        try:
            runoff_coeff_ref_for_sed = float(model_obj.get("coeff_at_ref", runoff_coeff_ref_for_sed))
        except Exception:
            pass

    precip_series = Path(args.precip_series)
    sediment_data = Path(args.sediment_data)

    # Percentis de chuva e modelo empírico de sedimento
    p90_mm = _percentile_monthly_totals(precip_series, args.p90)
    p95_mm = _percentile_monthly_totals(precip_series, args.p95)

    ratio_by_area = _fit_sediment_ratio_by_area(sediment_data, fracionado_unit=str(args.fracionado_unit))

    # sed_inc por paliçada (usando area_group)
    sed_p90_by_pid: dict[int, float | None] = {}
    sed_p95_by_pid: dict[int, float | None] = {}
    sed_p90_base_by_pid: dict[int, float | None] = {}
    sed_p95_base_by_pid: dict[int, float | None] = {}
    sed_scale_by_pid: dict[int, float] = {}
    runoff_coeff_by_pid: dict[int, float] = {}
    runoff_coeff_source_by_pid: dict[int, str] = {}
    runoff_coeff_rule_by_pid: dict[int, str] = {}
    for cfg in pal_cfgs:
        ratio = ratio_by_area.get(cfg.area_group)
        if ratio is None:
            sed_p90_base_by_pid[cfg.id] = None
            sed_p95_base_by_pid[cfg.id] = None
            sed_p90_by_pid[cfg.id] = None
            sed_p95_by_pid[cfg.id] = None
            sed_scale_by_pid[cfg.id] = 1.0
            runoff_coeff_by_pid[cfg.id] = runoff_coeff_default
            runoff_coeff_source_by_pid[cfg.id] = "args" if args.runoff_coeff is not None else "defaults"
            runoff_coeff_rule_by_pid[cfg.id] = ""
        else:
            sed_p90_base = float(max(0.0, float(ratio) * p90_mm))
            sed_p95_base = float(max(0.0, float(ratio) * p95_mm))

            vib_cm_h = vib_by_area.get(str(cfg.area_group).strip().upper())

            if args.runoff_coeff is not None:
                runoff_coeff_used = float(args.runoff_coeff)
                runoff_source = "args"
                runoff_rule = ""
            else:
                runoff_coeff_used, runoff_source, runoff_rule = _runoff_coeff_from_vib(
                    vib_cm_h,
                    cfg_obj=cfg_obj,
                    fallback_coeff=runoff_coeff_default,
                )

            sed_scale = 1.0
            if runoff_coeff_ref_for_sed > 0:
                sed_scale = float(runoff_coeff_used) / float(runoff_coeff_ref_for_sed)

            sed_p90_base_by_pid[cfg.id] = sed_p90_base
            sed_p95_base_by_pid[cfg.id] = sed_p95_base
            sed_scale_by_pid[cfg.id] = float(sed_scale)
            runoff_coeff_by_pid[cfg.id] = float(runoff_coeff_used)
            runoff_coeff_source_by_pid[cfg.id] = str(runoff_source)
            runoff_coeff_rule_by_pid[cfg.id] = str(runoff_rule)

            # Acoplamento (mais forte): sed_inc_eff = sed_inc_base * (runoff_coeff_eff / runoff_coeff_ref)
            sed_p90_by_pid[cfg.id] = float(max(0.0, sed_p90_base * sed_scale))
            sed_p95_by_pid[cfg.id] = float(max(0.0, sed_p95_base * sed_scale))

    # Pseudo-DEM do recorte (mesma lógica do v2, mas sem ortomosaico)
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

    # Suavização adicional
    sigma_krig = max(4.0, args.blur_radius * 3.0)
    dem_fill = dem_norm.copy()
    if np.isnan(dem_fill).any():
        mean_dem = np.nanmean(dem_fill)
        if not np.isfinite(mean_dem):
            mean_dem = 0.5
        dem_fill[np.isnan(dem_fill)] = mean_dem

    dem_smooth = _gaussian_blur_separable(dem_fill, sigma=sigma_krig)
    dem_smooth[nodata] = np.nan

    # Normaliza e converte para cm (escala imposta)
    finite = np.isfinite(dem_smooth)
    if finite.any():
        vmin = float(np.nanmin(dem_smooth[finite]))
        vmax = float(np.nanmax(dem_smooth[finite]))
        if vmax > vmin:
            dem_smooth = (dem_smooth - vmin) / (vmax - vmin)
            dem_smooth = np.clip(dem_smooth, 0, 1)

    base_elev_cm = dem_smooth * args.max_depth_cm

    elev_p90_cm, met_p90 = _apply_deposition(
        base_elev_cm,
        nodata,
        pal_cfgs,
        sed_p90_by_pid,
        default_length_pct=float(args.deposit_length_pct),
        mode=str(args.mode),
    )
    elev_p95_cm, met_p95 = _apply_deposition(
        base_elev_cm,
        nodata,
        pal_cfgs,
        sed_p95_by_pid,
        default_length_pct=float(args.deposit_length_pct),
        mode=str(args.mode),
    )

    # Figura 1x2 somente mapas
    fig = plt.figure(figsize=(13, 6.5), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0])

    cmap_name = "Spectral_r"
    cmap = plt.get_cmap(cmap_name)

    # Escala dinâmica para evitar "travamento" em vmax fixo (ex.: capacity mode)
    finite1 = np.isfinite(elev_p90_cm)
    finite2 = np.isfinite(elev_p95_cm)
    vmax_data = float(
        np.nanmax(
            np.concatenate(
                [elev_p90_cm[finite1].ravel() if finite1.any() else np.array([0.0]), elev_p95_cm[finite2].ravel() if finite2.any() else np.array([0.0])]
            )
        )
    )
    vmax_plot = max(float(args.max_depth_cm), vmax_data)

    def draw_panel(
        ax: Any,
        elev_cm: np.ndarray,
        title: str,
        scenario_label: str,
        scenario_mm: float,
        metrics: list[dict[str, object]],
    ):
        ax.set_axis_off()
        if not args.no_panel_titles:
            ax.set_title(title, fontsize=12, y=1.0)

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

        # Contornos
        if elev_masked.count() > 0:
            levels = np.arange(0, vmax_plot + 1, args.contour_interval)
            cs = ax.contour(elev_masked, levels=levels, colors="black", linewidths=0.55, alpha=0.6)
            ax.clabel(cs, inline=True, fontsize=8, fmt="%d", colors="black")

        # Paliçadas
        for pid, x0, y0, x1, y1, _ in PALICADAS:
            X0, Y0 = pct_to_px(x0, y0, ww, hh)
            X1, Y1 = pct_to_px(x1, y1, ww, hh)
            ax.plot([X0, X1], [Y0, Y1], color="black", linewidth=2.0, alpha=0.75)

            m = next((mm for mm in metrics if mm.get("palicada") == f"P{pid}"), None)
            applied: float | None
            try:
                applied = float(m.get("applied_peak_cm")) if m is not None and m.get("applied_peak_cm") is not None else None
            except Exception:
                applied = None

            if applied is None or not np.isfinite(applied):
                continue

            pico = abs(float(applied))
            pico_s = f"{pico:.1f}".replace(".", ",")
            label = f"Pico – {pico_s} cm"

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

        if not args.no_overlay_box:
            ax.text(
                0.02,
                0.98,
                f"{scenario_label} mensal = {scenario_mm:.1f} mm\nmodo={args.mode}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                color="black",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75, edgecolor="none"),
            )

        return im_

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    im1 = draw_panel(ax1, elev_p90_cm, "(a) Simulação P90 — elevação relativa (cm)", "P90", p90_mm, met_p90)
    im2 = draw_panel(ax2, elev_p95_cm, "(b) Simulação P95 — elevação relativa (cm)", "P95", p95_mm, met_p95)

    # Saídas isoladas (para montagem editorial): painéis separados + escala para colorbar externa
    def _save_single_panel(elev_cm: np.ndarray, metrics: list[dict[str, object]], out_file: str) -> None:
        outp = Path(out_file)
        outp.parent.mkdir(parents=True, exist_ok=True)

        f = plt.figure(figsize=(6.5, 6.5), constrained_layout=True)
        ax = f.add_subplot(1, 1, 1)
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
        ax.imshow(elev_masked, cmap=cmap, origin="upper", vmin=0, vmax=vmax_plot, alpha=0.65)

        if elev_masked.count() > 0:
            levels = np.arange(0, vmax_plot + 1, args.contour_interval)
            cs = ax.contour(elev_masked, levels=levels, colors="black", linewidths=0.55, alpha=0.6)
            ax.clabel(cs, inline=True, fontsize=8, fmt="%d", colors="black")

        for pid, x0, y0, x1, y1, _ in PALICADAS:
            X0, Y0 = pct_to_px(x0, y0, ww, hh)
            X1, Y1 = pct_to_px(x1, y1, ww, hh)
            ax.plot([X0, X1], [Y0, Y1], color="black", linewidth=2.0, alpha=0.75)

            m = next((mm for mm in metrics if mm.get("palicada") == f"P{pid}"), None)
            try:
                applied = float(m.get("applied_peak_cm")) if m is not None and m.get("applied_peak_cm") is not None else None
            except Exception:
                applied = None
            if applied is None or not np.isfinite(applied):
                continue

            pico = abs(float(applied))
            label = f"Pico – {pico:.1f}".replace(".", ",") + " cm"
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

        f.savefig(outp, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(f)

    if args.out_p90:
        _save_single_panel(elev_p90_cm, met_p90, str(args.out_p90))
    if args.out_p95:
        _save_single_panel(elev_p95_cm, met_p95, str(args.out_p95))

    if args.out_scale_json:
        outj = Path(args.out_scale_json)
        outj.parent.mkdir(parents=True, exist_ok=True)
        outj.write_text(
            json.dumps(
                {
                    "vmin": 0.0,
                    "vmax": float(vmax_plot),
                    "cmap": cmap_name,
                    "label": "Elevação Relativa Total (Base + Sedimento) [cm]",
                    "mode": str(args.mode),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    # Colorbar único
    if not args.no_colorbar:
        if args.colorbar_orientation == "horizontal":
            cbar = plt.colorbar(im2, ax=[ax1, ax2], orientation="horizontal", fraction=0.06, pad=0.08, shrink=0.9)
            cbar.set_label("Elevação Relativa Total (Base + Sedimento) [cm]")
        else:
            cbar = plt.colorbar(im2, ax=[ax1, ax2], fraction=0.046, pad=0.04, shrink=0.75)
            cbar.set_label("Elevação Relativa Total (Base + Sedimento) [cm]", rotation=270, labelpad=15)

    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    # CSV
    if args.out_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)

        # Junta P90 e P95 por paliçada em um único registro por P1..P4
        rows_out: list[dict[str, object]] = []
        for pid in [1, 2, 3, 4]:
            k = f"P{pid}"
            m90 = next((m for m in met_p90 if m.get("palicada") == k), None)
            m95 = next((m for m in met_p95 if m.get("palicada") == k), None)
            if m90 is None and m95 is None:
                continue

            base = dict(m90 or m95)  # mantém campos estáveis
            area_group = base.get("area_group")
            vib_cm_h = None
            runoff_index_1_over_vib = None
            if area_group is not None:
                vib_cm_h = vib_by_area.get(str(area_group).strip().upper())
                if vib_cm_h is not None and vib_cm_h > 0:
                    runoff_index_1_over_vib = 1.0 / vib_cm_h

            # Usa exatamente os mesmos valores que entraram na simulação (sed_inc por pid)
            runoff_coeff_used = runoff_coeff_by_pid.get(pid, float(runoff_coeff_default))
            runoff_coeff_source = runoff_coeff_source_by_pid.get(
                pid,
                "args" if args.runoff_coeff is not None else "defaults",
            )
            runoff_coeff_rule = runoff_coeff_rule_by_pid.get(pid, "")
            sed_scale_runoff = sed_scale_by_pid.get(pid, 1.0)
            p90_sed_inc_cm_base = sed_p90_base_by_pid.get(pid)
            p95_sed_inc_cm_base = sed_p95_base_by_pid.get(pid)
            row: dict[str, object] = {
                "palicada": k,
                "area_group": area_group,
                "vib_cm_h": vib_cm_h,
                "runoff_index_1_over_vib": runoff_index_1_over_vib,
                "runoff_coeff": runoff_coeff_used,
                "runoff_coeff_source": runoff_coeff_source,
                "runoff_coeff_rule": runoff_coeff_rule,
                "sed_scale_runoff": sed_scale_runoff,
                "p90_sed_inc_cm_base": p90_sed_inc_cm_base,
                "p95_sed_inc_cm_base": p95_sed_inc_cm_base,
                "crest_height_cm": base.get("crest_height_cm"),
                "current_fill_cm": base.get("current_fill_cm"),
                "remaining_capacity_cm": base.get("remaining_cm"),
                "deposit_length_pct": base.get("deposit_length_pct"),
                "p90_mm": p90_mm,
                "p95_mm": p95_mm,
            }
            if m90 is not None:
                row.update(
                    {
                        "p90_sed_inc_cm": m90.get("sed_inc_cm"),
                        "p90_applied_peak_cm": m90.get("applied_peak_cm"),
                        "p90_deposit_mean_cm": m90.get("deposit_mean_cm"),
                        "p90_deposit_area_px": m90.get("deposit_area_px"),
                        "p90_overflow": m90.get("overflow"),
                        "p90_n_events_to_fill": m90.get("n_events_to_fill"),
                    }
                )
            if m95 is not None:
                row.update(
                    {
                        "p95_sed_inc_cm": m95.get("sed_inc_cm"),
                        "p95_applied_peak_cm": m95.get("applied_peak_cm"),
                        "p95_deposit_mean_cm": m95.get("deposit_mean_cm"),
                        "p95_deposit_area_px": m95.get("deposit_area_px"),
                        "p95_overflow": m95.get("overflow"),
                        "p95_n_events_to_fill": m95.get("n_events_to_fill"),
                    }
                )
            rows_out.append(row)

        fieldnames = [
            "palicada",
            "area_group",
            "vib_cm_h",
            "runoff_index_1_over_vib",
            "runoff_coeff",
            "runoff_coeff_source",
            "runoff_coeff_rule",
            "sed_scale_runoff",
            "p90_sed_inc_cm_base",
            "p95_sed_inc_cm_base",
            "crest_height_cm",
            "current_fill_cm",
            "remaining_capacity_cm",
            "deposit_length_pct",
            "p90_mm",
            "p95_mm",
            "p90_sed_inc_cm",
            "p90_applied_peak_cm",
            "p90_deposit_mean_cm",
            "p90_deposit_area_px",
            "p90_overflow",
            "p90_n_events_to_fill",
            "p95_sed_inc_cm",
            "p95_applied_peak_cm",
            "p95_deposit_mean_cm",
            "p95_deposit_area_px",
            "p95_overflow",
            "p95_n_events_to_fill",
        ]
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows_out)

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
