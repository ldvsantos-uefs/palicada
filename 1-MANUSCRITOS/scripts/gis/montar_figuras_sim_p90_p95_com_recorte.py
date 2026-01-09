#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Monta figuras 3-painéis (a)(b)(c) a partir de PNGs existentes.

Objetivo editorial
- Remover a necessidade de uma figura separada apenas para o recorte.
- Inserir o recorte ortomosaico como painel (a) nas figuras de simulação.

Entradas (já existentes em media/figuras)
- fig_ravina1_recorte_ab_v2.png  -> usa-se apenas o painel (a) (metade esquerda)
- fig_ravina1_mapas_sim_p90_p95_...one_event.png -> recorta em 2 (P90/P95)
- fig_ravina1_mapas_sim_p90_p95_...capacity.png  -> recorta em 2 (P90/P95)

Saídas (sobrescreve)
- fig_ravina1_mapas_sim_p90_p95_recorte_tif_one_event.png
- fig_ravina1_mapas_sim_p90_p95_recorte_tif_capacity.png
"""

from __future__ import annotations

from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from PIL import Image


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
    """Recorta bordas quase brancas/transparentes para evitar 'molduras'."""
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


def _find_colorbar_bottom(arr: np.ndarray) -> tuple[int, int] | None:
    """Encontra (y0,y1) da colorbar horizontal no rodapé (heurístico)."""
    if arr.ndim != 3 or arr.shape[2] < 4:
        return None

    h, w = arr.shape[:2]
    y_start = int(h * 0.70)
    sub = arr[y_start:, :, :]

    rgb = sub[:, :, :3].astype(np.int16)
    alpha = sub[:, :, 3].astype(np.int16)

    not_transparent = alpha > 5
    not_white = (rgb.mean(axis=2) < 245)
    mask = not_transparent & not_white

    if not mask.any():
        return None

    row_counts = mask.sum(axis=1)
    thresh = int(w * 0.15)
    rows = np.where(row_counts > thresh)[0]
    if rows.size == 0:
        return None

    y0 = int(rows.min())
    y1 = int(rows.max()) + 1
    # Expande um pouco para capturar rótulos/traços
    y0 = max(y0 - 2, 0)
    y1 = min(y1 + 2, sub.shape[0])
    return y_start + y0, y_start + y1


def _make_triptych(
    *,
    recorte_a: np.ndarray,
    p90: np.ndarray,
    p95: np.ndarray,
    scale: dict | None,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if scale is None:
        fig = plt.figure(figsize=(18, 6))
        gs = fig.add_gridspec(1, 3, wspace=0.0)
        ax_a = fig.add_subplot(gs[0, 0])
        ax_b = fig.add_subplot(gs[0, 1])
        ax_c = fig.add_subplot(gs[0, 2])
        axes = [ax_a, ax_b, ax_c]
        fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.0)
    else:
        fig = plt.figure(figsize=(18, 6.6))
        gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.06], wspace=0.0, hspace=0.10)
        ax_a = fig.add_subplot(gs[0, 0])
        ax_b = fig.add_subplot(gs[0, 1])
        ax_c = fig.add_subplot(gs[0, 2])
        axes = [ax_a, ax_b, ax_c]

        ax_cb_blank = fig.add_subplot(gs[1, 0])
        ax_cb_blank.set_axis_off()
        ax_cb = fig.add_subplot(gs[1, 1:3])

        cmap_name = str(scale.get("cmap", "Spectral_r"))
        vmin = float(scale.get("vmin", 0.0))
        vmax = float(scale.get("vmax", 100.0))
        label = str(scale.get("label", ""))
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap_name))
        cbar = fig.colorbar(sm, cax=ax_cb, orientation="horizontal")
        if label:
            cbar.set_label(label)

        fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.08, wspace=0.0, hspace=0.10)

    for ax, img, label in zip(axes, [recorte_a, p90, p95], ["(a)", "(b)", "(c)"]):
        ax.imshow(img)
        ax.set_axis_off()
        # rótulo dentro do painel para evitar 'risco' por recorte
        ax.text(
            0.01,
            0.99,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=16,
            fontweight="bold",
            color="black",
        )

    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    root_dir = Path(__file__).resolve().parents[3]
    fig_dir = root_dir / "1-MANUSCRITOS" / "1-CONTROLE_PLITOSSOLO" / "media" / "figuras"

    recorte_ab = fig_dir / "fig_ravina1_recorte_ab_v2.png"
    one_event = fig_dir / "fig_ravina1_mapas_sim_p90_p95_recorte_tif_one_event.png"
    capacity = fig_dir / "fig_ravina1_mapas_sim_p90_p95_recorte_tif_capacity.png"

    if not recorte_ab.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {recorte_ab}")

    recorte_arr = _open_rgba(recorte_ab)
    recorte_a = _crop_half(recorte_arr, side="left")
    recorte_a = _trim_whitespace(recorte_a)

    # Preferir painéis isolados (evita recortes que geram 'figuras atrás')
    one_p90_path = fig_dir / f"{one_event.stem}_P90.png"
    one_p95_path = fig_dir / f"{one_event.stem}_P95.png"
    one_scale_path = fig_dir / f"{one_event.stem}_scale.json"

    cap_p90_path = fig_dir / f"{capacity.stem}_P90.png"
    cap_p95_path = fig_dir / f"{capacity.stem}_P95.png"
    cap_scale_path = fig_dir / f"{capacity.stem}_scale.json"

    if not one_p90_path.exists() or not one_p95_path.exists():
        raise FileNotFoundError(
            "Painéis isolados não encontrados para one_event. Gere arquivos: "
            f"{one_p90_path.name} e {one_p95_path.name}"
        )
    if not cap_p90_path.exists() or not cap_p95_path.exists():
        raise FileNotFoundError(
            "Painéis isolados não encontrados para capacity. Gere arquivos: "
            f"{cap_p90_path.name} e {cap_p95_path.name}"
        )

    one_p90 = _trim_whitespace(_open_rgba(one_p90_path))
    one_p95 = _trim_whitespace(_open_rgba(one_p95_path))
    cap_p90 = _trim_whitespace(_open_rgba(cap_p90_path))
    cap_p95 = _trim_whitespace(_open_rgba(cap_p95_path))

    one_scale = json.loads(one_scale_path.read_text(encoding="utf-8")) if one_scale_path.exists() else None
    cap_scale = json.loads(cap_scale_path.read_text(encoding="utf-8")) if cap_scale_path.exists() else None

    _make_triptych(recorte_a=recorte_a, p90=one_p90, p95=one_p95, scale=one_scale, out_path=one_event)
    print(f"✅ Atualizada (3 painéis): {one_event}")

    _make_triptych(recorte_a=recorte_a, p90=cap_p90, p95=cap_p95, scale=cap_scale, out_path=capacity)
    print(f"✅ Atualizada (3 painéis): {capacity}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
