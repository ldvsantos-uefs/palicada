from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt


ROOT = Path(r"C:\Users\vidal\OneDrive\Documentos\13 - CLONEGIT\artigo-posdoc\3-EROSIBIDADE")

MORF_XLSX = ROOT / r"2-DADOS\PLANILHA DE COLETA DE DADOS DE RAVINAS E VOÇOROCAS (1).xlsx"
SHEET = "Table 2"

OUT_PNG = (
    ROOT
    / r"1-MANUSCRITOS\2-CARACTERIZACAO_FEICAO\media\fig_S1_pca_morfometria.png"
)


def _to_float(value: object) -> float:
    s = str(value).strip()
    if s == "" or s.lower() == "nan":
        return float("nan")
    # fix common typos
    s = s.replace("O", "0").replace("o", "0").replace("l", "1").replace("I", "1")
    s = s.replace(",", ".")
    # remove replicate prefixes like 1-0.75, 2-1.10
    s = re.sub(r"\b\d\s*-\s*(?=\d)", "", s)
    nums = [float(m.group(1)) for m in re.finditer(r"(\d+(?:\.\d+)?)", s)]
    return float(np.mean(nums)) if nums else float("nan")


def _extract_morfometria() -> pd.DataFrame:
    df = pd.read_excel(MORF_XLSX, sheet_name=SHEET, dtype=str)
    df = df.iloc[1:].copy()  # drop header-description row

    # known layout for this sheet
    col_feicao = "Ravina (Num)"
    col_comp = "Comprimento (montante/ Jusante) (m)"

    # segment columns by position (from earlier inspection)
    col_larg_med_sup = df.columns[5]
    col_alt_sup = df.columns[7]

    col_larg_med_med = df.columns[9]
    col_alt_med = df.columns[11]

    col_larg_med_inf = df.columns[13]
    col_alt_inf = df.columns[15]

    rows: list[dict[str, float]] = []
    for _, r in df.iterrows():
        feicao = _to_float(r.get(col_feicao))
        if np.isnan(feicao):
            continue

        comprimento = _to_float(r.get(col_comp))

        largura_media = np.nanmean(
            [
                _to_float(r.get(col_larg_med_sup)),
                _to_float(r.get(col_larg_med_med)),
                _to_float(r.get(col_larg_med_inf)),
            ]
        )

        prof_max = np.nanmax(
            [
                _to_float(r.get(col_alt_sup)),
                _to_float(r.get(col_alt_med)),
                _to_float(r.get(col_alt_inf)),
            ]
        )

        rows.append(
            {
                "feicao": int(round(feicao)),
                "comprimento_m": comprimento,
                "largura_media_m": largura_media,
                "prof_max_m": prof_max,
            }
        )

    out = pd.DataFrame(rows).drop_duplicates(subset=["feicao"]).set_index("feicao")
    return out


def main() -> None:
    out = _extract_morfometria()

    # PCA requires complete rows
    cols = ["comprimento_m", "largura_media_m", "prof_max_m"]
    complete = out.dropna(subset=cols).copy()

    if len(complete) < 3:
        raise SystemExit(f"Poucas feições completas para PCA (n={len(complete)}).")

    X = complete[cols].to_numpy(dtype=float)
    Xz = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2, random_state=0)
    scores = pca.fit_transform(Xz)

    # loadings for biplot (components_ are unit vectors in feature space)
    loadings = pca.components_.T

    explained = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=200)

    ax.scatter(scores[:, 0], scores[:, 1], s=55)

    for (feicao, (x, y)) in zip(complete.index.tolist(), scores, strict=False):
        ax.text(x + 0.04, y + 0.04, str(feicao), fontsize=10)

    # scale arrows to match score space
    arrow_scale = 1.8
    for i, var in enumerate(cols):
        ax.arrow(
            0,
            0,
            loadings[i, 0] * arrow_scale,
            loadings[i, 1] * arrow_scale,
            head_width=0.06,
            head_length=0.08,
            linewidth=1.2,
            length_includes_head=True,
        )
        ax.text(
            loadings[i, 0] * (arrow_scale + 0.12),
            loadings[i, 1] * (arrow_scale + 0.12),
            var.replace("_m", ""),
            fontsize=10,
        )

    ax.axhline(0, linewidth=0.8)
    ax.axvline(0, linewidth=0.8)

    ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}%)")
    ax.set_title("PCA (morfometria): feições com dados completos")

    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG)
    plt.close(fig)

    print(f"OK: {OUT_PNG}")


if __name__ == "__main__":
    main()
