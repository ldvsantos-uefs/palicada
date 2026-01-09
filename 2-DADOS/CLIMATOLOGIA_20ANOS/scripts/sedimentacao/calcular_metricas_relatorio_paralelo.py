"""\
Calcula métricas descritivas para RELATORIO_PARALELO_SEDIMENTOS.md.

Entrada
- CLIMATOLOGIA_20ANOS/dados/dados_integrados_sedimentacao.csv

Saída
- stdout: resumo em texto/markdown
- arquivo: RELATORIO_PARALELO_SEDIMENTOS_media/metricas_relatorio_paralelo.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> tuple[float | None, float | None]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3:
        return None, None
    r, p = stats.pearsonr(x, y)
    return float(r), float(p)


def _safe_r2(x: np.ndarray, y: np.ndarray) -> float | None:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3:
        return None
    model = LinearRegression().fit(x.reshape(-1, 1), y)
    return float(model.score(x.reshape(-1, 1), y))


def main() -> None:
    repo_dir = Path(__file__).resolve().parents[3]
    csv_path = repo_dir / "CLIMATOLOGIA_20ANOS" / "dados" / "dados_integrados_sedimentacao.csv"

    df = pd.read_csv(csv_path)
    df["DATA"] = pd.to_datetime(df["DATA"], errors="coerce")

    # No CSV integrado, a coluna RAINFALL está preenchida somente para SUP.
    # Para análises por segmento, tratar precipitação e EI30 como covariáveis externas
    # comuns aos segmentos, vinculadas por DATA.
    df = df.dropna(subset=["AREA", "DATA", "FRACIONADO", "EI30"]).copy()

    meteo = (
        df[df["AREA"] == "SUP"][["DATA", "RAINFALL", "EI30"]]
        .dropna(subset=["DATA", "RAINFALL", "EI30"])
        .sort_values("DATA")
        .drop_duplicates(subset=["DATA"], keep="last")
    )
    df = df.merge(meteo[["DATA", "RAINFALL", "EI30"]], on="DATA", how="left", suffixes=("", "_METEO"))
    # Preferir a série meteo por DATA (RAINFALL só existe plenamente em SUP no CSV)
    if "RAINFALL_METEO" in df.columns:
        df["RAINFALL"] = df["RAINFALL_METEO"].combine_first(df.get("RAINFALL"))
    if "EI30_METEO" in df.columns:
        df["EI30"] = df["EI30_METEO"].combine_first(df.get("EI30"))
    df = df.drop(columns=[c for c in ["RAINFALL_METEO", "EI30_METEO"] if c in df.columns])
    df = df.dropna(subset=["RAINFALL", "EI30"]).copy()

    # Para análises deposicionais, considerar apenas incrementos positivos
    df["FRAC_POS"] = df["FRACIONADO"].clip(lower=0)

    # Sinal negativo indica meses com balanço erosivo/remoção (útil como evidência complementar)
    neg_counts = df.groupby("AREA")["FRACIONADO"].apply(lambda s: int((s < 0).sum()))
    neg_pct = (neg_counts / df.groupby("AREA")["FRACIONADO"].count() * 100).round(2)

    # Lags (condições antecedentes simples)
    df = df.sort_values(["AREA", "DATA"])
    df["RAINFALL_LAG1"] = df.groupby("AREA")["RAINFALL"].shift(1)
    df["EI30_LAG1"] = df.groupby("AREA")["EI30"].shift(1)

    period_min = df["DATA"].min()
    period_max = df["DATA"].max()

    # Contribuição por segmento
    totals = df.groupby("AREA")["FRAC_POS"].sum().sort_values(ascending=False)
    total_all = float(totals.sum())
    contrib_pct = (totals / total_all * 100).round(2)

    # Sazonalidade por trimestre
    df["QUARTER"] = df["DATA"].dt.to_period("Q").astype(str)
    df["QNUM"] = df["DATA"].dt.quarter

    q_totals = df.groupby("QNUM")["FRAC_POS"].sum().sort_index()
    q_pct = (q_totals / q_totals.sum() * 100).round(2)

    # Extremos (percentis globais em FRAC_POS)
    p90 = float(df["FRAC_POS"].quantile(0.90))
    p95 = float(df["FRAC_POS"].quantile(0.95))

    df["EXT_P90"] = df["FRAC_POS"] >= p90
    df["EXT_P95"] = df["FRAC_POS"] >= p95

    contrib_p95 = float(df.loc[df["EXT_P95"], "FRAC_POS"].sum() / df["FRAC_POS"].sum() * 100) if df["FRAC_POS"].sum() > 0 else 0.0
    count_p95 = int(df["EXT_P95"].sum())

    contrib_p95_by_area = (
        df[df["EXT_P95"]].groupby("AREA")["FRAC_POS"].sum()
        / df.groupby("AREA")["FRAC_POS"].sum()
        * 100
    ).round(2).replace([np.inf, -np.inf], np.nan).dropna()

    # Correlações e R²
    metrics_corr = {}
    for key, xcol in [
        ("chuva", "RAINFALL"),
        ("ei30", "EI30"),
        ("chuva_lag1", "RAINFALL_LAG1"),
        ("ei30_lag1", "EI30_LAG1"),
    ]:
        r, p = _safe_pearson(df[xcol].to_numpy(float), df["FRAC_POS"].to_numpy(float))
        r2 = _safe_r2(df[xcol].to_numpy(float), df["FRAC_POS"].to_numpy(float))
        metrics_corr[key] = {"r": r, "p": p, "r2": r2}

    metrics_corr_by_area = {}
    for area, g in df.groupby("AREA"):
        area_metrics = {}
        for key, xcol in [
            ("chuva", "RAINFALL"),
            ("ei30", "EI30"),
            ("chuva_lag1", "RAINFALL_LAG1"),
            ("ei30_lag1", "EI30_LAG1"),
        ]:
            r, p = _safe_pearson(g[xcol].to_numpy(float), g["FRAC_POS"].to_numpy(float))
            r2 = _safe_r2(g[xcol].to_numpy(float), g["FRAC_POS"].to_numpy(float))
            area_metrics[key] = {"r": r, "p": p, "r2": r2}
        metrics_corr_by_area[str(area)] = area_metrics

    out = {
        "n": int(len(df)),
        "periodo": {"inicio": str(period_min.date()), "fim": str(period_max.date())},
        "contribuicao_segmentos_pct": contrib_pct.to_dict(),
        "trimestre_contribuicao_pct": q_pct.to_dict(),
        "meses_fracionado_negativo": {
            "contagem_por_segmento": neg_counts.to_dict(),
            "percentual_por_segmento": neg_pct.to_dict(),
        },
        "extremos": {
            "p90_fracionado_pos": p90,
            "p95_fracionado_pos": p95,
            "contagem_p95": count_p95,
            "contribuicao_p95_pct": round(contrib_p95, 2),
            "contribuicao_p95_por_segmento_pct": contrib_p95_by_area.to_dict(),
        },
        "correlacoes_globais": metrics_corr,
        "correlacoes_por_segmento": metrics_corr_by_area,
    }

    out_dir = repo_dir / "RELATORIO_PARALELO_SEDIMENTOS_media"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "metricas_relatorio_paralelo.json"
    json_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # Saída resumida
    print("MÉTRICAS - RELATÓRIO PARALELO (SEDIMENTAÇÃO)\n")
    print(f"Período: {out['periodo']['inicio']} a {out['periodo']['fim']}  |  n = {out['n']}")
    print("\nContribuição por segmento (% do FRACIONADO positivo):")
    for k, v in out["contribuicao_segmentos_pct"].items():
        print(f"  {k}: {v}%")

    print("\nMeses com FRACIONADO negativo (contagem e %):")
    for k in out["meses_fracionado_negativo"]["contagem_por_segmento"].keys():
        c = out["meses_fracionado_negativo"]["contagem_por_segmento"][k]
        p = out["meses_fracionado_negativo"]["percentual_por_segmento"][k]
        print(f"  {k}: {c} ({p}%)")

    print("\nContribuição por trimestre (% do FRACIONADO positivo):")
    for k, v in out["trimestre_contribuicao_pct"].items():
        print(f"  Q{k}: {v}%")

    print("\nExtremos (FRACIONADO positivo):")
    print(f"  P90 = {out['extremos']['p90_fracionado_pos']:.4f} cm")
    print(f"  P95 = {out['extremos']['p95_fracionado_pos']:.4f} cm")
    print(f"  Eventos ≥ P95: {out['extremos']['contagem_p95']}")
    print(f"  Contribuição dos eventos ≥ P95: {out['extremos']['contribuicao_p95_pct']}% do total")

    print("\nCorrelação global (Pearson) com FRACIONADO positivo:")
    for key, m in out["correlacoes_globais"].items():
        r = m["r"]
        p = m["p"]
        r2 = m["r2"]
        if r is None:
            continue
        print(f"  {key}: r={r:.3f} p={p:.4f} R²={r2:.3f}")

    print(f"\nArquivo salvo: {json_path}")


if __name__ == "__main__":
    main()
