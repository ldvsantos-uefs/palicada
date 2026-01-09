from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SedimentacaoMetricas:
    max_cum_cm_sup: float
    max_cum_cm_med: float
    max_cum_cm_inf: float

    share_q1_pct: float
    share_q2_pct: float

    share_total_sup_pct: float
    share_total_med_pct: float
    share_total_inf_pct: float

    chuva_p50_mm: float
    chuva_p75_mm: float
    chuva_p90_mm: float
    chuva_p95_mm: float
    chuva_max_mm: float

    n_mod: int
    n_alto: int
    n_muito_alto: int
    n_extremo: int

    sed_mean_mod_cm: float
    sed_mean_alto_cm: float
    sed_mean_muito_alto_cm: float
    sed_mean_extremo_cm: float

    eff_alto_cm_per_mm: float
    eff_extremo_cm_per_mm: float
    eff_drop_pct: float

    dep_p90_cm: float
    dep_p95_cm: float
    n_dep_extremos: int
    share_dep_extremos_pct: float

    share_extremos_sup_pct: float
    share_extremos_med_pct: float
    share_extremos_inf_pct: float


def _read_sed_sheet(xlsx: Path, sheet: str, area: str) -> pd.DataFrame:
    df = pd.read_excel(xlsx, sheet_name=sheet)

    # Layouts variam entre abas.
    # - Em algumas (ex.: R5), DATA está em Unnamed: 1 e os Vergs em Unnamed: 2/3/4.
    # - Em outras (ex.: R4), DATA está em Unnamed: 1 e o primeiro Verg está na coluna
    #   nomeada como "Sedimentação ..." (seguida por Unnamed: 3/4).
    cols = list(df.columns)

    date_col: str | None = None
    if "Unnamed: 1" in cols:
        date_col = "Unnamed: 1"
    else:
        for c in cols:
            if isinstance(c, str) and c.lower().startswith("sedimenta"):
                date_col = c
                break
    if date_col is None:
        raise KeyError(f"Não foi possível detectar coluna de data na aba '{sheet}'.")

    idx = cols.index(date_col)
    candidate = cols[idx + 1 : idx + 8]
    rep_cols: list[str] = []
    for c in candidate:
        if isinstance(c, str) and c.lower().startswith("eros"):
            break
        rep_cols.append(c)
        if len(rep_cols) == 3:
            break
    if len(rep_cols) < 1:
        raise KeyError(f"Não foi possível detectar colunas de Verg na aba '{sheet}'.")

    out = df[[date_col, *rep_cols]].copy()
    out = out.rename(columns={date_col: "DATA"})
    out["DATA"] = pd.to_datetime(out["DATA"], errors="coerce")

    for c in rep_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna(subset=["DATA"])
    out = out.dropna(how="all", subset=rep_cols)

    out["SED_CM"] = out[rep_cols].mean(axis=1, skipna=True)
    out = out.dropna(subset=["SED_CM"])

    out["AREA"] = area

    # Alinha ao mês (1º dia) para compatibilidade com o CSV integrado.
    out["MES"] = out["DATA"].dt.to_period("M").dt.to_timestamp(how="start")

    # Se houver duplicata no mesmo mês (ex.: re-medição), usa a última.
    out = out.sort_values(["AREA", "DATA"]).drop_duplicates(subset=["AREA", "MES"], keep="last")

    out = out.sort_values(["AREA", "MES"])
    out["SEDIMENT"] = out["SED_CM"] / 100.0  # metros
    out["FRACIONADO"] = out.groupby("AREA")["SEDIMENT"].diff().fillna(out["SEDIMENT"])

    out["YEAR"] = out["MES"].dt.year.astype(int)
    out["MONTH"] = out["MES"].dt.month.astype(int)
    out["DATA"] = out["MES"].dt.strftime("%Y-%m-%d")

    return out[["AREA", "MONTH", "YEAR", "SEDIMENT", "FRACIONADO", "DATA"]]


def _monthly_rainfall_mm(root: Path) -> pd.DataFrame:
    daily = pd.read_csv(root / "2-DADOS" / "CLIMATOLOGIA_20ANOS" / "dados" / "serie_precipitacao_20anos.csv")
    daily["Data"] = pd.to_datetime(daily["Data"], errors="coerce")
    daily["Precipitacao_mm"] = pd.to_numeric(daily["Precipitacao_mm"], errors="coerce").fillna(0.0)
    daily = daily.dropna(subset=["Data"])

    monthly = (
        daily.assign(MES=daily["Data"].dt.to_period("M").dt.to_timestamp(how="start"))
        .groupby("MES", as_index=False)["Precipitacao_mm"]
        .sum()
        .rename(columns={"Precipitacao_mm": "RAINFALL"})
    )
    return monthly


def _monthly_ei30(root: Path) -> pd.DataFrame:
    ei30 = pd.read_csv(root / "2-DADOS" / "CLIMATOLOGIA_20ANOS" / "dados" / "ei30_mensal.csv")
    ei30["Data"] = pd.to_datetime(ei30["Data"], errors="coerce")
    ei30["precipitacao"] = pd.to_numeric(ei30["precipitacao"], errors="coerce")
    ei30 = ei30.dropna(subset=["Data", "precipitacao"])

    ei30 = ei30.assign(MES=ei30["Data"].dt.to_period("M").dt.to_timestamp(how="start"))
    ei30 = ei30.groupby("MES", as_index=False)["precipitacao"].sum().rename(columns={"precipitacao": "EI30"})
    return ei30


def _meteo_from_classificados(root: Path) -> pd.DataFrame:
    """Obtém RAINFALL e EI30 mensal da base já usada nas análises.

    Essa escolha é deliberada para manter consistência com as estatísticas e
    figuras já calibradas no projeto. O XLSX atualiza sedimentação; não deve
    redefinir a climatologia do período monitorado.
    """

    path = root / "2-DADOS" / "CLIMATOLOGIA_20ANOS" / "dados" / "dados_classificados_eventos.csv"
    if not path.exists():
        return pd.DataFrame(columns=["MES", "RAINFALL", "EI30"])

    df = pd.read_csv(path)
    df["DATA"] = pd.to_datetime(df["DATA"], errors="coerce")
    df = df.dropna(subset=["DATA"])

    # A precipitação mensal foi consolidada no segmento SUP.
    sup = df[df["AREA"] == "SUP"].copy()
    sup["RAINFALL"] = pd.to_numeric(sup["RAINFALL"], errors="coerce")
    sup["EI30"] = pd.to_numeric(sup["EI30"], errors="coerce")
    sup = sup.dropna(subset=["RAINFALL", "EI30"])
    sup["MES"] = sup["DATA"].dt.to_period("M").dt.to_timestamp(how="start")

    meteo = (
        sup.sort_values("DATA")
        .drop_duplicates(subset=["MES"], keep="last")
        [["MES", "RAINFALL", "EI30"]]
        .reset_index(drop=True)
    )
    return meteo


def _compute_metrics(df: pd.DataFrame) -> SedimentacaoMetricas:
    df = df.copy()
    df["MES"] = pd.to_datetime(df["DATA"], errors="coerce")

    df["FRAC_M"] = pd.to_numeric(df["FRACIONADO"], errors="coerce").fillna(0.0)
    df["SED_M"] = pd.to_numeric(df["SEDIMENT"], errors="coerce").fillna(0.0)
    df["RAINFALL"] = pd.to_numeric(df["RAINFALL"], errors="coerce").fillna(0.0)

    df["FRAC_CM_POS"] = (df["FRAC_M"].clip(lower=0.0) * 100.0)
    df["SED_CM"] = df["SED_M"] * 100.0

    # Máximos acumulados por segmento
    max_cum = df.groupby("AREA")["SED_CM"].max().to_dict()

    # Deposição total por segmento (incremental positiva)
    totals = df.groupby("AREA")["FRAC_CM_POS"].sum()
    total_system = float(totals.sum()) if float(totals.sum()) != 0.0 else 1.0

    # Sazonalidade trimestral (Q1/Q2) em toda a janela de monitoramento
    df["MONTH"] = pd.to_datetime(df["DATA"]).dt.month
    q1 = df[df["MONTH"].isin([1, 2, 3])]["FRAC_CM_POS"].sum()
    q2 = df[df["MONTH"].isin([4, 5, 6])]["FRAC_CM_POS"].sum()

    share_q1_pct = float(q1 / total_system * 100.0)
    share_q2_pct = float(q2 / total_system * 100.0)

    # Série mensal do sistema (soma dos segmentos) para classificação por chuva
    sys_month = (
        df.groupby("MES", as_index=False)
        .agg(RAINFALL=("RAINFALL", "mean"), SED_INC_CM=("FRAC_CM_POS", "sum"))
        .sort_values("MES")
    )

    # Quantis de chuva na janela monitorada (mensal)
    rain = sys_month["RAINFALL"].to_numpy(dtype=float)
    p50, p75, p90, p95 = np.percentile(rain, [50, 75, 90, 95])
    rmax = float(np.max(rain))

    def cls(v: float) -> str:
        if v < p50:
            return "BAIXO"
        if v < p75:
            return "MOD"
        if v < p90:
            return "ALTO"
        if v < p95:
            return "MUITO_ALTO"
        return "EXTREMO"

    sys_month["CLASSE"] = [cls(v) for v in sys_month["RAINFALL"]]

    # Tabela 3 apenas para classes >= P50 (como no texto)
    tab = sys_month[sys_month["CLASSE"].isin(["MOD", "ALTO", "MUITO_ALTO", "EXTREMO"])].copy()

    def mean_sed_cm(k: str) -> float:
        arr = tab.loc[tab["CLASSE"] == k, "SED_INC_CM"].to_numpy(dtype=float)
        return float(np.mean(arr)) if len(arr) else 0.0

    n_mod = int((tab["CLASSE"] == "MOD").sum())
    n_alto = int((tab["CLASSE"] == "ALTO").sum())
    n_muito_alto = int((tab["CLASSE"] == "MUITO_ALTO").sum())
    n_extremo = int((tab["CLASSE"] == "EXTREMO").sum())

    sed_mean_mod_cm = mean_sed_cm("MOD")
    sed_mean_alto_cm = mean_sed_cm("ALTO")
    sed_mean_muito_alto_cm = mean_sed_cm("MUITO_ALTO")
    sed_mean_extremo_cm = mean_sed_cm("EXTREMO")

    # Eficiência (cm/mm), usando meses com chuva > 0 para evitar divisão por zero
    tab_eff = tab[tab["RAINFALL"] > 0.0].copy()
    tab_eff["EFF"] = tab_eff["SED_INC_CM"] / tab_eff["RAINFALL"]

    eff_alto = tab_eff.loc[tab_eff["CLASSE"] == "ALTO", "EFF"].mean()
    eff_ext = tab_eff.loc[tab_eff["CLASSE"] == "EXTREMO", "EFF"].mean()

    eff_alto = float(eff_alto) if pd.notna(eff_alto) else 0.0
    eff_ext = float(eff_ext) if pd.notna(eff_ext) else 0.0

    eff_drop_pct = float((1.0 - (eff_ext / eff_alto)) * 100.0) if eff_alto > 0.0 else 0.0

    # Extremos deposicionais (P90/P95) sobre a distribuição de incrementos mensais do sistema
    dep_pos = sys_month["SED_INC_CM"].to_numpy(dtype=float)
    dep_pos = dep_pos[dep_pos > 0.0]
    dep_p90 = float(np.percentile(dep_pos, 90)) if len(dep_pos) else 0.0
    dep_p95 = float(np.percentile(dep_pos, 95)) if len(dep_pos) else 0.0

    sys_month["DEP_EXTREMO"] = sys_month["SED_INC_CM"] >= dep_p95
    n_dep_extremos = int(sys_month["DEP_EXTREMO"].sum())

    total_dep = float(sys_month["SED_INC_CM"].sum()) if float(sys_month["SED_INC_CM"].sum()) != 0.0 else 1.0
    share_dep_extremos_pct = float(sys_month.loc[sys_month["DEP_EXTREMO"], "SED_INC_CM"].sum() / total_dep * 100.0)

    # Participação dos meses deposicionais extremos por segmento
    extreme_months = set(sys_month.loc[sys_month["DEP_EXTREMO"], "MES"].to_list())
    df["DEP_EXTREMO"] = df["MES"].isin(extreme_months)

    share_extremos = {}
    for area in ["SUP", "MED", "INF"]:
        sub = df[df["AREA"] == area]
        denom = float(sub["FRAC_CM_POS"].sum())
        if denom <= 0.0:
            share_extremos[area] = 0.0
        else:
            share_extremos[area] = float(sub.loc[sub["DEP_EXTREMO"], "FRAC_CM_POS"].sum() / denom * 100.0)

    return SedimentacaoMetricas(
        max_cum_cm_sup=float(max_cum.get("SUP", 0.0)),
        max_cum_cm_med=float(max_cum.get("MED", 0.0)),
        max_cum_cm_inf=float(max_cum.get("INF", 0.0)),
        share_q1_pct=share_q1_pct,
        share_q2_pct=share_q2_pct,
        share_total_sup_pct=float(totals.get("SUP", 0.0) / total_system * 100.0),
        share_total_med_pct=float(totals.get("MED", 0.0) / total_system * 100.0),
        share_total_inf_pct=float(totals.get("INF", 0.0) / total_system * 100.0),
        chuva_p50_mm=float(p50),
        chuva_p75_mm=float(p75),
        chuva_p90_mm=float(p90),
        chuva_p95_mm=float(p95),
        chuva_max_mm=rmax,
        n_mod=n_mod,
        n_alto=n_alto,
        n_muito_alto=n_muito_alto,
        n_extremo=n_extremo,
        sed_mean_mod_cm=sed_mean_mod_cm,
        sed_mean_alto_cm=sed_mean_alto_cm,
        sed_mean_muito_alto_cm=sed_mean_muito_alto_cm,
        sed_mean_extremo_cm=sed_mean_extremo_cm,
        eff_alto_cm_per_mm=eff_alto,
        eff_extremo_cm_per_mm=eff_ext,
        eff_drop_pct=eff_drop_pct,
        dep_p90_cm=dep_p90,
        dep_p95_cm=dep_p95,
        n_dep_extremos=n_dep_extremos,
        share_dep_extremos_pct=share_dep_extremos_pct,
        share_extremos_sup_pct=float(share_extremos.get("SUP", 0.0)),
        share_extremos_med_pct=float(share_extremos.get("MED", 0.0)),
        share_extremos_inf_pct=float(share_extremos.get("INF", 0.0)),
    )


def main() -> int:
    root = Path(__file__).resolve().parents[3]

    xlsx = root / "2-DADOS" / "Dados de Sedimentação - TCC e EE_2.xlsx"
    if not xlsx.exists():
        raise FileNotFoundError(f"XLSX não encontrado: {xlsx}")

    out_csv = root / "2-DADOS" / "CLIMATOLOGIA_20ANOS" / "dados" / "dados_integrados_sedimentacao.csv"

    # Mapeamento de abas para os segmentos usados no manuscrito
    parts = [
        _read_sed_sheet(xlsx, "PSup R4", "SUP"),
        _read_sed_sheet(xlsx, "PInt R4", "MED"),
        _read_sed_sheet(xlsx, "PInf R4", "INF"),
    ]
    sed = pd.concat(parts, ignore_index=True)

    # Junta chuva e EI30 por mês (preservando série já usada nas análises)
    sed["MES"] = pd.to_datetime(sed["DATA"], errors="coerce")

    meteo = _meteo_from_classificados(root)
    if len(meteo) == 0:
        # Fallback: se a base classificada não existir, recompõe por série diária.
        rain = _monthly_rainfall_mm(root)
        ei30 = _monthly_ei30(root)
        meteo = rain.merge(ei30, on="MES", how="outer")

    sed = sed.merge(meteo, on="MES", how="left")

    # Mantém compatibilidade com colunas legadas
    sed["RAINFALL"] = pd.to_numeric(sed["RAINFALL"], errors="coerce")
    sed["EI30"] = pd.to_numeric(sed["EI30"], errors="coerce")

    out = sed[["AREA", "MONTH", "YEAR", "SEDIMENT", "RAINFALL", "FRACIONADO", "DATA", "EI30"]].copy()

    area_order = {"SUP": 0, "MED": 1, "INF": 2}
    out["_AREA_ORDER"] = out["AREA"].map(area_order).fillna(999).astype(int)
    out = (
        out.sort_values(["_AREA_ORDER", "YEAR", "MONTH"], kind="mergesort")
        .drop(columns=["_AREA_ORDER"])
        .reset_index(drop=True)
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    metrics = _compute_metrics(out)
    metrics_path = root / "1-MANUSCRITOS" / "1-CONTROLE_PLITOSSOLO" / "media" / "analises_estatisticas" / "metricas_sedimentacao_atualizadas.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(asdict(metrics), ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"OK: CSV atualizado em {out_csv}")
    print(f"OK: métricas salvas em {metrics_path}")
    print(json.dumps(asdict(metrics), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
