from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Módulo 1: Carregar propriedades do bambu
def load_bamboo_properties(csv_path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return {
            # Valores de referência típicos na literatura (ordem de grandeza)
            # Mantemos conservador e totalmente explícito no output.
            "Density": 0.68,
            "Tensile_Strength": 180.0,
            "Modulus_Elasticity": 12.0,
            "_source": "default_literature_typical_range",
        }

    df = pd.read_csv(csv_path)
    cols = {
        "Density": ["Density", "density", "densidade"],
        "Tensile_Strength": ["Tensile_Strength", "tensile_strength", "ft", "resistencia_tracao"],
        "Modulus_Elasticity": ["Modulus_Elasticity", "modulus_elasticity", "E", "modulo_elasticidade"],
    }
    picked = {}
    for key, candidates in cols.items():
        found = next((c for c in candidates if c in df.columns), None)
        if found is None:
            raise KeyError(f"Coluna não encontrada para {key}. Esperado uma de: {candidates}. Colunas: {list(df.columns)}")
        picked[key] = df[found].astype(float).mean()

    picked["_source"] = str(csv_path)
    return picked

# Módulo 2: Modelo de degradação
def degradation_model(t_years: np.ndarray, k_year: float) -> np.ndarray:
    """Modelo simples de perda de resistência: fator residual exp(-k*t)."""
    return np.exp(-k_year * t_years)

# Módulo 3: Simulação MEF
def run_strength_scenarios(params: dict, years: int = 10) -> pd.DataFrame:
    """Gera cenários de resistência residual ao longo do tempo.

    Observação: isto NÃO é um MEF completo; é um modelo temporal paramétrico
    para dar suporte quantitativo ao texto do artigo.
    """

    t = np.arange(0, years + 1, dtype=float)
    tensile0 = float(params["Tensile_Strength"])

    # Cenários de degradação anual (k) — ajustáveis quando houver dados locais.
    scenarios = {
        "optimistic": 0.03,
        "baseline": 0.06,
        "pessimistic": 0.10,
    }

    rows: list[dict] = []
    for scenario, k_year in scenarios.items():
        factor = degradation_model(t, k_year=k_year)
        residual_strength = tensile0 * factor
        for year, rs, f in zip(t.astype(int), residual_strength, factor, strict=True):
            rows.append(
                {
                    "year": int(year),
                    "scenario": scenario,
                    "k_year": float(k_year),
                    "strength_factor": float(f),
                    "tensile_strength_mpa": float(rs),
                }
            )

    meta = {
        "density_g_cm3": float(params.get("Density", np.nan)),
        "modulus_elasticity_gpa": float(params.get("Modulus_Elasticity", np.nan)),
        "source": params.get("_source", "unknown"),
    }
    df = pd.DataFrame(rows)
    for k, v in meta.items():
        df[k] = v
    return df

if __name__ == '__main__':
    repo_root = Path(__file__).resolve().parents[1]

    input_csv = repo_root / "2-DADOS" / "CLIMATOLOGIA_20ANOS" / "dados" / "bamboo_properties.csv"
    out_dir = repo_root / "2-DADOS" / "simulacoes"
    out_dir.mkdir(parents=True, exist_ok=True)

    props = load_bamboo_properties(input_csv)
    df = run_strength_scenarios(props, years=10)
    out_csv = out_dir / "resistencia_temporal.csv"
    df.to_csv(out_csv, index=False)
    print(f"✓ Salvo: {out_csv}")