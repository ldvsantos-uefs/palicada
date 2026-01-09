from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Configuração estética
sns.set(style='whitegrid', palette='muted')

# Gerar figura de eficiência
def plot_efficiency(data_path, output_path):
    data_path = Path(data_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    # Compatibilidade com o CSV gerado em analise_robusta_palicadas.py
    if "years" in df.columns and "residual_strength" in df.columns:
        df = df.rename(columns={"years": "year", "residual_strength": "tensile_strength_mpa"})

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df,
        x="year",
        y="tensile_strength_mpa",
        hue="scenario" if "scenario" in df.columns else None,
    )
    plt.xlabel("Ano")
    plt.ylabel("Resistência à tração residual (MPa)")
    plt.title("Evolução da resistência residual do bambu (cenários)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

if __name__ == '__main__':
    repo_root = Path(__file__).resolve().parents[1]
    plot_efficiency(
        repo_root / "2-DADOS" / "simulacoes" / "resistencia_temporal.csv",
        repo_root / "1-MANUSCRITOS" / "1-CONTROLE_PLITOSSOLO" / "media" / "figuras" / "resistencia_temporal.png",
    )
    print("✓ Figura salva: resistencia_temporal.png")