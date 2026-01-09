#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""resumo_relatorio_drone.py

Extrai métricas principais de um PDF de relatório de fotogrametria (ex.: Agisoft/Pix4D)
para gerar um resumo reutilizável no manuscrito (Tabela/Apêndice).

Saídas:
- Markdown (.md) com um quadro-resumo
- JSON (.json) com os valores extraídos (para rastreabilidade)

Uso:
  python scripts\\gis\\resumo_relatorio_drone.py \
    --pdf "2-DADOS\\DADOS\\REPORT ORTOMOSAICO DAS RAVINAS.pdf" \
    --out-md "1-CONTROLE_PLITOSSOLO\\media\\dados_gis\\relatorio_drone_resumo.md" \
    --out-json "1-CONTROLE_PLITOSSOLO\\media\\dados_gis\\relatorio_drone_resumo.json"
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from pypdf import PdfReader


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extrai métricas de relatório de ortomosaico (PDF).")
    p.add_argument("--pdf", required=True, help="Caminho do PDF (relatório do ortomosaico)")
    p.add_argument("--out-md", required=True, help="Saída Markdown")
    p.add_argument("--out-json", required=True, help="Saída JSON")
    p.add_argument("--max-pages", type=int, default=3, help="Número de páginas para varrer (normalmente 2-3 bastam)")
    return p.parse_args()


def _read_text(pdf_path: Path, max_pages: int) -> str:
    reader = PdfReader(str(pdf_path))
    chunks: list[str] = []
    for page in reader.pages[:max_pages]:
        chunks.append(page.extract_text() or "")
    text = "\n".join(chunks)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _pick(pattern: str, text: str) -> str | None:
    m = re.search(pattern, text, flags=re.IGNORECASE)
    if not m:
        return None
    g = m.group(1).strip()
    return g


def main() -> int:
    args = parse_args()
    pdf_path = Path(args.pdf)
    out_md = Path(args.out_md)
    out_json = Path(args.out_json)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF não encontrado: {pdf_path}")

    text = _read_text(pdf_path, max_pages=args.max_pages)

    data = {
        "title": _pick(r"MAPEAMENTO A[ÉE]REO COM ARP\s*(.*?)\s*(\d{1,2}\s+\w+\s+\d{4})", text),
        "date": _pick(r"MAPEAMENTO A[ÉE]REO COM ARP.*?(\d{1,2}\s+\w+\s+\d{4})", text),
        "number_of_images": _pick(r"Number of images:\s*([0-9,]+)", text),
        "flying_altitude_m": _pick(r"Flying altitude:\s*([0-9.]+)\s*m", text),
        "ground_resolution": _pick(r"Ground resolution:\s*([0-9.]+\s*\w+/pix)", text),
        "coverage_area_m2": _pick(r"Coverage area:\s*([0-9.eE+\-]+)\s*m\^?2", text),
        "camera_stations": _pick(r"Camera stations:\s*([0-9,]+)", text),
        "tie_points": _pick(r"Tie points:\s*([0-9,]+)", text),
        "projections": _pick(r"Projections:\s*([0-9,]+)", text),
        "reprojection_error_pix": _pick(r"Reprojection error:\s*([0-9.]+)\s*pix", text),
    }

    # limpa None
    data = {k: v for k, v in data.items() if v is not None}

    out_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        "# Resumo do levantamento por drone (relatório do ortomosaico)",
        "",
        "Este quadro resume os parâmetros do processamento reportados no PDF.",
        "",
        "| Parâmetro | Valor |",
        "|---|---|",
    ]

    mapping = {
        "date": "Data do relatório",
        "number_of_images": "Número de imagens",
        "flying_altitude_m": "Altitude de voo (m)",
        "ground_resolution": "Resolução no terreno",
        "coverage_area_m2": "Área de cobertura (m²)",
        "camera_stations": "Estações de câmera",
        "tie_points": "Tie points",
        "projections": "Projeções",
        "reprojection_error_pix": "Erro de reprojeção (pix)",
    }

    for key, label in mapping.items():
        if key in data:
            md_lines.append(f"| {label} | {data[key]} |")

    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print("OK. Resumos gerados:")
    print(out_md)
    print(out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
