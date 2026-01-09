from __future__ import annotations

import re
from pathlib import Path

DOC = Path(r"c:\Users\vidal\OneDrive\Documentos\13 - CLONEGIT\artigo-posdoc\3-EROSIBIDADE\1-MANUSCRITOS\1-CONTROLE_PLITOSSOLO\Controle_Ravinas_Paliçadas.md")


def main() -> None:
    text = DOC.read_text(encoding="utf-8")
    lines = text.splitlines()

    fig_captions: list[tuple[int, int, str]] = []
    tab_captions: list[tuple[int, int, str]] = []

    for line_no, line in enumerate(lines, start=1):
        m = re.search(r"!\[Figura\s+(\d+)\s+--", line)
        if m:
            fig_captions.append((line_no, int(m.group(1)), line.strip()))

        m = re.search(r"\*\*Tabela\s+(\d+)\*\*", line)
        if m:
            tab_captions.append((line_no, int(m.group(1)), line.strip()))

    print("FIG CAPTIONS:")
    for line_no, n, _ in fig_captions:
        print(f"L{line_no}: Figura {n}")

    print("\nTABLE CAPTIONS:")
    for line_no, n, _ in tab_captions:
        print(f"L{line_no}: Tabela {n}")

    print(f"\nTOTAL fig {len(fig_captions)} tab {len(tab_captions)}")


if __name__ == "__main__":
    main()
