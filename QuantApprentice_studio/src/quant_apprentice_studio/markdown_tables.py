from __future__ import annotations

from pathlib import Path
from typing import Dict, List


def _parse_table_lines(lines: List[str]) -> List[Dict[str, str]]:
    rows = []
    for line in lines:
        text = line.strip()
        if not text.startswith("|") or text.count("|") < 2:
            continue
        cells = [cell.strip() for cell in text.strip("|").split("|")]
        rows.append(cells)
    if len(rows) < 2:
        return []
    header = rows[0]
    body = rows[2:] if len(rows) >= 2 else []
    return [dict(zip(header, row)) for row in body]


def load_section_tables(path: Path) -> Dict[str, List[Dict[str, str]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    current_section = ""
    buffer: List[str] = []
    output: Dict[str, List[Dict[str, str]]] = {}
    for line in lines:
        if line.startswith("## "):
            if current_section and buffer:
                frame = _parse_table_lines(buffer)
                if frame:
                    output[current_section] = frame
            current_section = line[3:].strip()
            buffer = []
            continue
        if line.strip().startswith("|"):
            buffer.append(line)
            continue
        if current_section and buffer and line.strip() == "":
            frame = _parse_table_lines(buffer)
            if frame:
                output[current_section] = frame
            buffer = []
    if current_section and buffer:
        frame = _parse_table_lines(buffer)
        if frame:
            output[current_section] = frame
    return output
