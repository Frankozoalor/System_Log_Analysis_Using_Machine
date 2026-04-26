from __future__ import annotations

from pathlib import Path
import re
import pandas as pd

HDFS_PATTERN = re.compile(
    r"^(?P<date>\d{6})\s+"
    r"(?P<time>\d{6})\s+"
    r"(?P<pid>\d+)\s+"
    r"(?P<level>[A-Z]+)\s+"
    r"(?P<component>[^:]+):\s+"
    r"(?P<message>.*)$"
)

BLOCK_RE = re.compile(r"(blk_-?\d+)")

def parse_log_file(file_path: str | Path) -> pd.DataFrame:
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix.lower() not in {".log", ".csv"}:
        raise ValueError("Only .log or .csv files are supported.")

    rows: list[dict] = []

    with path.open("r", encoding="utf-8", errors="ignore") as file:
        for i, line in enumerate(file):
            raw_line = line.strip()
            if not raw_line:
                continue

            match = HDFS_PATTERN.match(raw_line)
            block_match = BLOCK_RE.search(raw_line)

            if match:
                message = match.group("message")
                rows.append(
                    {
                        "line_index": i,
                        "date": match.group("date"),
                        "time": match.group("time"),
                        "pid": int(match.group("pid")),
                        "level": match.group("level"),
                        "component": match.group("component"),
                        "message": message,
                        "block_id": block_match.group(1) if block_match else "NO_BLOCK",
                        "line": raw_line,
                    }
                )
            else:
                rows.append(
                    {
                        "line_index": i,
                        "date": "",
                        "time": "",
                        "pid": 0,
                        "level": "UNKNOWN",
                        "component": "UNKNOWN",
                        "message": raw_line,
                        "block_id": block_match.group(1) if block_match else "NO_BLOCK",
                        "line": raw_line,
                    }
                )

    return pd.DataFrame(rows)


   