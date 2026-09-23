#!/usr/bin/env python3
"""Low-memory staged feature analysis for large sliding-window datasets.

This script is designed for Colab / constrained RAM:
- read Parquet/CSV directly with DuckDB
- do not load the full dataset into Pandas
- keep only aggregated results in memory

Typical usage:
    python single_feature_analysis.py \
        --windows /content/windows_10_2000.parquet \
        --raw /content/410_kalibrasi.md \
        --out /content/hypothesis_output

If the window file already contains target columns, the raw file may be omitted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import duckdb

THRESHOLDS = [2, 5, 10, 20, 50, 100]
TARGET_COLUMNS = ["target_low"] + [f"target_ge_{t}" for t in THRESHOLDS]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Low-memory staged hypothesis analysis for large window datasets.")
    p.add_argument("--windows", required=True, type=Path, help="CSV/Parquet file with sliding-window rows.")
    p.add_argument("--raw", type=Path, default=None, help="Raw Markdown file used to derive next-game target.")
    p.add_argument("--out", type=Path, default=Path("hypothesis_output"), help="Output directory")
    p.add_argument("--threads", type=int, default=2, help="DuckDB threads (keep low on Colab free tier)")
    p.add_argument("--min-cell", type=int, default=100, help="Minimum rows per bucket to keep for interpretation")
    return p.parse_args()


def qident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def sql_path(path: Path) -> str:
    return "'" + str(path.resolve()).replace("'", "''") + "'"


def relation(path: Path) -> str:
    if path.suffix.lower() == ".parquet":
        return f"read_parquet({sql_path(path)})"
    if path.suffix.lower() == ".csv":
        return f"read_csv_auto({sql_path(path)}, header=true)"
    raise ValueError("--windows harus berekstensi .parquet atau .csv")


def discover_columns(con: duckdb.DuckDBPyConnection, path: Path) -> list[str]:
    rows = con.execute(f"DESCRIBE SELECT * FROM {relation(path)}").fetchall()
    return [str(row[0]).strip() for row in rows]


def read_raw_rows(raw_path: Path) -> list[tuple[int, float, int]]:
    """Parse a markdown table with columns like game_id and gr_result.

    The format is similar to the repo's file 410_kalibrasi.md.
    """
    text = raw_path.read_text(encoding="utf-8")
    lines = [ln.strip() for ln in text.splitlines() if ln.lstrip().startswith("|")]
    if len(lines) < 3:
        raise ValueError(f"Tidak menemukan tabel Markdown valid di {raw_path}")

    header_idx = None
    header_cells = None
    for idx, ln in enumerate(lines):
        cells = [c.strip() for c in ln.strip("|").split("|")]
        if cells and cells[0] == "":
            cells = cells[1:]
        lowered = [c.lower() for c in cells]
        if "game_id" in lowered or "gr_result" in lowered:
            header_idx = idx
            header_cells = cells
            break

    if header_idx is None or header_cells is None:
        raise ValueError(f"Header game_id/gr_result tidak ditemukan di {raw_path}")

    rows: list[tuple[int, float, int]] = []
    for ln in lines[header_idx + 2:]:
        cells = [c.strip() for c in ln.strip("|").split("|")]
        if cells and cells[0] == "":
            cells = cells[1:]
        if not cells or len(cells) < 2:
            continue
        try:
            game_idx = header_cells.index("game_id") if "game_id" in header_cells else None
            gr_idx = header_cells.index("gr_result") if "gr_result" in header_cells else None
            if game_idx is None or gr_idx is None:
                continue
            game_id = int(float(cells[game_idx].replace(",", "")))
            gr_value = float(cells[gr_idx].replace(",", "."))
        except (IndexError, TypeError, ValueError):
            continue
        rows.append((game_id, gr_value, len(rows)))

    if not rows:
        raise ValueError(f"Tidak ada baris valid game_id/gr_result di {raw_path}")
    rows.sort(key=lambda x: x[0])
    return rows


def add_target_table(con: duckdb.DuckDBPyConnection, raw_path: Path | None) -> None:
    """Build raw_next from markdown if target columns are absent."""
    if raw_path is None:
        return
    rows = read_raw_rows(raw_path)
    con.execute("CREATE OR REPLACE TEMP TABLE raw_rounds(game_id BIGINT, gr_result DOUBLE, raw_order BIGINT)")
    con.executemany("INSERT INTO raw_rounds VALUES (?, ?, ?)", rows)
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE raw_next AS
        SELECT game_id AS last_game_id,
               LEAD(game_id) OVER (ORDER BY raw_order) AS target_game_id,
               LEAD(gr_result) OVER (ORDER BY raw_order) AS target_gr
        FROM raw_rounds
        """
    )


def build_target_view(con: duckdb.DuckDBPyConnection, rel: str, columns: list[str], raw_path: Path | None) -> tuple[str, list[str]]:
    cols_lower = {c.lower(): c for c in columns}
    has_target = all(c.lower() in cols_lower for c in ["target_low"] + [f"target_ge_{t}" for t in THRESHOLDS])
    if has_target:
        con.execute(f"CREATE OR REPLACE TEMP VIEW data AS SELECT * FROM {rel}")
        return "data", columns

    if raw_path is None:
        raise ValueError("Target belum ada di file window dan --raw tidak disediakan.")

    if "last_game_id" not in cols_lower:
        raise ValueError("Kolom last_game_id tidak ditemukan. File window harus memiliki last_game_id untuk menurunkan target dari raw.")

    join_col = qident(cols_lower["last_game_id"])
    target_expr = ["r.target_game_id", "r.target_gr"]
    target_expr += ["r.target_gr < 2 AS target_low"]
    target_expr += [f"r.target_gr >= {t} AS target_ge_{t}" for t in THRESHOLDS]

    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW data AS
        SELECT w.*, {', '.join(target_expr)}
        FROM {rel} AS w
        LEFT JOIN raw_next AS r ON CAST(w.{join_col} AS BIGINT) = r.last_game_id
        WHERE r.target_gr IS NOT NULL
        """
    )
    return "data", columns + ["target_game_id", "target_gr"] + TARGET_COLUMNS


def find_col(names: dict[str, str], *candidates: str) -> str | None:
    for candidate in candidates:
        if candidate.lower() in names:
            return names[candidate.lower()]
    return None


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute(f"PRAGMA threads={max(1, min(args.threads, 4))}")
    con.execute("PRAGMA enable_progress_bar=false")

    rel = relation(args.windows)
    columns = discover_columns(con, args.windows)
    add_target_table(con, args.raw)
    view_name, columns = build_target_view(con, rel, columns, args.raw)

    names = {c.lower(): c for c in columns}
    window_size = find_col(names, "window_size")
    window_end = find_col(names, "window_end", "window_no", "raw_order")
    duration = find_col(names, "window_duration", "duration", "duration_seconds", "window_seconds")
    recency = find_col(names, "since_ge_10", "since_10")
    frequency = find_col(names, "count_ge_10", "count_10")

    baseline_sql = f"""
        SELECT COUNT(*)::BIGINT AS n,
               AVG(CAST(target_low AS DOUBLE)) AS low_rate,
               AVG(CAST(target_ge_2 AS DOUBLE)) AS p_ge_2,
               AVG(CAST(target_ge_5 AS DOUBLE)) AS p_ge_5,
               AVG(CAST(target_ge_10 AS DOUBLE)) AS p_ge_10,
               AVG(CAST(target_ge_20 AS DOUBLE)) AS p_ge_20,
               AVG(CAST(target_ge_50 AS DOUBLE)) AS p_ge_50,
               AVG(CAST(target_ge_100 AS DOUBLE)) AS p_ge_100
        FROM {view_name}
        WHERE target_low IS NOT NULL
    """
    baseline = con.execute(baseline_sql).fetchdf()
    baseline.to_csv(args.out / "baseline.csv", index=False)
    base_low = float(baseline.iloc[0]["low_rate"]) if not baseline.empty else 0.0

    tests: list[tuple[str, str, str]] = []
    if duration:
        d_col = qident(duration)
        tests.append(
            (
                "H1_window_duration",
                f"CASE WHEN {d_col} < 10 THEN '0-10' WHEN {d_col} < 20 THEN '10-20' WHEN {d_col} < 30 THEN '20-30' WHEN {d_col} < 45 THEN '30-45' WHEN {d_col} < 60 THEN '45-60' WHEN {d_col} < 90 THEN '60-90' WHEN {d_col} < 120 THEN '90-120' ELSE '>=120' END",
                duration,
            )
        )

    if duration and window_size:
        d_col = qident(duration)
        w_col = qident(window_size)
        con.execute(
            f"""
            CREATE OR REPLACE TEMP VIEW duration_reference AS
            SELECT {w_col} AS window_size_ref,
                   MEDIAN(CAST({d_col} AS DOUBLE)) AS median_duration
            FROM {view_name}
            GROUP BY {w_col}
            """
        )
        con.execute(
            f"""
            CREATE OR REPLACE TEMP VIEW ratio_data AS
            SELECT x.*, CASE WHEN x.{w_col} > 0 THEN x.{d_col} / NULLIF(r.median_duration, 0) END AS duration_ratio
            FROM {view_name} AS x
            JOIN duration_reference AS r ON x.{w_col} = r.window_size_ref
            """
        )
        tests.append(
            (
                "H2_duration_ratio",
                "CASE WHEN duration_ratio < 0.5 THEN '<0.5' WHEN duration_ratio < 0.75 THEN '0.5-0.75' WHEN duration_ratio < 1 THEN '0.75-1' WHEN duration_ratio < 1.25 THEN '1-1.25' WHEN duration_ratio < 1.5 THEN '1.25-1.5' WHEN duration_ratio < 2 THEN '1.5-2' ELSE '>=2' END",
                "duration_ratio",
            )
        )

    if recency:
        r_col = qident(recency)
        tests.append(
            (
                "H3_since_ge_10",
                f"CASE WHEN {r_col} = 0 THEN '0' WHEN {r_col} <= 2 THEN '1-2' WHEN {r_col} <= 5 THEN '3-5' WHEN {r_col} <= 10 THEN '6-10' WHEN {r_col} <= 20 THEN '11-20' ELSE '>20' END",
                recency,
            )
        )

    if frequency:
        f_col = qident(frequency)
        tests.append(
            (
                "H4_count_ge_10",
                f"CASE WHEN {f_col} = 0 THEN '0' WHEN {f_col} = 1 THEN '1' WHEN {f_col} = 2 THEN '2' WHEN {f_col} <= 5 THEN '3-5' ELSE '>5' END",
                frequency,
            )
        )

    frames = []
    for hypothesis, bucket_expr, feature in tests:
        source = "ratio_data" if hypothesis == "H2_duration_ratio" else view_name
        query = f"""
            SELECT '{hypothesis}' AS hypothesis,
                   CAST(({bucket_expr}) AS VARCHAR) AS bucket,
                   COUNT(*)::BIGINT AS n,
                   SUM(CAST(target_low AS BIGINT))::BIGINT AS low_count,
                   AVG(CAST(target_low AS DOUBLE)) AS low_rate,
                   AVG(CAST(target_ge_2 AS DOUBLE)) AS p_ge_2,
                   AVG(CAST(target_ge_5 AS DOUBLE)) AS p_ge_5,
                   AVG(CAST(target_ge_10 AS DOUBLE)) AS p_ge_10,
                   AVG(CAST(target_ge_20 AS DOUBLE)) AS p_ge_20,
                   AVG(CAST(target_ge_50 AS DOUBLE)) AS p_ge_50,
                   AVG(CAST(target_ge_100 AS DOUBLE)) AS p_ge_100,
                   ({base_low}) AS baseline_low_rate,
                   AVG(CAST(target_low AS DOUBLE)) - ({base_low}) AS delta_low_rate,
                   CASE WHEN ({base_low}) > 0 THEN AVG(CAST(target_low AS DOUBLE)) / ({base_low}) END AS low_rate_ratio
            FROM {source}
            WHERE target_low IS NOT NULL AND ({bucket_expr}) IS NOT NULL
            GROUP BY bucket
            ORDER BY bucket
        """
        frames.append(con.execute(query).fetchdf())

    if frames:
        import pandas as pd
        results = pd.concat(frames, ignore_index=True)
        results["enough_rows"] = results["n"] >= args.min_cell
        results.to_csv(args.out / "hypothesis_results.csv", index=False)
    else:
        pd = __import__("pandas")
        pd.DataFrame(columns=["hypothesis", "bucket", "n", "low_count", "low_rate", "p_ge_2", "baseline_low_rate", "delta_low_rate", "low_rate_ratio"]).to_csv(args.out / "hypothesis_results.csv", index=False)

    period_expr = "'all'"
    if window_end:
        e_col = qident(window_end)
        period_expr = (
            f"CASE WHEN {e_col} < (SELECT MIN({e_col}) + (MAX({e_col}) - MIN({e_col})) / 3 FROM {view_name}) THEN 'early' "
            f"WHEN {e_col} < (SELECT MIN({e_col}) + 2 * (MAX({e_col}) - MIN({e_col})) / 3 FROM {view_name}) THEN 'middle' ELSE 'late' END"
        )

    period = con.execute(
        f"""
        SELECT CAST({period_expr} AS VARCHAR) AS period,
               COUNT(*)::BIGINT AS n,
               AVG(CAST(target_low AS DOUBLE)) AS low_rate,
               AVG(CAST(target_ge_2 AS DOUBLE)) AS p_ge_2,
               AVG(CAST(target_ge_5 AS DOUBLE)) AS p_ge_5,
               AVG(CAST(target_ge_10 AS DOUBLE)) AS p_ge_10,
               AVG(CAST(target_ge_20 AS DOUBLE)) AS p_ge_20
        FROM {view_name}
        WHERE target_low IS NOT NULL
        GROUP BY period
        ORDER BY period
        """
    ).fetchdf()
    period.to_csv(args.out / "period_results.csv", index=False)

    summary = {
        "windows_file": str(args.windows),
        "raw_file": str(args.raw) if args.raw else None,
        "columns": columns,
        "target_source": "existing" if any(c.lower() == "target_low" for c in columns) else "derived_from_raw",
        "detected_features": {
            "window_size": window_size,
            "window_end": window_end,
            "duration": duration,
            "since_ge_10": recency,
            "count_ge_10": frequency,
        },
        "hypotheses_run": [name for name, _, _ in tests],
        "min_cell": args.min_cell,
        "notes": [
            "Sliding windows overlap, so row counts are not independent samples.",
            "Use the CSV outputs for screening, then validate candidates chronologically.",
        ],
    }
    (args.out / "run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nOutput written to: {args.out.resolve()}")
    print("Need to send back: run_summary.json, baseline.csv, hypothesis_results.csv, period_results.csv")


if __name__ == "__main__":
    main()
