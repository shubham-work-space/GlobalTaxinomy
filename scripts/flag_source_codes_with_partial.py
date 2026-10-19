"""Flag each source_code that has at least one partial_flag = P.

For both OSCA→ISCO and ISCO→OSCA ground-truth files: build
{source_code: 1 if any row has P, else 0}, write a `has_partial` column,
and print source codes with no P at all.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GT_DIR = ROOT / "ISCO_OSCA" / "Ground Truth"

FILES = [
    ("osca_to_isco08.csv", "osca_to_isco08_with_partial_flag.csv"),
    ("isco08_to_osca.csv", "isco08_to_osca_with_partial_flag.csv"),
]


def process(input_path: Path, output_path: Path) -> None:
    df = pd.read_csv(input_path)

    is_partial = (
        df["partial_flag"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
        .eq("P")
    )

    has_partial: dict[str, int] = {}
    for source_code, partial in zip(df["source_code"].astype(str), is_partial):
        if source_code not in has_partial:
            has_partial[source_code] = 0
        if partial:
            has_partial[source_code] = 1

    df["has_partial"] = df["source_code"].astype(str).map(has_partial)
    df.to_csv(output_path, index=False)

    no_partial = sorted(code for code, flag in has_partial.items() if flag == 0)
    with_partial = sum(1 for flag in has_partial.values() if flag == 1)
    taxonomy = df["source_taxonomy"].iloc[0] if len(df) else "?"

    print("=" * 60)
    print(f"{taxonomy}  ({input_path.name})")
    print(f"Rows: {len(df)}")
    print(f"Unique source_codes: {len(has_partial)}")
    print(f"  with at least 1 P: {with_partial}")
    print(f"  with no P at all:  {len(no_partial)}")
    print(f"Wrote: {output_path}")
    print()
    print("source_codes with no partial_flag (has_partial=0):")
    for code in no_partial:
        print(code)
    print()


def main() -> None:
    for infile, outfile in FILES:
        process(GT_DIR / infile, GT_DIR / outfile)


if __name__ == "__main__":
    main()
