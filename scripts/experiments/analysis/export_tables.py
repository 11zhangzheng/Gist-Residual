"""Compile future validated result JSON files into machine-readable table data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fidmem.experiments.results import collect_tables
from fidmem.production.authority import canonical_json_bytes


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate real results and export table data"
    )
    parser.add_argument("results", nargs="+")
    parser.add_argument("--schemas", default="configs/experiments/result_schemas.yaml")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    tables = collect_tables(args.results, args.schemas)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes({"schema_version": 1, "tables": tables}))
    print(
        json.dumps(
            {
                "output": str(output),
                "row_counts": {key: len(value) for key, value in tables.items()},
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
