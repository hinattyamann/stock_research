from __future__ import annotations

import json
from pathlib import Path
import sys

def main(a: str, b: str) -> int:
    pa, pb = Path(a), Path(b)
    ja = json.loads(pa.read_text(encoding="utf-8"))
    jb = json.loads(pb.read_text(encoding="utf-8"))

    # Compare only the most important numeric summaries
    keys = [
        ("summary_main", "pooled_macro_f1"),
        ("summary_main", "pooled_balanced_acc"),
        ("summary_subset", "pooled_macro_f1"),
        ("summary_subset", "pooled_balanced_acc"),
    ]
    ok = True
    for sec, k in keys:
        va = ja.get(sec, {}).get(k, None)
        vb = jb.get(sec, {}).get(k, None)
        print(f"{sec}.{k}: {va} vs {vb}")
        if va != vb:
            ok = False
    return 0 if ok else 1

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python tools/compare_run_summaries.py <run_summary_a.json> <run_summary_b.json>")
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
