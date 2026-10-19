#!/usr/bin/env python3
"""Build a CSV mapping each unique WhatsGNU strain name to a sequential integer ID.

Source: extracted verbatim from BACTERIA_QUICKSTART.md (Step 4) in the Redcarpet repo.
Usage: python3 make_hash_file.py <WhatsGNU_hits.txt> <output.csv>
"""
import csv, sys
hits_file, out_csv = sys.argv[1], sys.argv[2]
seen = {}
with open(hits_file) as f:
    next(f, None)  # skip header
    for line in f:
        parts = line.rstrip("\n").split("\t")
        for tok in parts[1:]:
            if not tok:
                continue
            strain = tok.split("|", 1)[0].strip()
            if strain and strain not in seen:
                seen[strain] = len(seen)
with open(out_csv, "w", newline="") as f:
    w = csv.writer(f)
    for name, i in seen.items():
        w.writerow([name, i])
print(f"Wrote {len(seen)} strain mappings to {out_csv}")
