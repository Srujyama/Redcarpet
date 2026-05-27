#!/usr/bin/env python3
"""Redcarpet pipeline preflight checks.

The pipeline does NOT download databases or annotate genomes. This script
verifies that the prerequisites the user is responsible for are actually in
place, and fails loudly with an actionable message if not:

  * The chosen WhatsGNU backend's database exists and looks complete.
      - atb:     <db_dir>/lmdb_counts/shard_00, <db_dir>/lmdb_postings/shard_00,
                 <db_dir>/indexes/genome_species.u32
      - classic: the ortholog .pickle file exists and is non-trivial in size.
  * Every input .faa parses as protein FASTA (has '>' records, sequences are
    amino-acid-like and not nucleotide), and contains at least one protein.

Usage:
  preflight.py --backend atb --db_dir DIR --shards 8 --faa_dir DIR [--touch OK_FILE]
  preflight.py --backend classic --pickle FILE --faa_dir DIR [--touch OK_FILE]

Exit code 0 and (optionally) writes OK_FILE on success; non-zero on failure.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Nucleotide FASTA gets silently hashed into meaningless alleles, so catch it.
NUCLEOTIDE_CHARS = set("ACGTUN")
AMINO_ACID_CHARS = set("ACDEFGHIKLMNPQRSTVWYBZXJUO*")


def fail(msg: str) -> "None":
    print(f"\n[PREFLIGHT FAILED] {msg}\n", file=sys.stderr)
    sys.exit(1)


def check_atb_db(db_dir: Path, shards: int) -> None:
    if not db_dir.exists():
        fail(
            f"ATB database directory not found: {db_dir}\n"
            f"  Download it first, e.g.:\n"
            f"    python WhatsGNU-ATB/scripts/download_osf.py "
            f"--folder WGNU_ATB_DB --out-dir {db_dir}"
        )

    counts = db_dir / "lmdb_counts"
    postings = db_dir / "lmdb_postings"
    gspec = db_dir / "indexes" / "genome_species.u32"

    if not counts.exists():
        fail(f"ATB counts DB missing: {counts} (expected under {db_dir})")
    if not postings.exists():
        fail(
            f"ATB postings DB missing: {postings}\n"
            f"  Redcarpet needs postings (per-protein genome lists). The "
            f"WGNU_ATB_DB download includes the posting shards."
        )
    if not gspec.exists():
        fail(f"ATB genome-species index missing: {gspec}")

    # Verify every expected shard exists for both stores (shards are named
    # shard_XX in lowercase hex: shard_00 .. shard_{shards-1}). A missing shard
    # usually means an interrupted download.
    for store, label in ((counts, "lmdb_counts"), (postings, "lmdb_postings")):
        for sid in range(shards):
            shard_dir = store / f"shard_{sid:02x}"
            data_mdb = shard_dir / "data.mdb"
            if not shard_dir.exists() or not data_mdb.exists():
                fail(
                    f"ATB shard missing or incomplete: {shard_dir} "
                    f"(expected {data_mdb}).\n"
                    f"  The DB download looks incomplete, or --shards ({shards}) "
                    f"does not match how the DB was built. Re-run download_osf.py "
                    f"(it resumes safely) or fix the shards setting."
                )

    print(f"[preflight] ATB DB OK: {db_dir} "
          f"({shards} shards × counts+postings, genome_species present)")


def check_classic_db(pickle: Path) -> None:
    if not pickle.exists():
        fail(
            f"Classic WhatsGNU ortholog pickle not found: {pickle}\n"
            f"  Download/unzip the Ortholog DB first (see BACTERIA_QUICKSTART.md)."
        )
    size_mb = pickle.stat().st_size / 1e6
    # A real ortholog pickle is hundreds of MB+; a few KB means a truncated zip.
    if size_mb < 1.0:
        fail(
            f"Classic pickle suspiciously small ({size_mb:.2f} MB): {pickle}\n"
            f"  The database zip was likely truncated mid-download. Re-download "
            f"with `curl -L -C -` and verify with `unzip -t`."
        )
    print(f"[preflight] Classic pickle OK: {pickle} ({size_mb:.0f} MB)")


def looks_like_protein(seq: str) -> bool:
    """Heuristic: a peptide should contain amino-acid letters beyond ACGTUN.

    Pure-nucleotide input (only A/C/G/T/U/N) is the common foot-gun: it parses
    as FASTA but produces garbage alleles. We flag that explicitly.
    """
    letters = {c for c in seq.upper() if c.isalpha()}
    if not letters:
        return False
    if letters <= NUCLEOTIDE_CHARS:
        return False  # looks like a nucleotide sequence
    # Otherwise, require it to be within the amino-acid alphabet.
    return letters <= AMINO_ACID_CHARS


def check_faa(path: Path) -> None:
    n_records = 0
    first_seq: list[str] = []
    capturing_first = False
    with path.open("rt", encoding="utf-8", errors="replace") as f:
        first_char = f.read(1)
        if first_char != ">":
            fail(
                f"{path.name} does not start with '>': not a FASTA file.\n"
                f"  Redcarpet needs a Bakta protein FASTA (.faa). If this is an "
                f"assembly (.fna), annotate it with Bakta first."
            )
        f.seek(0)
        for line in f:
            if line.startswith(">"):
                n_records += 1
                capturing_first = n_records == 1
                continue
            if capturing_first:
                first_seq.append(line.strip())
            # keep scanning so n_records reflects the whole file (cheap: one pass)

    if n_records == 0:
        fail(f"{path.name} has no FASTA records (no '>' headers).")

    seq = "".join(first_seq)
    if not seq:
        fail(f"{path.name}: first record has no sequence.")
    if not looks_like_protein(seq):
        fail(
            f"{path.name}: first sequence does not look like protein "
            f"(amino acids). It may be a nucleotide FASTA (.fna). Annotate "
            f"with Bakta to produce a protein .faa before querying."
        )
    print(f"[preflight] {path.name} OK: {n_records} protein records")


def discover_faa(faa_dir: Path) -> list[Path]:
    if faa_dir.is_file():
        return [faa_dir]
    if not faa_dir.is_dir():
        fail(f"FAA input path not found: {faa_dir}")
    faas = sorted(p for p in faa_dir.iterdir()
                  if p.is_file() and p.name.endswith(".faa"))
    if not faas:
        fail(f"No .faa files found in {faa_dir}")
    return faas


def main() -> int:
    ap = argparse.ArgumentParser(description="Redcarpet pipeline preflight checks")
    ap.add_argument("--backend", required=True, choices=["atb", "classic"])
    ap.add_argument("--db_dir", help="ATB DB directory (backend=atb)")
    ap.add_argument("--shards", type=int, default=8, help="ATB shards (backend=atb)")
    ap.add_argument("--pickle", help="Ortholog pickle (backend=classic)")
    ap.add_argument("--faa_dir", required=True, help="Folder of .faa files (or one .faa)")
    ap.add_argument("--touch", help="On success, create this sentinel file")
    args = ap.parse_args()

    if args.backend == "atb":
        if not args.db_dir:
            fail("--db_dir is required for backend=atb")
        check_atb_db(Path(args.db_dir), args.shards)
    else:
        if not args.pickle:
            fail("--pickle is required for backend=classic")
        check_classic_db(Path(args.pickle))

    faas = discover_faa(Path(args.faa_dir))
    for p in faas:
        check_faa(p)

    print(f"\n[preflight] All checks passed: backend={args.backend}, "
          f"{len(faas)} genome(s).")
    if args.touch:
        Path(args.touch).parent.mkdir(parents=True, exist_ok=True)
        Path(args.touch).write_text("preflight ok\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
