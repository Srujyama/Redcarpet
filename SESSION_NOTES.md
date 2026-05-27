# Session Notes — Redcarpet pipeline (handoff for future Claude sessions)

**Read this first.** It is the canonical entry point for resuming work on the
Redcarpet pipeline. It explains what was built, where everything lives, how to
run/verify it, and the non-obvious gotchas — so a fresh session can be effective
immediately without re-deriving the context.

---

## 0. Where to start / where things live

- **The repo** (this directory) is the deliverable. Work here.
- **The workspace** `~/Downloads/Microbial_Stuff/Redcarpet_Pipeline_Workspace/`
  is the home base that ties together the repo + all the large/external material
  via symlinks. Its `README.md` is the physical-layout map. Start a session by
  reading *this* file, then that one if you need to locate an asset.
- **Large/external assets** (gitignored, NOT in the repo):
  - `WGNU_ATB_DB/` — 26 GB pre-built ATB LMDB database (in the repo dir, gitignored).
  - `WhatsGNU-ATB/` — clone of microbialARC/WhatsGNU-ATB (has `download_osf.py`).
  - `scripts/Redcarpet_01062025.py` — the ORIGINAL Redcarpet matrix script (separately maintained, not committed).
  - `WhatsGNU/`, `whatsgnu_db/` — symlinks to `~/Downloads/Redcarpet_Demo_PI/` (classic backend: WhatsGNU tool + Sau ortholog pickle + test genomes).
  - `~/Downloads/Microbial_Stuff/Microbial_Datasets/` — pre-made Redcarpet reports (Sau/TB/Kp/Pa/TW20) for changepoint analysis.

## 1. What this project is

A Snakemake pipeline for alignment-free recombination detection:

```
.faa  →  WhatsGNU (per-protein genome hits)  →  Redcarpet (N×N Jaccard matrix)  →  CarpetCleanChangepoints (regions)
```

Two interchangeable WhatsGNU backends, chosen by `config/config.yaml`
`whatsgnu_backend: atb | classic`:
- **atb** (default): new WhatsGNU-ATB, exact-match against 2.4M AllTheBacteria
  genomes via the LMDB DB. Inputs = your `.faa` + the LMDB.
- **classic**: original WhatsGNU + a per-species Ortholog `.pickle`. Inputs =
  your `.faa` + the pickle (auto-builds a strain→int CSV mid-run for Redcarpet).

## 2. Repo layout (committed)

```
workflow/Snakefile                     — the pipeline (preflight→whatsgnu→redcarpet→changepoints)
workflow/scripts/Query_WhatsGNU_ATB.py — ATB query, upstream + a --dump_hits patch (REDCARPET PATCH markers)
workflow/scripts/redcarpet_fast.py     — fast, BYTE-IDENTICAL Redcarpet matrix engine (default)
workflow/scripts/make_hash_file.py     — strain→int map (classic backend)
workflow/scripts/preflight.py          — validates DB present + inputs are protein FASTA
Redcarpet/CarpetCleanChangepoints.py   — changepoint/region analysis (pre-existing)
config/config.yaml                     — backend switch, paths, engine, parameters
data/faa/                              — drop query .faa here (sampleA/B are tiny smoke-test slices)
```

## 3. How to run

```bash
conda env create -f redcarpet.yml      # one-time
conda activate redcarpet
snakemake -s workflow/Snakefile --cores 4        # add -n for dry run
```
Outputs land in `results/<sample>/` (gitignored). Backend + paths in `config/config.yaml`.

## 4. The two key things built this project

### (a) The ATB → Redcarpet bridge (`--dump_hits`)
Redcarpet needs, per protein, the list of genome IDs carrying that allele. The
old WhatsGNU emits this (`-i`); WhatsGNU-ATB computes it internally but never
writes it. The vendored `Query_WhatsGNU_ATB.py` adds **only** a `--dump_hits`
flag that writes a Redcarpet-compatible hits file (`protein\tgid1.gid2...`, or
`no_hits`). ATB IDs are integers, so Redcarpet needs no `--hash_file` for atb.
Everything else in that file is byte-for-byte upstream.

### (b) The fast Redcarpet engine (`redcarpet_fast.py`)
The original `Redcarpet_01062025.py` does an N²/2 Python loop of `np.in1d` over
genome-ID sets. At ATB scale (~42k IDs/protein) that took **>2 hours** for one
genome. `redcarpet_fast.py` computes all pairwise intersections in one sparse
matmul (Mᵀ·M) → **~4 min**, and is **byte-identical** (proven by matching
SHA-256 on both classic and ATB reports). Selected via `redcarpet.engine: fast`
(default) | `original`.

## 5. Verified end-to-end (do not need to redo unless code changes)

- **classic** on full NCTC8325 (2630 proteins) reproduces the validated numbers
  EXACTLY: 9 changepoints `[380,1300,1370,1745,1815,1885,1940,2135,2175]`, 10
  regions, 2 similar pairs.
- **atb** on full NCTC8325 against the real 2.4M-genome DB: query ~40s, fast
  Redcarpet ~4 min, 9 changepoints / 10 regions / 1 similar pair (differs from
  classic because the reference panel is broader — expected).
- The pre-made `Microbial_Datasets/` reports all still run through
  `CarpetCleanChangepoints.py` (Sau/TB/Kp/Pa/TW20, all ID-format variants).

## 6. Gotchas (learned the hard way)

- Redcarpet `cd`s into its output dir → pass script/hash paths as ABSOLUTE.
- Redcarpet report name = input basename minus `WhatsGNU_hits.txt` + `bk_all_`
  (or `bk_<N>_`) + `Redcarpet_report.txt`.
- Don't reference a backend-conditional input (classic-only `hash_csv`) directly
  in a Snakemake shell template — it fails to format on the other backend. Pass
  as a param (empty for the other backend).
- gitignore: use patterns WITHOUT trailing slash too, so symlinks to external
  copies (WhatsGNU, whatsgnu_db, WGNU_ATB_DB) are also ignored.
- `--dump_hits` holds all posting lists in RAM (~1 GB for NCTC8325); fast engine
  peaks ~7 GB at ATB scale. Fine on a normal workstation.

## 7. Git / PR state

- Remote `origin` = github.com/Srujyama/Redcarpet (fork); `upstream` =
  microbialARC/Redcarpet. Branch: `fix/readme-carpetclean-testing`.
- Committed deliverables: `workflow/`, `config/config.yaml`, `data/faa/{README,sampleA,sampleB}`,
  and updated `README.md`, `redcarpet.yml`, `requirements.txt`, `.gitignore`, this file.
- Nothing is auto-committed/pushed beyond what the user approves.

## 8. Possible next steps (not yet done)

- A `setup.sh` one-shot (env + clone tools + kick off DB download).
- A "changepoint-only" pipeline mode to run a folder of existing Redcarpet
  reports (like `Microbial_Datasets/`) through Snakemake directly.
- Bakta annotation rule (currently inputs must already be protein `.faa`).
