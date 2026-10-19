## Redcarpet (**Re**combination **D**etection using **C**omparative **A**nalysis of **R**egional **P**atterns of **E**xact Match **T**argets)

Redcarpet is a alignment-free recombination detection tool that utilizes genomic database distributions of exact protein matches. Redcarpet builds on the [WhatsGNU](https://github.com/ahmedmagds/WhatsGNU) method, which uses exact matching for identifying proteomic novelty.

Redcarpet takes in a single query genome, and for each encoded protein, determines the set of genomes in a database that contain an exact protein sequence match. It then computes the Jaccard similarity coefficient between genome sets for all pairwise protein comparisons in the genome.

---

## Pipeline (Snakemake)

The full pipeline — WhatsGNU → Redcarpet → changepoint analysis — is wrapped in a Snakemake workflow that batches over a folder of query proteomes and supports **two interchangeable WhatsGNU backends**:

| Backend | What it is | Database |
|---------|------------|----------|
| `atb` (default) | New [WhatsGNU-ATB](https://github.com/microbialARC/WhatsGNU-ATB) — exact-match against **2,438,285 AllTheBacteria genomes** | Pre-built LMDB `WGNU_ATB_DB/` |
| `classic` | Original [WhatsGNU](https://github.com/ahmedmagds/WhatsGNU) — per-species Ortholog database | `*_Ortholog_*.pickle` |

Both produce the same downstream artifacts. The pipeline **does not download databases or annotate genomes** — those are your responsibility. A preflight step verifies the database is present and that inputs are protein FASTA before any heavy work runs.

```
.faa  →  WhatsGNU (hits file)  →  Redcarpet (N×N Jaccard matrix)  →  CarpetCleanChangepoints
```

### Layout

```
config/config.yaml                     — pipeline configuration (backend, paths, parameters)
workflow/Snakefile                     — the pipeline
workflow/scripts/Query_WhatsGNU_ATB.py — ATB query, patched with --dump_hits (see below)
workflow/scripts/make_hash_file.py     — strain-name → integer-ID map (classic backend)
workflow/scripts/preflight.py          — database + input validation
Redcarpet/CarpetCleanChangepoints.py   — changepoint analysis
data/faa/                              — drop your *.faa query proteomes here
results/<sample>/                      — outputs, one subfolder per genome
```

### 1. Install

```bash
conda env create -f redcarpet.yml
conda activate redcarpet
```

This installs Snakemake, the analysis stack, and `lmdb` (needed by the ATB backend). You also need the relevant WhatsGNU tool available:

- **atb**: clone [WhatsGNU-ATB](https://github.com/microbialARC/WhatsGNU-ATB) for its `download_osf.py` (to fetch the DB). The query script itself is vendored here (patched — see [The ATB → Redcarpet bridge](#the-atb--redcarpet-bridge)).
- **classic**: clone [WhatsGNU](https://github.com/ahmedmagds/WhatsGNU) (provides `bin/WhatsGNU_main.py`).

### 2. Get the database (one-time)

**ATB** — download the pre-built LMDB database (no OSF account needed):

```bash
python WhatsGNU-ATB/scripts/download_osf.py --folder WGNU_ATB_DB --out-dir ./WGNU_ATB_DB
```

This is large; the downloader resumes safely if interrupted. The DB contains `lmdb_counts/`, `lmdb_postings/`, and `indexes/genome_species.u32`, built with **8 shards**.

**Classic** — download and unzip the Ortholog database for your species (see [Additional/BACTERIA_QUICKSTART.md](Additional/BACTERIA_QUICKSTART.md) for verified per-species commands and checksums).

### 3. Configure

Edit `config/config.yaml`. The key knob is the backend:

```yaml
whatsgnu_backend: "atb"      # or "classic"
faa_dir: "data/faa"          # folder of *.faa query proteomes
results_dir: "results"

atb:
  db_dir: "WGNU_ATB_DB"
  shards: 8

classic:
  whatsgnu_dir: "WhatsGNU"
  pickle: "whatsgnu_db/WhatsGNU_Sau_Ortholog/Sau_Ortholog_10350.pickle"
  database_mode: "ortholog"

redcarpet:
  script: "scripts/Redcarpet_01062025.py"   # maintained separately (see note below)
```

> The core Redcarpet matrix script (`Redcarpet_01062025.py`) is **maintained separately and not included in this repository**. Point `redcarpet.script` at your copy.

### 4. Run

```bash
snakemake -s workflow/Snakefile --cores 4          # run
snakemake -s workflow/Snakefile --cores 4 -n       # dry run (show the plan)
```

Each query genome `X.faa` produces `results/X/`:

```
results/X/
├── X.WhatsGNU_hits.txt                — per-protein genome-ID hits (pipeline-internal)
├── X.faa.whatsgnu.tsv                 — GNU scores (atb backend only)
├── X.faa.similarity.tsv              — genome similarity ranking (atb backend only)
├── X.strain_hashes.csv               — strain→int map (classic backend only)
├── redcarpet/X.bk_all_Redcarpet_report.txt   — N×N Jaccard matrix (the "red carpet")
├── carpet/Information.txt            — changepoint summary
├── carpet/X.bk_all_Redcarpet_report/        — full changepoint outputs (see below)
└── logs/                             — per-step logs
```

### Performance and the fast Redcarpet engine

The WhatsGNU-ATB **query** is fast — ~40 s for a 2,630-protein genome against all 2.4M genomes (in line with the upstream "~5–150 s per genome, ~2–4 GB RAM" figures). The historically expensive step is **Redcarpet**'s N×N Jaccard matrix: it compares every protein's genome-set against every other's (≈N²/2 comparisons), and the per-comparison cost scales with genome-set size. ATB sets are far larger than the classic Ortholog DB, which made the original script impractically slow:

| | genome-set size (per protein) | original engine | **fast engine** |
|---|---|---|---|
| classic (Sau Ortholog, ~10k strains) | ≤ 10,349 | ~2 min | **~26 s** |
| **atb (2.4M genomes)** | **~42,000 avg, up to ~160k** | **> 2 hours** | **~4 min** |

The pipeline ships a **fast Redcarpet engine** (`workflow/scripts/redcarpet_fast.py`, the default — `redcarpet.engine: fast`). It replaces the original's N²/2 per-pair `np.in1d` loop with a single sparse matrix multiply (`Mᵀ·M`) that computes all pairwise intersection counts at once. It is **byte-for-byte identical** to the original `Redcarpet_01062025.py` output — verified by matching SHA-256 on both the classic and ATB reports — just much faster. Set `redcarpet.engine: original` to use the upstream script instead (same output, slow).

`redcarpet.bottom_k` (cap each protein's smallest-k genome IDs) still works with either engine if you want to reduce set sizes further; the fast engine is quick enough that `bk_all` is fine for routine ATB use. The hits file and `--dump_hits` are unaffected by `bottom_k`.

> Memory notes: `--dump_hits` retains every protein's full posting list in RAM to write the hits file (~1 GB of integer IDs for NCTC8325 against the 2.4M-genome DB). The fast Redcarpet engine peaks around ~7 GB at ATB scale (the sparse incidence matrix). Both are fine on a normal workstation.

### The ATB → Redcarpet bridge

Redcarpet needs, for each query protein, the **list of genome IDs** that carry an identical protein allele. The old WhatsGNU emits exactly this via its `-i` option. WhatsGNU-ATB computes the same per-protein genome lists internally (to build its genome-level `similarity.tsv`) but **never writes them out**.

`workflow/scripts/Query_WhatsGNU_ATB.py` is a copy of upstream with **one added feature**: a `--dump_hits` flag that writes those per-protein lists to a Redcarpet-compatible hits file:

```
protein_query<TAB>hits
<protein_id><TAB><gid1>.<gid2>.<gid3>...
<protein_id_with_no_db_match><TAB>no_hits
```

The dot-separated integer genome IDs are exactly what Redcarpet parses on its no-`--hash_file` code path, so the ATB backend needs **no strain-hash file**. `--dump_hits` implies `--with_postings` and decodes the full posting list so the Jaccard sets are complete. The rest of the file is byte-for-byte upstream; the patch is clearly marked with `REDCARPET PATCH` comments.

---

## Running steps manually (without Snakemake)

The pipeline just orchestrates standalone scripts. To run a single step by hand:

### WhatsGNU-ATB (new)

```bash
python workflow/scripts/Query_WhatsGNU_ATB.py \
    --db_dir WGNU_ATB_DB --shards 8 \
    --faa your_genome.faa \
    --out_dir results/ \
    --with_postings \
    --dump_hits results/your_genome.WhatsGNU_hits.txt
```

Then feed the hits file to Redcarpet **without** `--hash_file`:

```bash
python scripts/Redcarpet_01062025.py results/your_genome.WhatsGNU_hits.txt
```

### WhatsGNU (classic)

WhatsGNU must first be run to get the hits file. To install WhatsGNU, see [here](https://github.com/ahmedmagds/WhatsGNU?tab=readme-ov-file#installation).

Note: before running the WhatsGNU database download script, create the `db/` directory first (`mkdir -p db/`), otherwise the download will fail.

Using a hashed database:

```WhatsGNU_main_hashes.py -d $database_path -csv $file.csv -i --hash_values -o $output_directory query_faa/```

Using an Ortholog database (e.g., `Sau_Ortholog_10350.pickle`):

```WhatsGNU_main.py -d $database_path -dm ortholog -i -o $output_directory query.faa```

```
whatsgnu_output/
├── GCF_000005845.2_prtn_id_hashes.csv
├── GCF_000005845.2_WhatsGNU_hits.txt
├── GCF_000005845.2_WhatsGNU_report.txt
└── WhatsGNU_20250108_115433.log
```

Then run Redcarpet:

```python3 Redcarpet.py $WhatsGNU_hits.txt```

```
usage: Redcarpet.py [-h] [-v] [-bk BOTTOM_K] [--hash_file HASH_FILE] ids_hits_file

Alignment-Free Recombination Detector

positional arguments:
  ids_hits_file         ids_hits_file from WhatsGNU -i option

options:
  -h, --help            show this help message and exit
  -v, --version         print version and exit
  -bk BOTTOM_K, --bottom_k BOTTOM_K
                        bottom-k cutoff for hits (default: all hits are used)
  --hash_file HASH_FILE
                        ids hash file for a WhatsGNU database
```

Note: the `--hash_file` argument is required when using a WhatsGNU **Ortholog** database, because its hits file contains string-format strain names (e.g., `N315_GCA_000009645.1_CC5_`) rather than integer IDs. If omitted, Redcarpet will fail with `ValueError: invalid literal for int()`. Build it with `workflow/scripts/make_hash_file.py`. The ATB backend does not need this (its hits are already integer IDs).

Note: the actual Redcarpet script (e.g., `Redcarpet_01062025.py`) is maintained separately and is not included in this repository.

### Changepoint Analysis

Once the Redcarpet report has been generated, run the changepoint analysis step.

#### Single File Processing
```python3 CarpetCleanChangepoints.py --mode single -i $input_report -o $output_directory```

#### Batch Processing
```python3 CarpetCleanChangepoints.py --mode batch --input_folder $input_folder -o $output_directory```

#### Command Line Options:
```--mode : Required. Processing mode - either single for single file or batch for folder processing

-i, --input_heatmap : Path to the heatmap file generated by Redcarpet (required for single mode)

--input_folder : Path to folder containing multiple Redcarpet reports (required for batch mode)

-o, --output_directory : Directory to store outputs (optional - defaults to input file/folder directory)

--similarity_threshold : P-value threshold for determining region similarity (default: 0.05, lower = more strict)

--k_neighbors : Number of nearest neighbors to compare for efficiency (default: 5, higher = more comparisons)
```

The output directory will contain the following files:

```
change_points.txt              — detected regions with boundaries
similar_regions.txt            — region pairs identified as similar
all_region_comparisons.txt     — all pairwise region comparisons with p-values
merged_change_points.txt       — merged regions
Information.txt                — summary statistics
heatmap_visualization_with_lines.png    — heatmap with changepoint lines
line_plot_visualization_with_lines.png  — line plot of first protein row
merged_regions/merged_change_points.txt — merged regions (subfolder copy)
merged_regions/merged_heatmap.png       — heatmap with merged changepoint lines
```

### Additional Resources Available

*Already completed heatmaps and reports for all S.aureus and K.pneumoniae*

See [Additional/BACTERIA_QUICKSTART.md](Additional/BACTERIA_QUICKSTART.md) for a verified manual end-to-end walkthrough of the classic backend (validated on *S. aureus* NCTC8325).

### Interpreting WhatsGNU-ATB GNU scores

When using the ATB backend, `*.whatsgnu.tsv` reports a GNU score per protein: the number of the 2,438,285 AllTheBacteria genomes carrying an identical copy of that protein. Thresholds follow the [WhatsGNU-ATB documentation](https://github.com/microbialARC/WhatsGNU-ATB):

- **>100,000:** highly conserved, ubiquitous allele.
- **1,000–10,000:** common allele.
- **1–100:** rare allele, likely strain-specific.
- **0:** unique to your query — not in any other AllTheBacteria genome.
