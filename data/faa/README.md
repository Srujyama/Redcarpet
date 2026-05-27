# Example query inputs

Drop your Bakta-annotated protein FASTA files (`*.faa`) here. The pipeline
processes every `*.faa` in this folder, one independent run per file.

`sampleA.faa` and `sampleB.faa` are **small truncated slices** of the public
WhatsGNU test genomes (NCTC8325 and H37Rv), included only as a fast smoke test
of the pipeline wiring. They are too small to produce meaningful changepoints —
replace them with your real proteomes for actual analysis.

Inputs must be protein FASTA (amino-acid sequences). If you have a nucleotide
assembly (`.fna`), annotate it with Bakta first; the preflight step will refuse
nucleotide input.
