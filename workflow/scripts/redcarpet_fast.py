#!/usr/bin/env python3
# Faster, drop-in-compatible reimplementation of the Redcarpet matrix step.
#
# GOAL: produce a *byte-identical* <name>_Redcarpet_report.txt to the original
# Redcarpet_01062025.py, but compute the N×N Jaccard matrix orders of magnitude
# faster. Nothing about the output values, ordering, or float formatting changes.
#
# HOW IT STAYS IDENTICAL
#   * Parsing is copied verbatim from Redcarpet_01062025.py (same protein_id
#     extraction, same no_hits->[0], same --hash_file strain->int mapping, same
#     sorted(set(...)) per protein, same --bottom_k truncation, same dedup).
#   * The Jaccard value for a pair is |A∩B| / (|A| + |B| − |A∩B|) with EXACT
#     integer intersection counts, then Python float division — identical to the
#     original's np.sum(np.in1d(...)) intersection. Same integers in => same
#     float out => same str() => same bytes.
#   * The report layout is reproduced exactly: header line of protein IDs, then
#     row i = the off-diagonal Jaccard values in increasing column order with
#     1.0 at position i, joined by str() and tabs.
#
# THE SPEEDUP
#   The original does N²/2 calls to np.in1d over large genome-ID sets (~42k each
#   at ATB scale). Here we build a sparse genome×protein incidence matrix once
#   and compute ALL pairwise intersection counts in a single sparse matmul
#   (Mᵀ·M). That replaces millions of Python-level set comparisons with one
#   vectorized SciPy operation, while yielding the exact same integer counts.
#
# This script is OUTPUT-COMPATIBLE but is NOT a line-for-line copy of the
# original; it is a separate, faster path the pipeline can opt into.

import os
import sys
import time
import argparse
import numpy as np
from scipy import sparse

START_TIME = time.time()

PARSER = argparse.ArgumentParser(
    prog="redcarpet_fast.py",
    description="Alignment-Free Recombination Detector (fast, output-identical)",
)
PARSER.add_argument("-v", "--version", action="version", version="%(prog)s 1.0-fast")
PARSER.add_argument("-bk", "--bottom_k", type=int,
                    help="bottom-k cutoff for hits (default: all hits are used)")
PARSER.add_argument("--hash_file", type=str, help="ids hash file for a WhatsGNU database")
PARSER.add_argument("ids_hits_file", type=str, help="ids_hits_file from WhatsGNU -i option")
if len(sys.argv) == 1:
    PARSER.print_help()
    sys.exit(0)
ARGS = PARSER.parse_args()

OS_SEPARATOR = os.sep
if ARGS.bottom_k:
    BOTTOM_K_CUTOFF = ARGS.bottom_k

QUERY = ARGS.ids_hits_file
QUERY_LIST = []
try:
    for file in os.listdir(QUERY):
        if file.endswith("WhatsGNU_hits.txt"):
            QUERY_LIST.append(QUERY + file)
    if len(QUERY_LIST) == 0:
        PARSER.exit(status=0, message="The directory did not have WhatsGNU_hits files\n")
except Exception:
    if QUERY.endswith("WhatsGNU_hits.txt"):
        QUERY_LIST.append(QUERY)
    else:
        PARSER.exit(
            status=0,
            message="You did not provide single faa file or path to directory with multiple faa files\n",
        )

if ARGS.hash_file:
    HASHFILE_OBJECT = open(ARGS.hash_file, "r")
    hash_dict = {}
    for line in HASHFILE_OBJECT:
        line = line.rstrip()
        strain_name, hash_value = line.split(",")
        hash_dict[strain_name] = int(hash_value)
    HASHFILE_OBJECT.close()
else:
    print("No hash file included.")

counter = 0
for QUERYFILE in QUERY_LIST:
    QUERYFILE_OBJECT = open(QUERYFILE, "r")
    if ARGS.bottom_k:
        file_report = (
            (QUERYFILE.rsplit(OS_SEPARATOR, 1)[-1]).split("WhatsGNU_hits.txt")[0]
            + "bk_{}_".format(str(BOTTOM_K_CUTOFF)) + "Redcarpet_report.txt"
        )
    else:
        file_report = (
            (QUERYFILE.rsplit(OS_SEPARATOR, 1)[-1]).split("WhatsGNU_hits.txt")[0]
            + "bk_all_" + "Redcarpet_report.txt"
        )

    # ---- Parse (verbatim semantics from Redcarpet_01062025.py) ----
    ids_hits_dict = {}
    proteins_list = []
    QUERYFILE_OBJECT.readline()  # skip header
    for line in QUERYFILE_OBJECT:
        line = line.rstrip()
        ids_hits_list = line.split("\t")

        if ARGS.hash_file:
            ids_hits_split = ids_hits_list[1:]
        else:
            ids_hits_split = [int(x) if x != 'no_hits' else 0
                              for x in ids_hits_list[1].split('.')]

        if ids_hits_list[1] == 'no_hits':
            strains_list = [0]
        else:
            if ARGS.hash_file:
                strains_list = [hash_dict[i.split('|')[0]] for i in ids_hits_split]
            else:
                strains_list = ids_hits_split
            strains_list.sort()
            if ARGS.bottom_k:
                strains_list = strains_list[:BOTTOM_K_CUTOFF]
            else:
                strains_list = strains_list[:]
        try:
            protein_id = line.split("\t")[0].split(" ", 1)[0].rsplit('|', 1)[0]
        except Exception:
            protein_id = line.split("\t")[0].split(" ", 1)[0].rsplit('_', 1)[0]
        if protein_id in proteins_list:
            counter += 1
            continue
        else:
            proteins_list.append(protein_id)
            ids_hits_dict[protein_id] = np.array(sorted(set(strains_list)))
    QUERYFILE_OBJECT.close()
    print("Done parsing id_hits_file in --- {:.3f} seconds ---".format(time.time() - START_TIME))

    print('Starting pairwise comparisons')
    sets_array = [ids_hits_dict[protein] for protein in proteins_list]
    n_sets = len(sets_array)
    sizes = np.array([len(s) for s in sets_array], dtype=np.int64)

    # ---- Vectorized pairwise intersection via sparse incidence matrix ----
    # Build a (num_genomes × n_proteins) boolean incidence matrix M where
    # M[g, p] = 1 iff protein p has genome id g in its (deduped, sorted) set.
    # Then (Mᵀ · M)[i, j] = |A_i ∩ A_j| — the exact same integer intersection
    # count the original computed with np.sum(np.in1d(...)).
    #
    # Genome IDs are mapped to a compact contiguous index so the matrix is dense
    # in rows actually used (keeps memory bounded regardless of max genome id).
    if n_sets == 0:
        pairwise_jaccard = np.zeros((0, 0))
    else:
        all_ids = np.concatenate(sets_array) if n_sets else np.array([], dtype=np.int64)
        uniq_ids, compact = np.unique(all_ids, return_inverse=True)
        n_genomes = len(uniq_ids)

        # Column (protein) index for each (genome,protein) incidence entry.
        col_idx = np.empty(len(all_ids), dtype=np.int64)
        pos = 0
        for p in range(n_sets):
            ln = sizes[p]
            col_idx[pos:pos + ln] = p
            pos += ln
        row_idx = compact  # genome (compact) index per entry
        data = np.ones(len(all_ids), dtype=np.int64)

        M = sparse.csr_matrix((data, (row_idx, col_idx)),
                              shape=(n_genomes, n_sets), dtype=np.int64)
        # Intersection counts for all pairs (n_sets × n_sets), exact integers.
        inter = (M.T @ M).toarray()  # symmetric; diagonal = |A_i|

        # Jaccard: |A∩B| / (|A| + |B| − |A∩B|). Compute with the SAME integer
        # numerator/denominator and Python float division to match str() bytes.
        sizes_i = sizes.reshape(-1, 1)
        sizes_j = sizes.reshape(1, -1)
        union = sizes_i + sizes_j - inter
        pairwise_jaccard = np.zeros((n_sets, n_sets), dtype=np.float64)
        nz = union > 0
        # true_divide on integer arrays gives identical IEEE-754 doubles as a/b.
        np.divide(inter, union, out=pairwise_jaccard, where=nz)

    print("Done with Jaccard similarity comparisons in --- {:.3f} seconds ---".format(
        time.time() - START_TIME))

    # ---- Write report (byte-identical layout to the original) ----
    # Row i: off-diagonal Jaccard in increasing column order, 1.0 at position i,
    # each value formatted with str() exactly as the original.
    output_file_report = open(file_report, "w")
    output_file_report.write('\t'.join(proteins_list) + '\n')
    for i in range(n_sets):
        row_vals = pairwise_jaccard[i].tolist()  # Python floats
        row_vals[i] = 1.0                         # self-similarity, as inserted
        output_file_report.write(
            proteins_list[i] + '\t' + '\t'.join(map(str, row_vals)) + '\n')
    output_file_report.close()

print("Done in --- {:.3f} seconds ---".format(time.time() - START_TIME))
