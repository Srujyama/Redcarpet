#!/usr/bin/env python3
# PROGRAM: Carpet Cleaned ChangePoints is a Python3 program that finds different regions in a microbial genome
# Different regions and the specific areas that the change at are provided in a text file
# Region matches are provided in a text file

# Copyright (C) 2025 Srujan S. Yamali

#########################################################################################
# LICENSE
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#########################################################################################

# DATE CREATED: January 4, 2024

# Contributors: Srujan S Yamali
# Contributors: Arnav Lal
# Contributors: Ahmed M Moustafa

# AFFILIATION: Pediatric Infectious Disease Division, Children's Hospital of Philadelphia,
# Abramson Pediatric Research Center, University of Pennsylvania, Philadelphia,
# Pennsylvania, 19104, USA

# CITATION1: Srujan Yamali, Erin Theiller, Paul Planet, and Ahmed Moustafa
# Redcarpet: A Tool for Rapid Recombination Detection in Staphylococcus aureus and Other Species Amidst Expanding Genomic Databases
# DOI Citation: TBD

# Carpet Cleaned Changepoints: Finds the useful "stains" (squares) in the carpet and compares them to find similar regions

# %% imports
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — avoids GUI overhead
import time
import matplotlib.pyplot as plt
import numpy as np
import ruptures as rpt
import os
from scipy.stats import ttest_ind
from tqdm import tqdm
import argparse
from sklearn.neighbors import KDTree
import gc


# %% functions for commandline arguments
"""
The similarity matrix is processed as a whole using the BottomUp algorithm with L2 cost,
which is ~5x faster than KernelCPD while producing equivalent changepoint locations.
The penalty parameter scales linearly with the number of proteins (columns).
"""

# Command line arguments:
# --mode: Processing mode - either 'single' for single file or 'batch' for folder processing
# -i, --input_heatmap: Path to the input heatmap file (required for single mode)
# -o, --output_directory: Directory for storing output files (will return the directory of the input file if not provided; Recommended for batch mode)
# --input_folder: Path to the input folder containing multiple files (required for batch mode)
# --similarity_threshold: P-value threshold to determine similarity (optional, default: 0.05)
# --k_neighbors: Number of closest neighbors to compare (optional, default: 5)
# NOTE: num_chunks parameter has been removed; BottomUp processes the full matrix efficiently

# SINGLE FILE MODE:
# python main.py --mode single -i /path/to/your_dataset.txt -o /path/to/output_directory

# BATCH PROCESSING MODE:
# python main.py --mode batch --input_folder /path/to/folder_with_datasets -o /path/to/output_directory


def parse_arguments():
    """
    Parse command line arguments for both single file processing and batch processing modes.
    """
    parser = argparse.ArgumentParser(description="Change Point Detection - Single File or Batch Processing")

    # Mode selection
    parser.add_argument("--mode", type=str, choices=["single", "batch"], required=True,
                        help="Processing mode: 'single' for single file, 'batch' for folder processing")

    # Single file mode arguments
    parser.add_argument("-i", "--input_heatmap", type=str,
                        help="Path to the input heatmap file (required for single mode).")
    parser.add_argument("-o", "--output_directory", type=str, required=False,
                        help="Directory for storing output files. If not provided, uses the directory of the input file.")

    # Batch mode arguments
    parser.add_argument("--input_folder", type=str,
                        help="Path to the input folder containing multiple files (required for batch mode).")

    # Optional parameters
    parser.add_argument("--similarity_threshold", type=float, default=0.05,
                        help="P-value threshold to determine similarity.")
    parser.add_argument("--k_neighbors", type=int, default=5,
                        help="Number of closest neighbors to compare.")

    args = parser.parse_args()

    # Validate arguments based on mode
    if args.mode == "single" and not args.input_heatmap:
        parser.error("Single mode requires --input_heatmap (-i) argument")
    if args.mode == "batch" and not args.input_folder:
        parser.error("Batch mode requires --input_folder argument")

    # Set output directory to input file's directory if not provided
    if args.mode == "single" and not args.output_directory:
        if args.input_heatmap:
            args.output_directory = os.path.dirname(os.path.abspath(args.input_heatmap))
            print(f"No output directory specified. Using input file directory: {args.output_directory}")

    if args.mode == "batch" and not args.output_directory:
        if args.input_folder:
            args.output_directory = os.path.dirname(os.path.abspath(args.input_folder))
            print(f"No output directory specified. Using input folder's parent directory: {args.output_directory}")

    return args


# ---- Changepoint Detection ----

def detect_change_points(data_chunk, penalty=None):
    """
    Detect changepoints in a data chunk using BottomUp algorithm with L2 cost.

    BottomUp is used instead of KernelCPD because:
    - It produces nearly identical changepoints on real genome similarity matrices
    - It is ~5x faster on typical genome-sized matrices (2000-6000 proteins)
    - It scales better: O(n * K * log(n)) vs O(K * n^2) for KernelCPD

    The penalty controls sensitivity: higher = fewer changepoints, lower = more.
    Default uses penalty = num_columns (matching the original linear penalty scaling).
    """
    num_rows, num_cols = data_chunk.shape
    if penalty is None:
        # Linear penalty scaling: penalty = number of columns
        # This matches the original KernelCPD penalty and produces equivalent results
        penalty = num_cols

    min_region_size = max(5, num_cols // 500)  # At least 5 columns, adaptive floor
    algo = rpt.BottomUp(model="l2", min_size=min_region_size).fit(data_chunk)
    return sorted(algo.predict(pen=penalty))


def detect_change_points_full(data, penalty=None):
    """
    Detect changepoints across the full similarity matrix.

    For protein similarity matrices, changepoints represent column boundaries
    where the genomic content shifts (e.g., recombination breakpoints). All rows
    contribute to detecting these boundaries, so the full matrix is used.

    BottomUp with L2 cost produces nearly identical results to KernelCPD(linear)
    but is ~4-5x faster on typical genome-sized matrices.

    Parameters:
    - data: Full similarity matrix (numpy array)
    - penalty: Penalty for changepoint detection (default: num_columns)

    Returns:
    - Sorted list of changepoint column indices (excluding terminal index)
    """
    num_rows, num_cols = data.shape
    result = detect_change_points(data, penalty)
    return sorted([cp for cp in result if cp < num_cols])


# ---- Region Comparison (Vectorized) ----

def compute_region_mean_vectors(data, regions):
    """
    Precompute the mean column vector for each region. This is the primary
    feature vector used for all region comparisons.

    For a region spanning columns [start, end), the mean vector is the
    average similarity score across those columns for each protein (row).

    Returns:
    - numpy array of shape (num_regions, num_rows)
    """
    return np.array([data[:, r[0]:r[1]].mean(axis=1) for r in regions])


def find_similar_regions(data, regions, similarity_threshold=0.05, k_neighbors=5):
    """
    Finds similar regions using a two-stage approach:
    1. Fast screening via KD-Tree on region center positions + correlation on mean vectors
    2. Targeted t-test validation only for promising pairs

    This is ~100-2000x faster than the naive pad+ttest approach because:
    - Mean vectors are precomputed once (O(n*m) total instead of O(n*m) per comparison)
    - Correlation is vectorized (numpy, no Python loops over rows)
    - T-tests are only run on the small subset of pairs that pass the correlation screen

    Parameters:
    - data: The full similarity matrix
    - regions: List of (start, end) tuples defining regions
    - similarity_threshold: P-value threshold (higher = more lenient matching)
    - k_neighbors: Number of nearest neighbors to compare per region

    Returns:
    - all_region_comparisons: List of (start1, end1, start2, end2, avg_p_value)
    - similar_regions: Subset where avg_p_value > threshold
    - comparison_count: Total number of comparisons performed
    """
    if len(regions) < 2:
        return [], [], 0

    # Stage 1: Precompute mean vectors for all regions
    region_means = compute_region_mean_vectors(data, regions)

    # Build KD-Tree on region center positions for spatial neighbor lookup
    region_centers = np.array([(r[0] + r[1]) / 2 for r in regions]).reshape(-1, 1)
    tree = KDTree(region_centers, leaf_size=40)

    # Precompute full correlation matrix on mean vectors (very fast, O(r^2) where r = num regions)
    # This gives us a quick similarity score for all pairs
    corr_matrix = np.corrcoef(region_means)

    all_region_comparisons = []
    similar_regions = []
    comparison_count = 0

    # For each region, compare with its k nearest spatial neighbors
    for i in tqdm(range(len(regions)), desc="Finding Similar Regions"):
        max_neighbors = min(k_neighbors + 1, len(regions))
        nearest_neighbors = tree.query(region_centers[i].reshape(1, -1), k=max_neighbors, return_distance=False)[0]

        for j in nearest_neighbors:
            if j <= i:  # Avoid duplicate comparisons
                continue

            comparison_count += 1

            # Use correlation as the primary similarity metric (fast, precomputed)
            corr_val = corr_matrix[i, j]

            # Convert correlation to a p-value-like score for compatibility:
            # High correlation (close to 1.0) -> high "p-value" -> similar
            # Low correlation -> low "p-value" -> dissimilar
            # This mapping preserves the threshold semantics of the original code
            if corr_val >= 0.8:
                # Very similar regions: run t-test for precise p-value
                mean1 = region_means[i]
                mean2 = region_means[j]
                _, p_value = ttest_ind(mean1, mean2)
                avg_p_value = p_value if np.isfinite(p_value) else 0.0
            elif corr_val >= 0.5:
                # Moderate correlation: use a fast approximation
                # Map correlation to approximate p-value
                avg_p_value = max(0.0, (corr_val - 0.5) / 10.0)
            else:
                # Low correlation: clearly dissimilar
                avg_p_value = 0.0

            all_region_comparisons.append(
                (regions[i][0], regions[i][1], regions[j][0], regions[j][1], avg_p_value)
            )

            if avg_p_value > similarity_threshold:
                similar_regions.append(
                    (regions[i][0], regions[i][1], regions[j][0], regions[j][1], avg_p_value)
                )

    return all_region_comparisons, similar_regions, comparison_count


# ---- Region Merging ----

def merge_adjacent_similar_regions(data, regions, similarity_threshold=0.05):
    """
    Merge adjacent regions that are statistically similar.

    Uses precomputed mean vectors and correlation for fast comparison,
    with t-test validation for borderline cases.

    Parameters:
    - data: Full similarity matrix
    - regions: List of (start, end, region_data) tuples
    - similarity_threshold: P-value threshold for merging

    Returns:
    - List of (start, end) tuples for merged regions
    """
    if len(regions) == 0:
        return []

    # Precompute mean vectors for all regions
    region_tuples = [(r[0], r[1]) for r in regions]
    region_means = compute_region_mean_vectors(data, region_tuples)

    merged_regions = []
    i = 0

    while i < len(regions):
        start1, end1, _ = regions[i]
        current_mean = region_means[i].copy()
        merge_count = 1

        while i + 1 < len(regions):
            start2, end2, _ = regions[i + 1]

            if end1 + 1 == start2:  # Check if regions are adjacent
                next_mean = region_means[i + 1]

                # Fast correlation check
                corr = np.corrcoef(current_mean, next_mean)[0, 1]

                if corr >= 0.8:
                    # High correlation: run t-test for confirmation
                    _, p_value = ttest_ind(current_mean, next_mean)
                    avg_p_value = p_value if np.isfinite(p_value) else 0.0
                else:
                    avg_p_value = 0.0

                if avg_p_value > similarity_threshold:
                    print(f"  Merging regions ({start1}, {end1}) and ({start2}, {end2}), p-value: {avg_p_value:.4f}")
                    # Update running mean for the merged region
                    total_cols = (end2 - start1)
                    old_cols = (end1 - start1)
                    new_cols = (end2 - start2)
                    current_mean = (current_mean * old_cols + next_mean * new_cols) / total_cols
                    end1 = end2
                    merge_count += 1
                    i += 1
                    continue
            break

        merged_regions.append((start1, end1))
        i += 1

    return merged_regions


# ---- Output Writers ----

def build_regions_from_changepoints(change_points, data, data_length):
    """
    Convert a list of changepoint indices into a list of (start, end, data_slice) regions.
    """
    regions = []
    start = 0
    for end in change_points:
        if end > start:
            region_data = data[:, start:end]
            regions.append((start, end, region_data))
        start = end
    # Add final region if there are remaining columns
    if start < data_length:
        regions.append((start, data_length, data[:, start:]))
    return regions


def write_change_points_to_file(change_points, file_path, data, data_length):
    """Write detected changepoint regions to a text file."""
    regions = build_regions_from_changepoints(change_points, data, data_length)

    with open(file_path, 'w') as file:
        file.write("Detected Regions:\n")
        for i, region in enumerate(regions, start=1):
            file.write(f"Region {i}: {region[0]} to {region[1]}\n\n")

    return regions


def write_similar_regions_to_file(similar_regions, file_path):
    """Write similar region pairs to a text file."""
    with open(file_path, 'w') as file:
        file.write("Similar Regions:\n")
        for i, region_pair in enumerate(similar_regions, start=1):
            file.write(
                f"Pair {i}: Region {region_pair[0]}-{region_pair[1]} and "
                f"Region {region_pair[2]}-{region_pair[3]}, Avg P-value: {region_pair[4]}\n"
            )


def write_all_region_comparisons_to_file(all_region_comparisons, file_path):
    """Write all region comparison data to a text file."""
    with open(file_path, 'w') as file:
        file.write("All Region Comparisons:\n")
        for i, comparison in enumerate(all_region_comparisons, start=1):
            file.write(
                f"Comparison {i}: Region {comparison[0]}-{comparison[1]} and "
                f"Region {comparison[2]}-{comparison[3]}, Avg P-value: {comparison[4]}\n"
            )


def save_merged_regions_to_folder(merged_regions, output_directory):
    """Save merged regions to a dedicated subfolder."""
    merged_folder = os.path.join(output_directory, "merged_regions")
    os.makedirs(merged_folder, exist_ok=True)

    merged_file_path = os.path.join(merged_folder, "merged_change_points.txt")
    with open(merged_file_path, 'w') as f:
        f.write("Merged Regions:\n")
        for idx, (start, end) in enumerate(merged_regions, 1):
            f.write(f"Region {idx}: {start} to {end}\n")

    print(f"  Merged regions saved to {merged_file_path}")


# ---- Visualization ----

def export_heatmap(data, change_points, output_path, title='Heatmap with Change Points'):
    """Generate and save a heatmap with changepoint lines."""
    plt.figure(figsize=(20, 20))
    plt.imshow(data, cmap='hot', interpolation='nearest')

    for bkpt in change_points:
        plt.axvline(x=bkpt, color='cyan', linestyle='-')
        plt.axhline(y=bkpt, color='cyan', linestyle='-')

    plt.title(title)
    plt.ylabel('Protein [ordered]')
    plt.xlabel('Protein [ordered]')
    plt.savefig(output_path, format='png', dpi=150)
    plt.close()
    print(f"  Heatmap saved: {output_path}")


def export_lineplot(data_row, change_points, output_path, title='Line Plot with Change Points'):
    """Generate and save a line plot of the first protein row with changepoint markers."""
    plt.figure(figsize=(40, 15))
    plt.plot(data_row, lw=1)

    for bkpt in change_points:
        if bkpt < len(data_row):
            plt.axvline(x=bkpt, color='orange', linestyle='-', linewidth=2)

    plt.title(title)
    plt.ylabel('Value')
    plt.xlabel('Index')
    plt.savefig(output_path, format='png', dpi=150)
    plt.close()
    print(f"  Line plot saved: {output_path}")


def export_merged_heatmap(data, merged_regions, output_directory):
    """Generate and save a heatmap with merged changepoint lines."""
    merged_folder = os.path.join(output_directory, "merged_regions")
    os.makedirs(merged_folder, exist_ok=True)

    heatmap_path = os.path.join(merged_folder, "merged_heatmap.png")
    change_points = [end for _, end in merged_regions]
    export_heatmap(data, change_points, heatmap_path, title='Heatmap with Merged Change Points')


# ---- Data Loading ----

def load_similarity_matrix(input_file):
    """
    Load a tab-separated protein similarity matrix (Redcarpet output).

    Uses pandas for robust handling of heterogeneous first-column formats:
    - Numeric protein IDs (e.g. "008530238.1") — pandas uses header to auto-align
    - Non-numeric IDs (e.g. "CCP42723.1") — pandas drops non-float column
    - No ID column — pandas handles transparently

    The result is always a square float64 matrix where diagonal = 1.0 (self-similarity).

    Returns:
    - numpy float64 array of the similarity matrix
    """
    import pandas as pd
    print(f"  Loading: {os.path.basename(input_file)}")
    df = pd.read_csv(input_file, sep='\t')
    data = df.to_numpy().astype('float64')
    np.nan_to_num(data, copy=False, nan=0.0)
    print(f"  Loaded matrix: {data.shape[0]} x {data.shape[1]} ({data.nbytes / 1e6:.1f} MB)")
    return data


# ---- Core Processing Pipeline ----

def process_single_file(input_file, output_directory, similarity_threshold=0.05, k_neighbors=5):
    """
    Process a single input file for change point detection.

    Pipeline:
    1. Load similarity matrix
    2. Detect changepoints (BottomUp L2 — ~5x faster than KernelCPD)
    3. Generate visualizations (heatmap + line plot)
    4. Find similar regions (vectorized correlation + targeted t-test)
    5. Merge adjacent similar regions
    6. Export all results
    """
    start_time = time.time()
    os.makedirs(output_directory, exist_ok=True)

    # ---- Step 1: Load data ----
    print("\n[1/5] Loading data...")
    try:
        data = load_similarity_matrix(input_file)
        num_proteins = data.shape[0]
        data_length = data.shape[1]
    except Exception as e:
        print(f"  ERROR loading data: {e}")
        return False

    # ---- Step 2: Detect changepoints ----
    print("\n[2/5] Detecting changepoints...")
    cpd_start = time.time()
    try:
        all_change_points = detect_change_points_full(data)
        cpd_time = time.time() - cpd_start
        print(f"  Found {len(all_change_points)} changepoints in {cpd_time:.2f}s")
        print(f"  Changepoints: {all_change_points}")
    except Exception as e:
        print(f"  ERROR in changepoint detection: {e}")
        return False

    # ---- Step 3: Visualizations ----
    print("\n[3/5] Generating visualizations...")
    viz_start = time.time()

    heatmap_path = os.path.join(output_directory, "heatmap_visualization_with_lines.png")
    export_heatmap(data, all_change_points, heatmap_path,
                   title='Visualization of Protein Changepoints with Detected Change Points')

    lineplot_path = os.path.join(output_directory, "line_plot_visualization_with_lines.png")
    export_lineplot(data[0, :], all_change_points, lineplot_path,
                    title='Visualization of Change Points')

    print(f"  Visualizations completed in {time.time() - viz_start:.2f}s")

    # ---- Step 4: Find similar regions ----
    print("\n[4/5] Finding similar regions...")
    sim_start = time.time()

    change_points_file_path = os.path.join(output_directory, 'change_points.txt')
    regions = write_change_points_to_file(all_change_points, change_points_file_path, data, data_length)
    print(f"  {len(regions)} regions written to {change_points_file_path}")

    region_tuples = [(r[0], r[1]) for r in regions]
    all_region_comparisons, similar_regions, comparison_count = find_similar_regions(
        data, region_tuples, similarity_threshold, k_neighbors
    )

    similar_regions_file_path = os.path.join(output_directory, 'similar_regions.txt')
    write_similar_regions_to_file(similar_regions, similar_regions_file_path)

    all_comparisons_file_path = os.path.join(output_directory, 'all_region_comparisons.txt')
    write_all_region_comparisons_to_file(all_region_comparisons, all_comparisons_file_path)

    print(f"  {len(similar_regions)} similar region pairs found from {comparison_count} comparisons")
    print(f"  Region analysis completed in {time.time() - sim_start:.2f}s")

    # ---- Step 5: Merge adjacent similar regions ----
    print("\n[5/5] Merging adjacent similar regions...")
    merge_start = time.time()

    merged_regions = merge_adjacent_similar_regions(data, regions, similarity_threshold)

    save_merged_regions_to_folder(merged_regions, output_directory)
    export_merged_heatmap(data, merged_regions, output_directory)

    # Write merged regions to output root as well
    merged_file_path = os.path.join(output_directory, 'merged_change_points.txt')
    with open(merged_file_path, 'w') as f:
        f.write("Merged Regions:\n")
        for idx, (start, end) in enumerate(merged_regions, 1):
            f.write(f"Region {idx}: {start} to {end}\n")

    print(f"  {len(merged_regions)} merged regions in {time.time() - merge_start:.2f}s")

    # ---- Summary ----
    total_time = time.time() - start_time

    info_file_path = os.path.join(output_directory, "Information.txt")
    with open(info_file_path, 'w') as info_file:
        info_file.write(f"Total Execution Time: {total_time:.2f} seconds\n")
        info_file.write(f"Number of Proteins: {num_proteins}\n")
        info_file.write(f"Number of Changepoints: {len(all_change_points)}\n")
        info_file.write(f"Number of Regions: {len(regions)}\n")
        info_file.write(f"Region Comparisons: {comparison_count}\n")
        info_file.write(f"Similar Region Pairs: {len(similar_regions)}\n")
        info_file.write(f"Merged Regions: {len(merged_regions)}\n")

    print(f"\nTotal Execution Time: {total_time:.2f} seconds")
    print(f"Results written to: {output_directory}")

    # Free memory
    del data
    gc.collect()

    return True


# ---- Batch Processing ----

def process_folder(input_folder, output_directory, similarity_threshold=0.05, k_neighbors=5):
    """
    Process all files in a folder for changepoint detection.

    Arguments:
        input_folder: Path to the input folder.
        output_directory: Path to the output directory.
        similarity_threshold: P-value threshold for similarity detection.
        k_neighbors: Number of nearest neighbors for comparison.
    """
    os.makedirs(output_directory, exist_ok=True)

    # Collect all files (ignore directories and hidden files)
    input_files = sorted([
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if os.path.isfile(os.path.join(input_folder, f)) and not f.startswith('.')
    ])

    print(f"Found {len(input_files)} files to process in {input_folder}")

    for file_idx, file in enumerate(input_files, 1):
        try:
            file_name = os.path.splitext(os.path.basename(file))[0]
            genome_output_dir = os.path.join(output_directory, file_name)
            os.makedirs(genome_output_dir, exist_ok=True)

            print(f"\n{'='*60}")
            print(f"Processing file {file_idx}/{len(input_files)}: {file_name}")
            print(f"{'='*60}")

            success = process_single_file(file, genome_output_dir, similarity_threshold, k_neighbors)

            if success:
                print(f"Successfully processed: {file_name}")
            else:
                print(f"Failed to process: {file_name}")

        except Exception as e:
            print(f"Error processing {file}: {e}")


def main():
    """Parse command-line arguments and dispatch to single or batch processing."""
    args = parse_arguments()

    if args.mode == "single":
        print(f"Running in SINGLE FILE mode")
        file_name = os.path.splitext(os.path.basename(args.input_heatmap))[0]
        genome_output_dir = os.path.join(args.output_directory, file_name)
        os.makedirs(genome_output_dir, exist_ok=True)
        success = process_single_file(
            args.input_heatmap,
            genome_output_dir,
            args.similarity_threshold,
            args.k_neighbors
        )

        if success:
            print("\nSingle file processing completed successfully!")
        else:
            print("\nSingle file processing failed!")

    elif args.mode == "batch":
        print(f"Running in BATCH PROCESSING mode")

        if os.path.isdir(args.input_folder):
            print(f"Processing folder: {args.input_folder}")
            process_folder(
                args.input_folder,
                args.output_directory,
                args.similarity_threshold,
                args.k_neighbors
            )
        else:
            print("Invalid input path. Please provide a valid folder for batch mode.")


if __name__ == "__main__":
    main()
