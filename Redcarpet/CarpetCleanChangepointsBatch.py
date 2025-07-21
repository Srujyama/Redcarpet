#!/usr/bin/env python3
# PROGRAM: Cluster Carpet Cleaner is a program that allows Carpet Cleaned ChangePoints to be used for batch processing

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

# DATE CREATED: April 8, 2025

# %%imports
import os
import argparse
import subprocess

# This script processes either a single file or all files inside a folder
# by running a specified external Python script on each input file.
# Example usage: python3 script.py /path/to/input /path/to/output /path/to/your_script.py

def run_script_on_file(input_file, output_directory, script_path):
    """
    Arguments:
        input_file (str): Path to the input file.
        output_directory (str): Path to the output directory where results will be saved.
        script_path (str): Path to the Python script to be executed on the input file.
    """
    try:
        # Extract the base file name (without extension) to create a unique output subdirectory
        file_name = os.path.splitext(os.path.basename(input_file))[0]
        genome_output_dir = os.path.join(output_directory, file_name)

        # Create the output subdirectory if it doesn't exist
        if not os.path.exists(genome_output_dir):
            os.makedirs(genome_output_dir)

        # Build the command to run the external script
        command = ["python3", script_path, "-i", input_file, "-o", genome_output_dir]

        # Execute the command and ensure any errors are raised
        subprocess.run(command, check=True)

        print(f"Successfully processed: {input_file}")

    except subprocess.CalledProcessError as e:
        # Catch and report any errors that occur while running the external script
        print(f"Error processing {input_file}: {e}")

def process_folder(input_folder, output_directory, script_path):
    # Create the main output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Collect all file paths inside the input folder (ignore directories)
    input_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]

    # Process each file individually
    for file in input_files:
        run_script_on_file(file, output_directory, script_path)

def main():
    # Parses command-line arguments and initiates processing based on whether the input is a file or a folder.
    parser = argparse.ArgumentParser(description="Process files or folders with a specified script.")
    parser.add_argument("input_path", type=str, help="Path to the input file or folder")
    parser.add_argument("output_directory", type=str, help="Path to the output directory")
    parser.add_argument("script_path", type=str, help="Path to the script to be executed")

    args = parser.parse_args()

    # Check if the input path is a directory (folder) or a single file
    if os.path.isdir(args.input_path):
        print(f"Processing folder: {args.input_path}")
        process_folder(args.input_path, args.output_directory, args.script_path)
    elif os.path.isfile(args.input_path):
        print(f"Processing single file: {args.input_path}")
        run_script_on_file(args.input_path, args.output_directory, args.script_path)
    else:
        print("Invalid input path. Please provide a valid file or folder.")

if __name__ == "__main__":
    main()
