import os
import sys
import math
import json


def split_files(input_dir, num_cores):
    # Get all files in the input directory
    all_files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, f)) and f.endswith(".cdf")
    ]
    total_files = len(all_files)

    # Calculate the number of files per core
    files_per_core = math.ceil(total_files / num_cores)

    # Split the files into groups for each core
    split_file_groups = [
        all_files[i : i + files_per_core] for i in range(0, total_files, files_per_core)
    ]

    # Save the split file groups to JSON files
    for idx, group in enumerate(split_file_groups):
        with open(f"input_file_lists/input_files_core_{idx}.json", "w") as f:
            json.dump(sorted(group), f)

    print(
        f"Split {total_files} files into {num_cores} groups, each having up to {files_per_core} files.\nThese have been saved as JSON files in the input_file_lists/ folder."
    )


if __name__ == "__main__":
    # If the folder input_file_lists does not exist, create it
    if not os.path.exists("input_file_lists"):
        os.makedirs("input_file_lists")
    # Otherwise, if there is anything in that folder, delete it
    else:
        for f in os.listdir("input_file_lists"):
            os.remove(os.path.join("input_file_lists", f))

    if len(sys.argv) != 3:
        print("Usage: python split_files.py <input_directory> <num_cores>")
        sys.exit(1)

    input_directory = sys.argv[1]
    num_cores = int(sys.argv[2])

    split_files(input_directory, num_cores)
