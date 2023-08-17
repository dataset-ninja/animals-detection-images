import hashlib
import os

from tqdm import tqdm


def calculate_checksum(file_path):
    """
    Calculate the MD5 checksum of a file.
    """
    with open(file_path, "rb") as file:
        md5_hash = hashlib.md5()
        while chunk := file.read(8192):
            md5_hash.update(chunk)
        return md5_hash.hexdigest()


def find_duplicate_files(directory):
    """
    Find duplicate files in a directory and its subdirectories.
    """
    file_checksums = {}
    duplicates = []

    # Traverse through the directory and its subdirectories

    for root, dirs, files in tqdm(os.walk(directory), desc="Processing files"):
        for file in files:
            file_path = os.path.join(root, file)

            # Calculate the checksum of the current file
            checksum = calculate_checksum(file_path)

            # Check if the checksum already exists in the file_checksums dictionary
            if checksum in file_checksums:
                duplicates.append((file_path, file_checksums[checksum]))
            else:
                file_checksums[checksum] = file_path

    return duplicates


# Specify the directory to search for duplicates
directory = "/home/grokhi/supervisely/dataset-ninja/animals-detection-images/APP_DATA/archive"

# Find duplicate files in the specified directory
duplicate_files = find_duplicate_files(directory)

output_file = "duplicate_files.txt"

with open(output_file, "w") as file:
    if duplicate_files:
        file.write(f"Duplicate files found {len(duplicate_files)}:\n")
        for file1, file2 in duplicate_files:
            file.write(f"{file1} (duplicate of {file2})\n")
    else:
        file.write("No duplicate files found.")
