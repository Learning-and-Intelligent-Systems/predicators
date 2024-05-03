import os
import shutil
import argparse
import re

"""Script for oganizing a sequence of images into a saved dataset for a VLM.

To read more about the format, look at the documentation inside the approaches/
folder.
"""

def extract_number(filename):
    match = re.search(r'\d+', filename)
    if match:
        return int(match.group())
    return None

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo_dir', required=True, type=str, help='Path to the directory containing images that need to be reorganized, relative to /saved_datasets/.')
    args = parser.parse_args()
    # Assume this script is run from the high-level predicators folder.
    demo_dir = 'saved_datasets/' + args.demo_dir

    traj_folders = [f for f in os.listdir(demo_dir) if 'traj_' in f[0:5]]
    for traj_folder in traj_folders:
        image_dir = os.path.join(demo_dir, traj_folder)
        import pdb; pdb.set_trace()
        # Count the number of images we have.
        # Assume for now that we only have 1 image per option execution.
        files = [f for f in os.listdir(image_dir) if f.endswith('.png') or f.endswith('.jpg')]
        sorted_files = sorted(files, key=extract_number)
        files = sorted_files
        num_files = len(files)

        # Create directories for each image and move each into its corresponding directory.
        for i, file in enumerate(files):
            # traj_number = str(i).zfill(len(str(num_files)))
            traj_dir = os.path.join(image_dir, f'{i}')
            os.makedirs(traj_dir, exist_ok=True)

            src_path = os.path.join(image_dir, file)
            dest_path = os.path.join(image_dir, f'{i}')
            shutil.move(src_path, dest_path)

if __name__ == "__main__":
    main()