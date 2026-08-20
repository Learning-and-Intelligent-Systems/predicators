"""Prepare all environment images for documentation."""
import os
import shutil
from pathlib import Path

from predicators import utils


def create_all_env_images() -> None:
    """Process images in the images directory:

    - From each subfolder that doesn't end with '_cf'
    - Take the first image from its seed0/support/task1 directory
    - Copy and rename them to a new folder called 'all_envs_demo'
    """
    # Define paths
    images_dir = Path(
        os.path.join(utils.get_path_to_predicators_root(), "images"))
    output_dir = Path(
        os.path.join(utils.get_path_to_predicators_root(), "images",
                     "all_envs_demo"))

    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)

    # Get all subfolders that don't end with '_cf' and are directories
    subfolders = [
        d for d in images_dir.iterdir() if d.is_dir()
        and not d.name.endswith('_cf') and d.name != 'all_envs_demo'
    ]

    print(f"Found {len(subfolders)} environment folders to process:")
    for folder in sorted(subfolders):
        print(f"  - {folder.name}")

    processed_count = 0

    for subfolder in sorted(subfolders):
        # Construct path to the task1 directory
        task_dir = subfolder / "seed0" / "support" / "task1"

        if not task_dir.exists():
            print(f"Warning: Path {task_dir} does not exist, "
                  f"skipping {subfolder.name}")
            continue

        # Get all image files in the task1 directory
        image_files = sorted([
            f for f in task_dir.iterdir()
            if f.is_file() and f.suffix.lower() in ['.png', '.jpg', '.jpeg']
        ])

        if not image_files:
            print(f"Warning: No image files found in {task_dir},"
                  f" skipping {subfolder.name}")
            continue

        # Take the first image
        first_image = image_files[0]

        # Create new filename with subfolder prefix
        original_extension = first_image.suffix
        new_filename = f"{subfolder.name}{original_extension}"
        output_path = output_dir / new_filename

        # Copy the image
        try:
            shutil.copy2(first_image, output_path)
            print(
                f"Copied: {subfolder.name}/{first_image.name} -> {new_filename}"
            )
            processed_count += 1
        except Exception as e:  # pylint: disable=broad-except
            print(f"Error copying {first_image}: {e}")

    print(f"\nProcessing complete! Successfully processed "
          f"{processed_count} environments.")
    print(f"Images saved to: {output_dir}")


if __name__ == "__main__":
    create_all_env_images()
