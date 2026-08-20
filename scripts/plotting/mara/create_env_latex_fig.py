"""Create LaTeX figures for environment visualizations."""
import math
import os
from pathlib import Path
from typing import Tuple

from predicators import utils


def get_optimal_grid_dimensions(num_images: int) -> Tuple[int, int]:
    """Calculate optimal m x n grid dimensions for displaying images. Tries to
    make the grid as square as possible.

    Args:
        num_images: Number of images to display

    Returns:
        Tuple of (rows, cols) for the grid
    """
    # Calculate square root and round up for columns
    sqrt_n = math.sqrt(num_images)
    cols = math.ceil(sqrt_n)
    rows = math.ceil(num_images / cols)

    return rows, cols


def clean_environment_name(filename: str) -> str:
    """Extract and clean the environment name from the filename.

    Args:
        filename: The image filename (e.g., "ants.png")

    Returns:
        Cleaned environment name suitable for LaTeX caption
    """
    # Remove file extension
    env_name = filename.split('.')[0]

    # Capitalize first letter
    env_name = env_name.capitalize()

    return env_name


def generate_latex_figure() -> str:
    """Generate LaTeX code for a figure containing all environment images.

    Returns:
        LaTeX code as a string
    """
    # Get path to images directory
    images_dir = Path(
        os.path.join(utils.get_path_to_predicators_root(), "images",
                     "all_envs_demo"))

    # Get all image files
    image_files = sorted([
        f.name for f in images_dir.iterdir()
        if f.is_file() and f.suffix.lower() in ['.png', '.jpg', '.jpeg']
    ])

    if not image_files:
        raise ValueError(f"No image files found in {images_dir}")

    num_images = len(image_files)
    rows, cols = 3, 4

    print(f"Found {num_images} images")
    print(f"Creating {rows}x{cols} grid layout")
    print(f"Images: {', '.join(image_files)}")

    # Calculate width for each subfigure (accounting for spacing)
    subfig_width = 0.95 / cols  # Leave some margin

    # Start building LaTeX code
    latex_code = []

    # Figure environment with caption
    latex_code.append("\\begin{figure}[htbp]")
    latex_code.append("    \\centering")

    # Process images in grid layout
    for i, image_file in enumerate(image_files):
        # Start new row if needed
        if i % cols == 0 and i > 0:
            latex_code.append("    \\\\")  # New row
            latex_code.append(
                "    \\vspace{0.2cm}")  # Add some vertical spacing

        # Add subfigure
        env_name = clean_environment_name(image_file)
        latex_code.append(
            f"    \\begin{{subfigure}}[b]{{{subfig_width:.3f}\\textwidth}}")
        latex_code.append("        \\centering")
        latex_code.append("        \\includegraphics[width=\\textwidth]"
                          f"{{figures/all_envs_demo/{image_file}}}")
        latex_code.append(f"        \\caption{{{env_name}}}")
        latex_code.append("    \\end{subfigure}")

        # Add horizontal spacing between subfigures (except for last in row)
        if (i + 1) % cols != 0 and i < num_images - 1:
            latex_code.append("    \\hfill")

    # Main figure caption and label
    latex_code.append("    \\caption{Overview of simulation environments "
                      "used in the experiments. Each subfigure shows a "
                      "representative image from the corresponding "
                      "environment.}")
    latex_code.append("    \\label{fig:environments}")
    latex_code.append("\\end{figure}")

    return "\n".join(latex_code)


def save_latex_to_file(
        latex_code: str,
        output_filename: str = "environments_figure.tex") -> None:
    """Save the generated LaTeX code to a file.

    Args:
        latex_code: The LaTeX code to save
        output_filename: Name of the output file
    """
    output_path = Path(
        os.path.join(utils.get_path_to_predicators_root(), "scripts",
                     "plotting", "mara", output_filename))

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(latex_code)

    print(f"\nLaTeX code saved to: {output_path}")


def create_environment_latex_figure() -> None:
    """Main function to create and save the LaTeX figure for environment
    images."""
    try:
        latex_code = generate_latex_figure()

        print("\n" + "=" * 60)
        print("GENERATED LATEX CODE:")
        print("=" * 60)
        print(latex_code)
        print("=" * 60)

        # Save to file
        save_latex_to_file(latex_code)

        print("\nNote: Make sure to include the following"
              " packages in your LaTeX document:")
        print("\\usepackage{graphicx}")
        print("\\usepackage{subcaption}")

    except Exception as e:  # pylint: disable=broad-except
        print(f"Error generating LaTeX figure: {e}")


if __name__ == "__main__":
    create_environment_latex_figure()
