"""Generate GIFs of random actions on all PyBullet environments.

Runs the random_actions_pybullet approach on each environment to produce MP4
videos, then converts them to GIFs and places them in
docs/envs/assets/random_action_gifs/.

Usage:
    PYTHONPATH=. python scripts/local/generate_random_action_gifs.py

Options:
    --skip-run      Skip running the experiments (just convert existing MP4s)
    --config, -c    Config file to use
                    (default: mara2/random_actions_pybullet.yaml)
    --video-dir     Directory where MP4s are written (default: videos)
    --output-dir    Directory for output GIFs
                    (default: docs/envs/assets/random_action_gifs)
    --fps           GIF frames per second (default: 20)
    --width         Resize GIF width in pixels (default: 480)
"""
import argparse
import glob
import os
import subprocess
import sys


def run_experiments(config: str) -> None:
    """Run launch_simp.py to generate MP4 videos."""
    cmd = [
        sys.executable,
        "scripts/local/launch_simp.py",
        "-c",
        config,
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=False)


def mp4_to_gif(
    mp4_path: str,
    gif_path: str,
    fps: int = 20,
    width: int = 480,
) -> bool:
    """Convert an MP4 file to an optimized GIF using ffmpeg."""
    os.makedirs(os.path.dirname(gif_path), exist_ok=True)
    # Two-pass ffmpeg: generate palette then use it for high-quality GIF
    # Simpler single-pass approach that's more robust
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        mp4_path,
        "-vf",
        (f"fps={fps},scale={width}:-1:flags=lanczos,"
         "split[s0][s1];[s0]palettegen=stats_mode=diff[p];"
         "[s1][p]paletteuse=dither=bayer:bayer_scale=5:"
         "diff_mode=rectangle"),
        "-loop",
        "0",
        gif_path,
    ]
    try:
        _result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ffmpeg failed for {mp4_path}: {e.stderr[-200:]}")
        return False


def find_mp4s(video_dir: str) -> dict[str, str]:
    """Find MP4 files and map environment names to file paths.

    Returns a dict of {env_short_name: mp4_path}.
    The naming convention from the framework is:
        {env}__{approach}__{seed}__{excluded}__{included}__
        {experiment_id}__task{n}.mp4
    We extract the env name (e.g. 'pybullet_cover') from the filename.
    """
    mp4s: dict[str, str] = {}
    pattern = os.path.join(video_dir, "*.mp4")
    for path in sorted(glob.glob(pattern)):
        basename = os.path.basename(path)
        # Extract env name: first segment before "__"
        parts = basename.split("__")
        if len(parts) >= 2:
            env_name = parts[0]
        else:
            env_name = os.path.splitext(basename)[0]
        # If multiple mp4s for same env (e.g. failure + success), prefer
        # non-failure version
        if "_failure" in basename:
            if env_name not in mp4s:
                mp4s[env_name] = path
        else:
            mp4s[env_name] = path
    return mp4s


def main() -> None:
    """Run the GIF generation pipeline."""
    parser = argparse.ArgumentParser(
        description="Generate random-action GIFs for PyBullet environments.")
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Skip running experiments; only convert existing MP4s to GIFs.",
    )
    parser.add_argument(
        "-c",
        "--config",
        default="mara2/random_actions_pybullet.yaml",
        help="Config YAML file (relative to scripts/configs/).",
    )
    parser.add_argument(
        "--video-dir",
        default="videos",
        help="Directory where MP4 videos are written.",
    )
    parser.add_argument(
        "--output-dir",
        default="docs/envs/assets/random_action_gifs",
        help="Output directory for GIFs.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="GIF frames per second.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=480,
        help="GIF width in pixels (height auto-scaled).",
    )
    args = parser.parse_args()

    # Step 1: Run experiments to generate MP4 videos
    if not args.skip_run:
        run_experiments(args.config)
    else:
        print("Skipping experiment run (--skip-run).")

    # Step 2: Find generated MP4 files
    mp4s = find_mp4s(args.video_dir)
    if not mp4s:
        print(f"No MP4 files found in {args.video_dir}/. Nothing to convert.")
        sys.exit(1)

    print(f"\nFound {len(mp4s)} environment video(s):")
    for env_name in sorted(mp4s):
        print(f"  {env_name}: {mp4s[env_name]}")

    # Step 3: Convert to GIFs
    os.makedirs(args.output_dir, exist_ok=True)
    successes = 0
    failures = 0
    for env_name in sorted(mp4s):
        mp4_path = mp4s[env_name]
        gif_path = os.path.join(args.output_dir, f"{env_name}.gif")
        print(f"\nConverting {env_name}...")
        if mp4_to_gif(mp4_path, gif_path, fps=args.fps, width=args.width):
            size_kb = os.path.getsize(gif_path) / 1024
            print(f"  -> {gif_path} ({size_kb:.0f} KB)")
            successes += 1
        else:
            failures += 1

    print(f"\nDone: {successes} GIF(s) generated, {failures} failure(s).")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
