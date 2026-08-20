"""Build the before/after 3x3 mosaics for the weekly-sync deck.

Rows are seeds 0/1/2, columns are the three 2026-07-29 launches of
``domino_high_friction_turn-agent_po_predicate_invention_al_margin``.
The "before" grid takes each run's FIRST cycle-0 explore episode (plans
built at the believed friction 0.1); the "after" grid takes its test
episode (planned on the calibrated belief). Both come straight from the
episode videos main.py saves per episode - no re-simulation.

Sources are 912x912 at 20 fps. Each tile is cropped to the table region
(the right third of the frame is empty floor plus the arm's shoulder),
downscaled once with lanczos, and given a hairline border; clips shorter
than the longest one hold their final frame so the grid loops together.

MP4 is the deck format: H.264 keeps the dominoes' actual colors at full
mosaic resolution for ~2 MB, where the old 762 px GIF spent its 256
palette entries on the tan table and turned the green domino grey. A GIF
of this mosaic at full size and frame rate measures 193 MB, so ``--gif``
writes the usable compromise instead - 900 px at 10 fps, ~6.5 MB - for
contexts that cannot embed video.

Usage:
    python docs/slides/assets/make_al_margin_mosaics.py [--gif]
"""
import argparse
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
VIDEOS = (REPO / "videos" / "agent_po_sim_predicate_invention" /
          "domino_high_friction_turn-agent_po_predicate_invention_al_margin")

LAUNCHES = [
    "run_20260729_125953", "run_20260729_232733", "run_20260729_232741"
]
SEEDS = [0, 1, 2]
# (output stem, episode suffix of the source clip)
GRIDS = [("al_margin_before_learning_3x3", "ep0__cycle0"),
         ("al_margin_after_learning_3x3", "task1__cycle0")]

# Crop applied to every 912x912 source frame: keeps the table, the
# dominoes and the gripper, drops the empty floor on the right.
CROP_W, CROP_H, CROP_X, CROP_Y = 720, 720, 0, 170
TILE = 456  # px per tile after downscale; 3 tiles -> ~1.4k mosaic
BORDER = 3  # hairline separator between tiles
FPS = 20
# --gif only: a full-resolution 20 fps GIF of this mosaic is 193 MB.
GIF_WIDTH, GIF_FPS = 900, 10


def _clip(seed: int, launch: str, suffix: str) -> Path:
    """The one video for this (seed, launch, episode role)."""
    run_dir = VIDEOS / f"seed{seed}" / launch
    matches = sorted(run_dir.glob(f"*__{suffix}.mp4"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one *__{suffix}.mp4 in {run_dir}, "
            f"found {[m.name for m in matches]}")
    return matches[0]


def _mosaic_filter(n_inputs: int) -> str:
    """Crop/scale/pad each input, then xstack them into a 3x3 grid."""
    step = TILE + 2 * BORDER
    per_input = "".join(
        f"[{i}:v]crop={CROP_W}:{CROP_H}:{CROP_X}:{CROP_Y},"
        f"scale={TILE}:{TILE}:flags=lanczos,fps={FPS},"
        f"pad={step}:{step}:{BORDER}:{BORDER}:white,"
        # Hold the last frame so short clips stay in step with long ones.
        f"tpad=stop_mode=clone:stop_duration=600[v{i}];"
        for i in range(n_inputs))
    layout = "|".join(f"{(i % 3) * step}_{(i // 3) * step}"
                      for i in range(n_inputs))
    return (f"{per_input}"
            f"{''.join(f'[v{i}]' for i in range(n_inputs))}"
            f"xstack=inputs={n_inputs}:layout={layout}:shortest=0[grid]")


def _duration(ffmpeg: str, path: Path) -> float:
    """Seconds of video, parsed from ffmpeg's own stream report."""
    out = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", str(path)],
        capture_output=True,
        text=True,
        check=False).stderr
    for line in out.splitlines():
        if "Duration:" in line:
            hh, mm, ss = line.split("Duration:")[1].split(",")[0].split(":")
            return int(hh) * 3600 + int(mm) * 60 + float(ss)
    raise RuntimeError(f"no duration reported for {path}")


def build(stem: str, suffix: str, also_gif: bool) -> None:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    clips = [
        _clip(seed, launch, suffix) for seed in SEEDS for launch in LAUNCHES
    ]
    longest = max(_duration(ffmpeg, c) for c in clips)
    mp4 = HERE / f"{stem}.mp4"
    inputs = [arg for c in clips for arg in ("-i", str(c))]
    subprocess.run([
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error", *inputs,
        "-filter_complex",
        _mosaic_filter(len(clips)), "-map", "[grid]", "-t", f"{longest:.2f}",
        "-c:v", "libx264", "-preset", "slow", "-crf", "20", "-pix_fmt",
        "yuv420p", "-movflags", "+faststart",
        str(mp4)
    ],
                   check=True)
    print(f"wrote {mp4} ({mp4.stat().st_size // 1024} KB)")

    if not also_gif:
        return
    gif = HERE / f"{stem}.gif"
    # Cropping to the table is what saves the colors here: the palette no
    # longer has to cover a frame-filling floor as well. stats_mode=diff
    # weights the moving region, where per-frame palettes (new=1) would
    # triple the file size for a mosaic this large.
    subprocess.run([
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i",
        str(mp4), "-filter_complex",
        f"fps={GIF_FPS},scale={GIF_WIDTH}:-1:flags=lanczos,split[a][b];"
        "[a]palettegen=stats_mode=diff:max_colors=256[p];"
        "[b][p]paletteuse=dither=sierra2_4a", "-loop", "0",
        str(gif)
    ],
                   check=True)
    print(f"wrote {gif} ({gif.stat().st_size // 1024} KB)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gif",
                        action="store_true",
                        help="also write a (larger, lower-fidelity) GIF")
    args = parser.parse_args()
    if not VIDEOS.is_dir():
        print(f"missing episode videos: {VIDEOS}", file=sys.stderr)
        return 1
    for stem, suffix in GRIDS:
        build(stem, suffix, args.gif)
    return 0


if __name__ == "__main__":
    sys.exit(main())
