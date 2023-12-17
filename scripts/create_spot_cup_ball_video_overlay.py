"""Create video overlays from saved info in the active sampler explorer."""

from pathlib import Path
from typing import Dict, Set

import dill as pkl
import imageio.v2 as iio
import matplotlib
import matplotlib.pyplot as plt
from moviepy.editor import ImageClip, concatenate_videoclips

from predicators import utils
from predicators.competence_models import OptimisticSkillCompetenceModel
from predicators.structs import Image, _GroundSTRIPSOperator

matplotlib.rcParams.update({'font.size': 18})

_SKILL_NAME_REPLACEMENTS = {
    "MoveToHandViewObject(robot, ball)": "MoveToView\n(robot, ball)",
    "MoveToHandViewObject(robot, cup)": "MoveToView\n(robot, ring)",
    "MoveToReachObject(robot, drafting_table)": "MoveToReach\n(robot, table)",
    "PickObjectFromTop(robot, ball, cup)": "Pick\n(robot, ball, ring)",
    "PickObjectFromTop(robot, ball, floor)": "Pick\n(robot, ball, floor)",
    "PlaceObjectOnTop(robot, ball, drafting_table)":
    "Place\n(robot, ball, table)",
}


def _create_image(data: Dict,
                  known_skills: Set[_GroundSTRIPSOperator]) -> Image:
    competence_models = data["competence_models"]
    sorted_skills = sorted(known_skills)
    practice_nsrt = data["next_practice_nsrt"]
    if practice_nsrt:
        practice_id = (practice_nsrt.name, practice_nsrt.objects)
    else:
        practice_id = None

    scale = 3
    fig, axes = plt.subplots(1,
                             len(sorted_skills),
                             figsize=(scale * len(sorted_skills), scale))
    bar_colors = ['tab:red', 'tab:blue']

    for i, (skill, ax) in enumerate(zip(sorted_skills, axes.flat)):
        if skill not in competence_models:
            competence_model = OptimisticSkillCompetenceModel("default")
        else:
            competence_model = competence_models[skill]
        values = [
            competence_model.get_current_competence(),
            competence_model.predict_competence(1),
        ]

        obj_str = ", ".join([o.name for o in skill.objects])
        skill_title = f"{skill.name}({obj_str})"
        skill_title = _SKILL_NAME_REPLACEMENTS.get(skill_title, skill_title)
        title_kwargs = {}
        if (skill.name, skill.objects) == practice_id:
            title_kwargs["fontweight"] = "bold"
        ax.set_title(skill_title, **title_kwargs)
        ax.bar(["Current", "Extrap"], values, color=bar_colors)
        if i == 0:
            ax.set_ylabel("Competence")

    plt.tight_layout()
    img = utils.fig2data(fig, dpi=150)
    plt.close()

    iio.imsave("example_overlay.png", img)

    return img


def _main() -> None:
    # Load the data and collect all the known skills.
    data_dir = Path("saved_datasets")
    timestamp_to_data = {}
    known_skills = set()
    for fp in data_dir.glob(
            "spot_ball_and_cup_sticky_table_env_explorer_timestamped_info_*.data"  # pylint: disable=line-too-long
    ):
        with open(fp, "rb") as f:
            save_dict = pkl.load(f)
        timestamp_to_data[save_dict["timestamp"]] = save_dict
        known_skills.update(save_dict["competence_models"].keys())
    # Create the images.
    timestamp_to_image = {}
    for timestamp, data in timestamp_to_data.items():
        print(f"Creating image for {timestamp}...")
        image = _create_image(data, known_skills)
        timestamp_to_image[timestamp] = image
    # Sort the images in timestamp order and record the durations.
    sorted_timestamps = sorted(timestamp_to_image)
    durations = [
        (t1 - t0).total_seconds()
        for t0, t1 in zip(sorted_timestamps[:-1], sorted_timestamps[1:])
    ]
    durations.append(1.0)  # default 1 second for the last frame
    # Create the video.
    print("Creating the video...")
    clips = []
    for (timestamp, duration) in zip(sorted_timestamps, durations):
        image = timestamp_to_image[timestamp]
        clip = ImageClip(image, duration=duration)
        clips.append(clip)
    final_clip = concatenate_videoclips(clips, method="compose")
    final_clip.write_videofile("output_video.mp4", fps=15)


if __name__ == "__main__":
    _main()
