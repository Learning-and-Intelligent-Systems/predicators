import cv2
import imageio
import textwrap
import numpy as np


# https://gist.github.com/EricCousineau-TRI/596f04c83da9b82d0389d3ea1d782592
def draw_text(
    img,
    texts,
    uv_top_left,
    color=(255, 255, 255),
    fontScale=0.5,
    thickness=1,
    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    outline_color=(0, 0, 0),
    line_spacing=1.5,
):
    """
    Draws multiline with an outline.
    """
    uv_top_left = np.array(uv_top_left, dtype=float)
    assert uv_top_left.shape == (2,)

    for line in texts:
        (w, h), _ = cv2.getTextSize(
            text=line,
            fontFace=fontFace,
            fontScale=fontScale,
            thickness=thickness,
        )
        uv_bottom_left_i = uv_top_left + [0, h]
        org = tuple(uv_bottom_left_i.astype(int))

        if outline_color is not None:
            cv2.putText(
                img,
                text=line,
                org=org,
                fontFace=fontFace,
                fontScale=fontScale,
                color=outline_color,
                thickness=thickness * 3,
                lineType=cv2.LINE_AA,
            )
        cv2.putText(
            img,
            text=line,
            org=org,
            fontFace=fontFace,
            fontScale=fontScale,
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

        uv_top_left += [0, h * line_spacing]

name = "videos/pybullet_blocks__oracle__1________task1.mp4"



video = imageio.mimread(name, memtest=False)
with open(f"{name}.states.txt", "r", encoding="utf-8") as f:
    states = f.read().split("\n")
with open(f"{name}.options.txt", "r", encoding="utf-8") as f:
    options = f.read().split("\n")

for option, state, img in zip(options, states, video):
    text = [f"Skill: {option}", "", "Abstract State:", ""]
    # text.extend(textwrap.wrap(state, width=30))
    text.extend(x + ")" for x in state[1:-1].split("), "))
    text[-1] = text[-1][:-1]
    draw_text(img, text, (50, 140), fontScale=1.25, color=(0, 0, 0), outline_color=None, thickness=2)

final_video = [img[:, :-256] for img in video[:-1]]

imageio.mimsave("videos/modified_video.mp4", final_video, fps=30)
