import os

import hickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import smart_settings
from PIL import Image, ImageDraw, ImageFont
from plotting_utils import get_input_dict_from_params, setup_video

from mbrl.environments import env_from_string
from mbrl.rair_utils import model_relational_rair

font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 45)

matplotlib.use("Qt5Agg")

### --------- INPUT THE RUN NAME  --------- ###
run_name = "gt_model/gt_quadruped_rair_relational"

### ------ INPUT VIDEO OR IMAGE MODE ------ ###
mode = "img"  # video or img
display_rair_value = False
### --------------------------------- ------ ###


working_dir = f"results/rair/roboyoga/{run_name}"

# Load settings!
params = smart_settings.load(
    os.path.join(working_dir, "settings.json"), make_immutable=False
)
# Make environment!
env = env_from_string(params.env, **params["env_params"])

# Set figure path!
fig_path = f"{working_dir}/{mode}s"
os.makedirs(fig_path, exist_ok=True)

# Load buffer!
buffer = hickle.load(os.path.join(working_dir, "checkpoints_latest/rollouts"))
ep_length = buffer[0]["observations"].shape[0]

assert len(buffer) < 50  # Specify iterations for free play runs!

# Set camera settings!
render_width = 400
render_height = 400

camera_id = 0 if params.env == "walker" else 2  # 2 for quadruped, 0 for walker

file_suffix = "_w_rair" if display_rair_value else ""

if mode == "img":
    fig = plt.figure(figsize=(4, 4))
    data = np.zeros((render_width, render_height))
    im = plt.imshow(data, vmin=0, vmax=1)
    plt.axis("off")
    fig.tight_layout()

for i in range(len(buffer)):
    env.reset()
    if display_rair_value or mode == "img":
        costs_rollout = model_relational_rair(
            buffer[i]["observations"],
            env,
            # non_dist=True if params["controller"] == "mpc-relational-rair-icem-torch" else False,
            **get_input_dict_from_params(params),
        )
    if mode == "video":
        # File name in setup video is: "{name_prefix}rollout{name_suffix}.mp4"
        video, video_path = setup_video(
            fig_path,
            name_suffix=f"_{i}{file_suffix}",
            name_prefix=f"{params.env}_",
            fps=15,
        )

        for t in range(ep_length):
            env.set_GT_state(buffer[i]["env_states"][t, :])
            frame = env.dmcenv.physics.render(
                height=render_height, width=render_width, camera_id=camera_id
            )
            if display_rair_value:
                img = Image.fromarray(frame)
                draw = ImageDraw.Draw(img)
                label = "RaIR: {:.2f}".format(costs_rollout[t])
                draw.text((0, 0), label, (255, 255, 255), font=font)

                video.append_data(np.array(img))
            else:
                video.append_data(frame)

        video.close()
    else:
        env.set_GT_state(buffer[i]["env_states"][-1, :])
        data = env.dmcenv.physics.render(
            height=render_height, width=render_width, camera_id=camera_id
        )
        im.set_data(data)
        fig.savefig(
            os.path.join(fig_path, f"rollout{i}_end.png"),
            bbox_inches="tight",
            dpi=300,
        )

        env.set_GT_state(buffer[i]["env_states"][0, :])
        data = env.dmcenv.physics.render(
            height=render_height, width=render_width, camera_id=camera_id
        )
        im.set_data(data)
        fig.savefig(
            os.path.join(fig_path, f"rollout{i}_beginning.png"),
            bbox_inches="tight",
            dpi=300,
        )

        t = np.argsort(costs_rollout)[0]
        env.set_GT_state(buffer[i]["env_states"][t, :])
        data = env.dmcenv.physics.render(
            height=render_height, width=render_width, camera_id=camera_id
        )
        im.set_data(data)
        fig.savefig(
            os.path.join(fig_path, f"rollout{i}_highest_rair_t{t}.png"),
            bbox_inches="tight",
            dpi=300,
        )

if mode == "img":
    plt.close(fig)

env.close()
