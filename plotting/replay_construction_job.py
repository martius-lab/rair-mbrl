import os

import hickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import smart_settings
from mujoco_py import GlfwContext
from plotting_utils import setup_video

from mbrl.environments import env_from_string
from mbrl.rair_utils import model_relational_rair

matplotlib.use("Qt5Agg")
# Needed for the recording only if LD_PRELOAD is not unset!
GlfwContext(offscreen=True)

### ------- INPUT THE RUN NAME  -------- ###
run_name = "gt_model/gt_rair_relational_6obj"

### ------ INPUT VIDEO OR IMAGE MODE ------ ###
mode = "img"  # video or img

working_dir = f"results/rair/construction/{run_name}"

# Load settings!
params = smart_settings.load(
    os.path.join(working_dir, "settings.json"), make_immutable=False
)
# Make environment!
env = env_from_string(params.env, **params["env_params"])

# Set figure path!
fig_path = f"results/rair/construction/{run_name}/{mode}s"
os.makedirs(fig_path, exist_ok=True)

# Load buffer!
buffer = hickle.load(os.path.join(working_dir, "checkpoints_latest/rollouts"))
ep_length = buffer[0]["observations"].shape[0]

assert len(buffer) < 50  # Specify iterations for free play runs!

# Set camera settings!
render_width = 512
render_height = 512
frame = env.render("rgb_array", render_width, render_height)
# Feel free to change these values to modify the camera angle!
env.viewer.cam.azimuth = 180.16790799446062
env.viewer.cam.distance = 1.0
env.viewer.cam.elevation = -10
env.sim.forward()

if mode == "img":
    fig = plt.figure(figsize=(4, 4))
    data = np.zeros((render_width, render_height))
    im = plt.imshow(data, vmin=0, vmax=1)
    plt.axis("off")
    fig.tight_layout()

for i in range(len(buffer)):
    env.reset()
    if mode == "video":
        # File name in setup video is: "{name_prefix}rollout{name_suffix}.mp4"
        video, video_path = setup_video(
            fig_path,
            name_suffix=f"_{i}",
            name_prefix="construction_",
            fps=env.get_fps(),
        )

        for t in range(ep_length):
            env.set_GT_state(buffer[i]["env_states"][t, :])
            frame = env.render("rgb_array", render_width, render_height)
            video.append_data(frame)

            if env.name == "FetchPickAndPlaceConstruction" and env.case == "Slide":
                del env.viewer._markers[:]
        video.close()
    else:
        env.set_GT_state(buffer[i]["env_states"][-1, :])
        data = env.render("rgb_array", render_width, render_height)
        im.set_data(data)
        fig.savefig(
            os.path.join(fig_path, f"rollout{i}_end.png"),
            bbox_inches="tight",
            dpi=300,
        )

        env.set_GT_state(buffer[i]["env_states"][0, :])
        data = env.render("rgb_array", render_width, render_height)
        im.set_data(data)
        fig.savefig(
            os.path.join(fig_path, f"rollout{i}_beginning.png"),
            bbox_inches="tight",
            dpi=300,
        )

        keys_to_extract = [
            "compression_ndim",
            "granularity",
            "precision",
            "bidirectional",
            "distance",
            "rounding",
        ]

        input_dict = {
            key: params["controller_params"][key]
            for key in keys_to_extract
            if key in params["controller_params"]
        }

        costs_rollout = model_relational_rair(
            buffer[i]["observations"],
            env,
            non_dist=True
            if params["controller"] == "mpc-relational-rair-icem-torch"
            else False,
            **input_dict,
        )

        t = np.argsort(costs_rollout)[0]
        env.set_GT_state(buffer[i]["env_states"][t, :])
        data = env.render("rgb_array", render_width, render_height)
        im.set_data(data)
        fig.savefig(
            os.path.join(fig_path, f"rollout{i}_highest_rair_t{t}.png"),
            bbox_inches="tight",
            dpi=300,
        )

if mode == "img":
    plt.close(fig)

env.close()
