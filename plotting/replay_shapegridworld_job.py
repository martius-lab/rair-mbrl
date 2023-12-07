import os

import hickle
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import smart_settings

from mbrl.environments import env_from_string
from mbrl.rair_utils import model_relational_rair

matplotlib.use("Qt5Agg")

mode = "img"  # video or img
run_name = "gt_model/gt_rair_relational_20obj"

working_dir = f"results/rair/shape_gridworld/{run_name}"

params = smart_settings.load(
    os.path.join(working_dir, "settings.json"), make_immutable=False
)
if mode == "video":
    params["env_params"]["render_delta"] = 30
    fig_path = f"results/rair/shape_gridworld/{run_name}/videos"
else:
    params["env_params"]["render_delta"] = 60
    fig_path = f"results/rair/shape_gridworld/{run_name}/imgs"

env = env_from_string(params.env, **params["env_params"])

os.makedirs(fig_path, exist_ok=True)


buffer = hickle.load(os.path.join(working_dir, "checkpoints_latest/rollouts"))

ep_length = buffer[0]["observations"].shape[0]


def animate(data):
    im.set_data(data)
    return im


fig = plt.figure(figsize=(4, 4))
data = np.zeros((env.width, env.height))
im = plt.imshow(data, vmin=0, vmax=1)
plt.axis("off")

for i in range(len(buffer)):
    env.reset()
    if mode == "video":
        im_list_new = []
        for t in range(ep_length):
            env.set_state_from_observation(buffer[i]["observations"][t, :])
            im_list_new.append(env.render())

        # Create the animation object
        anim = animation.FuncAnimation(
            fig, animate, frames=im_list_new, interval=50, repeat=False
        )
        # interval is the pause time in milliseconds
        anim.save(
            os.path.join(fig_path, f"animation_replayed_gt_compression_rollout{i}.mp4"),
            writer="ffmpeg",
            fps=80,
        )
    else:
        env.set_state_from_observation(buffer[i]["observations"][-1, :])
        data = env.render()
        im.set_data(data)
        fig.savefig(
            os.path.join(fig_path, f"rollout{i}_end.png"),
            bbox_inches="tight",
            dpi=300,
        )

        env.set_state_from_observation(buffer[i]["observations"][0, :])
        data = env.render()
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
        env.set_state_from_observation(buffer[i]["observations"][t, :])
        data = env.render()
        im.set_data(data)
        fig.savefig(
            os.path.join(fig_path, f"rollout{i}_highest_rair_t{t}.png"),
            bbox_inches="tight",
            dpi=300,
        )

plt.close(fig)

env.close()
