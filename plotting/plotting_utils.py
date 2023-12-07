import os

import imageio


def setup_video(output_path, name_suffix, name_prefix, fps):
    os.makedirs(output_path, exist_ok=True)
    file_path = os.path.join(output_path, f"{name_prefix}rollout{name_suffix}.mp4")
    i = 0
    while os.path.isfile(file_path):
        i += 1
        file_path = os.path.join(
            output_path, f"{name_prefix}rollout{name_suffix}_{i}.mp4"
        )
    print("Record video in {}".format(file_path))
    return (
        imageio.get_writer(
            file_path, fps=fps, codec="h264", quality=10, pixelformat="yuv420p"
        ),  # yuv420p, yuvj422p
        file_path,
    )


def get_input_dict_from_params(params):
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
    input_dict["non_dist"] = (
        False if params["controller"] == "mpc-distance-rair-icem-torch" else True
    )
    return input_dict
