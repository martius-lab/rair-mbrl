import numpy as np

from mbrl.rair_utils import model_relational_rair


def get_best_rair(observations, env, num_rollouts, ep_length, compression_ndim, granularity, precision=100):
    costs_next_obs = model_relational_rair(
        observations,
        env,
        compression_ndim=compression_ndim,
        bidirectional=False,
        granularity=granularity,
        precision=precision,
    )
    costs_next_obs = (-1) * costs_next_obs.reshape(num_rollouts, ep_length)
    best_costs_per_rollout = np.amax(costs_next_obs, axis=1)
    return np.mean(best_costs_per_rollout), np.mean(costs_next_obs[:, 0])


def rair_tracker_hook(_locals, _globals, **kwargs):
    logger = _locals["rollout_man"].logger
    metrics = _locals["metrics"]
    env = _locals["env"]
    latest_rollouts = _locals["buffer"]["rollouts"].latest_rollouts

    num_rollouts = _locals["params"]["number_of_rollouts"]
    ep_length = _locals["params"]["rollout_params"]["task_horizon"]

    best_rair_xy, baseline_rair_xy = get_best_rair(
        latest_rollouts["observations"], env, num_rollouts, ep_length, compression_ndim=2, granularity=5, precision=100
    )
    best_rair_xyz, baseline_rair_xyz = get_best_rair(
        latest_rollouts["observations"], env, num_rollouts, ep_length, compression_ndim=3, granularity=5, precision=100
    )

    metrics["best_rair_xy"] = best_rair_xy
    metrics["baseline_rair_xy"] = baseline_rair_xy
    metrics["best_rair_xyz"] = best_rair_xyz
    metrics["baseline_rair_xyz"] = baseline_rair_xyz

    logger.log(best_rair_xy, key="best_rair_xy")
    logger.log(baseline_rair_xy, key="baseline_rair_xy")
    logger.log(best_rair_xyz, key="best_rair_xyz")
    logger.log(baseline_rair_xyz, key="baseline_rair_xyz")
