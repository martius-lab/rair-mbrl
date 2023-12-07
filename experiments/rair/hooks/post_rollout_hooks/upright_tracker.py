import numpy as np


def upright_tracker_hook(_locals, _globals, **kwargs):
    logger = _locals["rollout_man"].logger
    metrics = _locals["metrics"]
    env = _locals["env"]
    latest_rollouts = _locals["buffer"]["rollouts"].latest_rollouts

    upright_reward = env._upright_reward_from_obs(latest_rollouts["observations"][..., env.agent_dim - 1])

    factor_for_relative_scaling = latest_rollouts["observations"].shape[0]
    rel_stand_time = np.sum(upright_reward >= 0.7) / factor_for_relative_scaling
    rel_lie_back_time = np.sum(upright_reward <= 0.3) / factor_for_relative_scaling
    rel_mid_air_time = np.sum(np.logical_and(upright_reward > 0.3, upright_reward < 0.7)) / factor_for_relative_scaling

    metrics["rel_stand_time"] = rel_stand_time
    metrics["rel_lie_back_time"] = rel_lie_back_time
    metrics["rel_mid_air_time"] = rel_mid_air_time

    logger.log(rel_stand_time, key="rel_stand_time")
    logger.log(rel_lie_back_time, key="rel_lie_back_time")
    logger.log(rel_mid_air_time, key="rel_mid_air_time")
