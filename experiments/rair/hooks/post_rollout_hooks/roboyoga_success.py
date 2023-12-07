import numpy as np


def roboyoga_success_hook(_locals, _globals, **kwargs):
    logger = _locals["rollout_man"].logger
    metrics = _locals["metrics"]
    env = _locals["env"]
    latest_rollouts = _locals["buffer"]["rollouts"].latest_rollouts

    costs = env.cost_fn(
        latest_rollouts["observations"].reshape(
            (-1,) + latest_rollouts[0]["observations"].shape
        ),
        latest_rollouts["actions"].reshape((-1,) + latest_rollouts[0]["actions"].shape),
        latest_rollouts["next_observations"].reshape(
            (-1,) + latest_rollouts[0]["next_observations"].shape
        ),
    )
    # costs:  [num_rollouts, task_horizon]

    returns_rewards = np.sum(-costs, -1)
    # returns_rewards: [num_rollouts]

    average_return = np.mean(returns_rewards)

    stable_T = 10
    success_rate = []
    for i in range(len(latest_rollouts)):
        rollout_success = costs[i, :] < 0.4
        dy = np.diff(rollout_success)
        success = np.logical_and(rollout_success[1:] == 1, dy == 0)
        success_rate.append(np.sum(success) > stable_T)
    success = np.asarray(success_rate) * 1.0
    mean_success = success.mean()

    metrics["average_return"] = average_return
    metrics["mean_success"] = mean_success

    logger.log(average_return, key="average_return")
    logger.log(mean_success, key="mean_success")

    # print("Return: {}".format(average_return))
    # print("Success rate: {}".format(mean_success))
