import numpy as np


def construction_success_hook(_locals, _globals, **kwargs):
    logger = _locals["rollout_man"].logger
    metrics = _locals["metrics"]
    env = _locals["env"]
    latest_rollouts = _locals["buffer"]["rollouts"].latest_rollouts
    assert env.name == "FetchPickAndPlaceConstruction"

    success_rate = []
    for i in range(len(latest_rollouts)):
        rollout_success = env.eval_success(latest_rollouts[i]["next_observations"])
        stable_T = 10
        if "tower" in env.case or "Pyramid" in env.case:
            # Stack is only successful if we have a full tower!
            # Check if the tower is stable for at least stable_T timesteps
            dy = np.diff(rollout_success)
            success = np.logical_and(rollout_success[1:] == env.num_blocks, dy == 0)
            success_rate.append(np.sum(success) > stable_T)
        elif env.case == "PickAndPlace":
            # We determine success as highest number of solved elements with at least 5 timesteps of success
            u, c = np.unique(rollout_success, return_counts=True)
            # u: unique values, c: counts
            # count_of_highest_success = c[np.argmax(u)]
            success_rate.append(u[c > 1][-1] / env.nObj)
        else:
            # For flip, throw and Playground env push tasks: just get success at the end of rollout
            success_rate.append(rollout_success[-1] / env.nObj)

    success = np.asarray(success_rate) * 1.0
    mean_success = success.mean()

    metrics["mean_success"] = mean_success

    logger.log(mean_success, key="mean_success")

    print(
        "Success rate over {} rollouts is {}".format(
            len(latest_rollouts), np.asarray(success_rate).mean()
        )
    )
