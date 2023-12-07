import numpy as np
import torch
from gym.utils import EzPickle

from mbrl.environments.abstract_environments import GroundTruthSupportEnv
from mbrl.environments.shape_gridworld.shape_gridworld_colored import ShapeGridworld


class ShapeGridworldColoredEnv(GroundTruthSupportEnv, ShapeGridworld):
    def __init__(self, name, **kwargs):

        GroundTruthSupportEnv.__init__(self, name=name, **kwargs)
        ShapeGridworld.__init__(self, **kwargs)
        EzPickle.__init__(self, name=name, **kwargs)

        self.agent_dim = 0
        self.object_dyn_dim = 2 + self.color_feature_size  # 2 for position + color size
        self.object_stat_dim = 0
        self.nObj = len(self.objects)

    def get_GT_state(self):
        # Object positions:
        state_obs = np.asarray(self.objects, dtype=np.float32)
        # Put object colors as well!
        state_obs = np.concatenate((state_obs, self.object_colors), axis=1).flatten()

        return np.hstack((state_obs, self.current_object, self.current_object_t))

    def set_GT_state(self, state):
        self.current_object = int(state[-2])
        self.current_object_t = int(state[-1])
        obj_state = state[:-2].reshape(-1, self.object_dyn_dim)[..., :2].tolist()

        for i in range(self.num_objects):
            self.objects[i] = obj_state[i].copy()

    def set_state_from_observation(self, observation):
        self.set_GT_state(observation)


if __name__ == "__main__":
    env = ShapeGridworldColoredEnv(
        name="ShapeGridworld",
        width=50,
        height=50,
        render_type="circles",
        render_delta=20,
        num_objects=10,
        object_persistency=10,
    )

    obs = env.reset()
    print(obs)
    # import copy
    # start_obs = copy.deepcopy(obs)

    # for _ in range(100):
    #     env.render()

    # for _ in range(200):
    #     action_tuple = np.random.uniform(low=-1.0, high=1.0, size=(len(env.agents) * 2,))
    #     env.step(action_tuple)
    #     env.render()

    # obs = env.reset()
    # print("reset the environment!")

    # for _ in range(200):
    #     env.render()

    # print("setting state from previous observation! ")
    # env.set_state_from_observation(start_obs)

    # for _ in range(200):
    #     env.render()
