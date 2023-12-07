import functools
import multiprocessing as mp
import os
from contextlib import contextmanager
from itertools import chain
from typing import Sequence, Tuple

import numpy as np
from forwardable import def_delegators, forwardable

from mbrl.controllers.abstract_controller import ParallelController
from mbrl.environments.abstract_environments import GroundTruthSupportEnv
from mbrl.models.gt_model import (
    AbstractGroundTruthModel,
    GroundTruthEnsembleModel,
    GroundTruthModel,
)
from mbrl.parallel_utils import CloudPickleWrapper, clear_mpi_env_vars
from mbrl.rolloutbuffer import RolloutBuffer, SimpleRolloutBuffer
from mbrl.seeding import Seeding


class ParallelGroundTruthModel(AbstractGroundTruthModel):
    """
    Parallel model might be an ensemble model
    """

    simulated_env: GroundTruthSupportEnv

    def __init__(self, num_parallel=4, ensemble_size=None, **kwargs):
        super().__init__(**kwargs)
        self.waiting = False
        self.closed = False
        self.num_parallel = num_parallel
        self.ensemble_size = ensemble_size
        ctx = mp.get_context("fork")
        self.remotes, self.work_remotes = zip(*[ctx.Pipe(duplex=True) for _ in range(self.num_parallel)])
        self.wrapped_class = GroundTruthEnsembleModel if ensemble_size else GroundTruthModel

        self.ps = [
            ctx.Process(
                target=worker,
                args=(
                    Seeding.SEED,
                    work_remote,
                    remote,
                    CloudPickleWrapper(self.env),
                    ensemble_size,
                ),
            )
            for (work_remote, remote) in zip(self.work_remotes, self.remotes)
        ]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            with clear_mpi_env_vars():
                p.start()
        for remote in self.work_remotes:
            remote.close()

        self.is_trained = True

    def train(self, rollout_buffer: RolloutBuffer):
        pass

    @property
    def __subclasses__(self):
        return self.__subclasses__ + self.wrapped_class.__subclasses__ + (self.wrapped_class,)

    def predict(self, *, observations, states, actions):
        self.remotes[0].send(("predict", (observations, states, actions)))
        return self.remotes[0].recv()

    def reset(self, observation):
        _ = [remote.send(("reset", observation)) for remote in self.remotes]
        init_states = [remote.recv() for remote in self.remotes]
        return init_states[0]

    def set_state(self, state):
        raise NotImplementedError

    def get_state(self):
        self.remotes[0].send(("get_state", None))
        return self.remotes[0].recv()

    def got_actual_observation_and_env_state(self, *, observation, env_state=None, model_state=None):
        if env_state is None:
            return self.reset(observation)
        else:
            return env_state

    def predict_n_steps(
        self, *, start_observations: np.ndarray, start_states: Sequence, policy: ParallelController, horizon
    ) -> Tuple[RolloutBuffer, Sequence]:
        # here we want to step through the envs in the direction of time
        if start_observations.ndim != 2:
            raise AttributeError("call predict_n_steps with a batch of states")
        if len(start_observations) != len(start_states):
            raise AttributeError("number of observations and states have to be the same")
        if start_states[0] is None:
            raise NotImplementedError

        num_simulations = start_observations.shape[0]
        chunks = np.array_split(range(num_simulations), self.num_parallel)
        chunks = [c for c in chunks if len(c) > 0]
        policies = [policy.get_parallel_policy_copy(c) for c in chunks]
        start_obs_chunks = [start_observations[c] for c in chunks]
        start_state_chunks = [[start_states[i] for i in c] for c in chunks]  # lists do not support slicing

        asked_remotes = []
        for remote, s_obs, s_states, sub_policy in zip(self.remotes, start_obs_chunks, start_state_chunks, policies):
            remote.send(("simulate", (s_obs, s_states, sub_policy, horizon)))
            asked_remotes.append(remote)
        rollout_buffer_states_list = [remote.recv() for remote in asked_remotes]
        if "MujocoException" in rollout_buffer_states_list:
            from mujoco_py import MujocoException

            raise MujocoException

        all_states = list(chain.from_iterable([s for _, s in rollout_buffer_states_list]))
        if isinstance(rollout_buffer_states_list[0][0], (RolloutBuffer)):
            all_rollouts = list(
                chain.from_iterable([rollout_buffer.rollouts for rollout_buffer, _s in rollout_buffer_states_list])
            )
            return RolloutBuffer(rollouts=all_rollouts), all_states
        else:
            rollout_buffers = [rb for rb, s_ in rollout_buffer_states_list]
            rollout_buffer = functools.reduce(lambda x, y: x + y, rollout_buffers)
            return rollout_buffer, all_states

    def save(self, path):
        pass

    def load(self, path):
        pass


def worker(seed, remote, parent_remote, env_wrapper, ensemble_size):
    parent_remote.close()
    if ensemble_size:
        gt_model = GroundTruthEnsembleModel(ensemble_size=ensemble_size, env=env_wrapper.x)
    else:
        gt_model = GroundTruthModel(env=env_wrapper.x)

    Seeding.set_seed(seed)
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "simulate":
                obs, states, policy, horizon = data
                try:
                    rollout_buffer, new_states = gt_model.predict_n_steps(
                        start_observations=obs,
                        start_states=states,
                        policy=policy,
                        horizon=horizon,
                    )
                except Exception as e:
                    if e.__class__.__name__ == "MujocoException":
                        remote.send("MujocoException")
                        continue
                    else:
                        raise e
                remote.send((rollout_buffer, new_states))
            elif cmd == "predict":
                obs, states, actions = data
                remote.send(gt_model.predict(observations=obs, states=states, actions=actions))
            elif cmd == "get_state":
                obs = data
                remote.send(gt_model.get_state())
            elif cmd == "reset":
                obs = data
                remote.send(gt_model.reset(obs))
            else:
                raise NotImplementedError("cmd: {}".format(cmd))
    except KeyboardInterrupt:
        print("ParallelModel worker: got KeyboardInterrupt")
    finally:
        gt_model.close()


# Some speed comparisons
# HumanoidStandup: 10 steps a 100 traj, 100 horizon:
# Single core: 2m36s
# 6 cores ParModel: 38s (speedup 4.1, and about 5 sec are startup overhead)
# 6 cores ParEnv: 1m40s (speedup 1.5, and about 5 sec are startup overhead)
#   -> remove parallel envs
