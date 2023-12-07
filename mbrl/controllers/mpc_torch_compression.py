import bz2

import lz4.frame
import torch

from mbrl import torch_helpers
from mbrl.controllers.mpc_torch import TorchMpcICem
from mbrl.rolloutbuffer import RolloutBuffer


# our improved CEM
class TorchCompressionMpcICem(TorchMpcICem):
    def __init__(
        self,
        *,
        compression_ndim=2,
        precision=100,
        compresslevel=9,
        compression_algo="bzip2",  # or lz4
        ensemble_disagreement=False,
        ensemble_disagreement_scale=1.0,
        extrinsic_reward=False,
        extrinsic_reward_scale=1.0,
        **kwargs,
    ):
        self.compression_ndim = compression_ndim
        self.precision = precision
        self.compresslevel = compresslevel
        self.compression_algo = compression_algo
        self._w_ensemble_disagreement = ensemble_disagreement
        self._maybe_ensemble_disagreement_scale = ensemble_disagreement_scale
        self._w_extrinsic_reward = extrinsic_reward
        self._maybe_extrinsic_reward_scale = extrinsic_reward_scale
        super().__init__(**kwargs)

        if not self._ensemble_size:
            self._w_ensemble_disagreement = False

    @torch.no_grad()
    def preallocate_memory(self):
        """
        Preallocate memory for distribution parameters in addition
        """
        super().preallocate_memory()

        self._compression_cost_per_path = torch.zeros(
            (
                int(
                    self.num_sim_traj
                    + self.fraction_elites_reused
                    * self.elites_size
                    * self.keep_previous_elites
                ),
                self.horizon,
            ),
            device=torch_helpers.device,
            dtype=torch.float32,
            requires_grad=False,
        )

        if self._w_ensemble_disagreement:
            self.stds_of_means_ = torch.zeros(
                (
                    self.num_sim_traj,
                    self.horizon,
                    self.env.observation_space.shape[0],
                ),
                device=torch_helpers.device,
                dtype=torch.float32,
                requires_grad=False,
            )

            self._epistemic_bonus_per_path = torch.zeros(
                (
                    self.num_sim_traj,
                    self.horizon,
                ),
                device=torch_helpers.device,
                dtype=torch.float32,
                requires_grad=False,
            )

    def get_entity_positions(self, obs):
        flat_object_dyn = obs.narrow(
            -1, self.env.agent_dim, self.env.object_dyn_dim * self.env.nObj
        )
        # -> Reshape so that .... x nObj x object_dim
        batched_object_dyn = flat_object_dyn.view(
            *obs.shape[:-1], self.env.nObj, self.env.object_dyn_dim
        )
        # only return the first compression_ndim dimensions, e.g. for 2: x,y
        if self._ensemble_size:
            return batched_object_dyn[..., : self.compression_ndim].reshape(
                *obs.shape[:-1], -1
            )
        else:
            return batched_object_dyn[..., : self.compression_ndim]

    def _model_compression_costs(self, rollout_buffer: RolloutBuffer):
        ensemble_dim = 1

        next_obs = rollout_buffer.as_array(
            "next_observations"
        )  # shape: [p,e,h,obs_dim]
        next_obs = self.get_entity_positions(
            next_obs
        )  # shape: [p,e,h,compression_ndim*nObj]

        if self._ensemble_size:
            ensemble_dim = 1
            # We mean the predictions over the ensemble dimension!
            next_obs_ensemble_mean = torch.mean(next_obs, dim=ensemble_dim)
            # next_obs_ensemble_mean shape: [p,h,compression_ndim*nObj]
            # back to object centric view!
            next_obs_ensemble_mean = next_obs_ensemble_mean.view(
                *next_obs_ensemble_mean.shape[:-1], self.env.nObj, -1
            )
        else:
            next_obs_ensemble_mean = next_obs

        next_obs_ensemble_mean = torch.round(
            next_obs_ensemble_mean * self.precision
        ).to(torch.int32)

        next_obs_ensemble_mean = torch_helpers.to_numpy(next_obs_ensemble_mean)
        num_particles = next_obs_ensemble_mean.shape[0]
        num_dims = next_obs_ensemble_mean.shape[-1]

        for p in range(num_particles):
            for h in range(self.horizon):
                buff = 0
                for d in range(num_dims):
                    if self.compression_algo == "bzip2":
                        compressed_obs = bz2.compress(
                            next_obs_ensemble_mean[p, h, :, d].tobytes(),
                            compresslevel=self.compresslevel,
                        )
                    else:
                        compressed_obs = lz4.frame.compress(
                            next_obs_ensemble_mean[p, h, :, d].tobytes(),
                            compression_level=self.compresslevel,
                        )
                    buff += len(compressed_obs)
                self._compression_cost_per_path[p, h] = buff

    def _model_epistemic_costs(self, rollout_buffer: RolloutBuffer):
        ensemble_dim = 1

        mean_next_obs = rollout_buffer.as_array(
            "next_observations"
        )  # shape: [p,e,h,obs_dim]
        torch.std(mean_next_obs, dim=ensemble_dim, out=self.stds_of_means_)

        self._epistemic_bonus_per_path = self.stds_of_means_.sum(dim=-1)  # [p,h]

    @torch.no_grad()
    def trajectory_cost_fn(
        self, cost_fn, rollout_buffer: RolloutBuffer, out: torch.Tensor
    ):
        if self.use_env_reward:
            raise NotImplementedError()
            # costs_path shape: [p,h] or [p,ensemble_models,h]
        num_particles = rollout_buffer.as_array("next_observations").shape[0]
        self._model_compression_costs(rollout_buffer)

        if self._ensemble_size:
            costs_path = (
                self._compression_cost_per_path.unsqueeze(1)
                .expand(-1, self._ensemble_size, -1)[:num_particles, ...]
                .clone()
            )
        else:
            costs_path = self._compression_cost_per_path[:num_particles, ...].clone()

        if self._w_ensemble_disagreement:
            self._model_epistemic_costs(rollout_buffer)
            costs_path += (
                self._maybe_ensemble_disagreement_scale
                * -self._epistemic_bonus_per_path.unsqueeze(1).expand(
                    -1, self._ensemble_size, -1
                )
            )

        if self._w_extrinsic_reward:
            env_cost = cost_fn(
                rollout_buffer.as_array("observations"),
                rollout_buffer.as_array("actions"),
                rollout_buffer.as_array("next_observations"),
            )  # shape: [p,h]
            costs_path += self._maybe_extrinsic_reward_scale * env_cost

        # Watch out: result is written to preallocated variable 'out'
        if self.cost_along_trajectory == "sum":
            return torch.sum(costs_path, axis=-1, out=out)
        elif self.cost_along_trajectory == "best":
            return torch.amin(costs_path[..., 1:], axis=-1, out=out)
        elif self.cost_along_trajectory == "final":
            raise NotImplementedError()
        else:
            raise NotImplementedError(
                "Implement method {} to compute cost along trajectory".format(
                    self.cost_along_trajectory
                )
            )

    def reset_horizon(self, horizon):
        if horizon == self.horizon:
            return
        self.horizon = horizon
        self._check_validity_parameters()

        # Re-allocate memory for controller:
        self.preallocate_memory()

        # Re-allocate and change horizon for model:
        if self._ensemble_size:
            self.forward_model.horizon = horizon
            if hasattr(self.forward_model, "preallocate_memory"):
                self.forward_model.preallocate_memory()
