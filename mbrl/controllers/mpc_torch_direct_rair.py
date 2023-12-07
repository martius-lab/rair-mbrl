import numpy as np
import torch
from numba import jit

from mbrl import torch_helpers
from mbrl.controllers.mpc_torch import TorchMpcICem
from mbrl.rolloutbuffer import RolloutBuffer


@jit(nopython=True)  # Set "nopython" mode for best performance, equivalent to @njit
def entropy(xs):
    xs = xs / np.sum(xs)
    return -np.sum(xs * np.log(xs))


#  iCEM with direct RaIR (similar to compression with bzip2)
class TorchDirectRairMpcICem(TorchMpcICem):
    def __init__(
        self,
        *,
        compression=True,
        all_axis=False,
        per_axis=True,
        compression_ndim=2,
        granularity=1,
        precision=100,
        ensemble_disagreement=False,
        ensemble_disagreement_scale=1.0,
        extrinsic_reward=False,
        extrinsic_reward_scale=1.0,
        **kwargs,
    ):
        self.compression = compression
        self.all_axis = all_axis
        self.per_axis = per_axis

        self.compression_ndim = compression_ndim
        self.granularity = granularity
        self.precision = precision
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

        self._direct_rair_cost_per_path = torch.zeros(
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
        # For now only return x-y
        if self._ensemble_size:
            return batched_object_dyn[..., : self.compression_ndim].reshape(
                *obs.shape[:-1], -1
            )
        else:
            return batched_object_dyn[..., : self.compression_ndim]

    def compress(self, values):
        compressed = {}
        for v in np.round(values):
            if v in compressed:
                compressed[v] += 1
            else:
                compressed[v] = 1
        return entropy(np.fromiter(compressed.values(), dtype=np.int16))

    def _model_direct_rair_costs(self, rollout_buffer: RolloutBuffer):

        next_obs = rollout_buffer.as_array(
            "next_observations"
        )  # shape: [p,e,h,obs_dim]

        next_obs = self.get_entity_positions(next_obs)
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

        next_obs_ensemble_mean = self.granularity * torch.floor(
            next_obs_ensemble_mean * self.precision / self.granularity
        )

        next_obs_ensemble_mean = torch_helpers.to_numpy(next_obs_ensemble_mean)
        num_particles = next_obs_ensemble_mean.shape[0]

        for p in range(num_particles):
            for h in range(self.horizon):
                buff = 0
                if self.per_axis:
                    for d in range(self.compression_ndim):
                        values2compress = list(next_obs_ensemble_mean[p, h, :, d])
                        buff += self.compress(values2compress)
                elif self.all_axis:
                    values2compress = list(next_obs_ensemble_mean[p, h, :, :].flatten())
                    buff += self.compress(values2compress)

                self._direct_rair_cost_per_path[p, h] = torch_helpers.to_tensor(
                    buff
                ).to(torch_helpers.device)

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

        if self.compression:
            self._model_direct_rair_costs(rollout_buffer)
        else:
            torch.fill_(self._direct_rair_cost_per_path, 0.0)

        if self._ensemble_size:
            costs_path = (
                self._direct_rair_cost_per_path.unsqueeze(1)
                .expand(-1, self._ensemble_size, -1)[:num_particles, ...]
                .clone()
            )
        else:
            costs_path = self._direct_rair_cost_per_path[:num_particles, ...].clone()

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
