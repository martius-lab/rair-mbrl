# import jax
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


def get_values2compress_np(
    next_obs_ensemble_mean,
    compression_ndim,
    horizon,
    precision,
    granularity,
    mask,
    rounding_fn,
    distance="euclidean",
):  # Function is compiled to machine code when called the first time
    num_particles = next_obs_ensemble_mean.shape[0]
    horizon = next_obs_ensemble_mean.shape[1]

    current_vecs = next_obs_ensemble_mean[..., :compression_ndim]

    diff_matrix = np.expand_dims(current_vecs, axis=3) - np.expand_dims(
        current_vecs, axis=2
    )
    diff_matrix = diff_matrix.reshape(num_particles, horizon, -1, compression_ndim)

    diff_matrix = granularity * rounding_fn(diff_matrix * precision / granularity)

    if distance == "euclidean":
        relation_matrix = np.sqrt(np.sum(diff_matrix**2, axis=-1, keepdims=True))
    elif distance == "manhattan":
        relation_matrix = np.sum(np.abs(diff_matrix), axis=-1, keepdims=True)
    elif distance == "inf":
        relation_matrix = np.amax(np.abs(diff_matrix), axis=-1, keepdims=True)

    return relation_matrix[:, :, mask, :]


def get_values2compress_torch(
    next_obs_ensemble_mean,
    compression_ndim,
    horizon,
    precision,
    granularity,
    mask,
    rounding_fn,
    distance="euclidean",
):  # Function is compiled to machine code when called the first time
    num_particles = next_obs_ensemble_mean.shape[0]
    horizon = next_obs_ensemble_mean.shape[1]

    current_vecs = next_obs_ensemble_mean[..., :compression_ndim]

    diff_matrix = current_vecs[:, :, :, None, :] - current_vecs[:, :, None, :, :]
    diff_matrix = diff_matrix.reshape(num_particles, horizon, -1, compression_ndim)

    diff_matrix = granularity * rounding_fn(diff_matrix * precision / granularity)

    if distance == "euclidean":
        relation_matrix = torch.sqrt(torch.sum(diff_matrix**2, dim=-1, keepdim=True))
    elif distance == "manhattan":
        relation_matrix = torch.sum(torch.abs(diff_matrix), dim=-1, keepdim=True)
    elif distance == "inf":
        relation_matrix = torch.amax(torch.abs(diff_matrix), dim=-1, keepdim=True)
    # relation_matrix shape: [num_particles, horizon, nObj, nObj, 1]

    # Add an extra dimension in the end and flatten the relation_matrix (nObj x nObj -> nObj**2)
    relation_matrix = relation_matrix.view(num_particles, horizon, -1, 1)

    return relation_matrix[:, :, mask, :]


#  iCEM with realtional RaIR with difference distance funcs
class TorchRelationalRairMpcICem(TorchMpcICem):
    def __init__(
        self,
        *,
        compression=True,
        compression_ndim=2,
        distance="euclidean",
        granularity=1,
        precision=100,
        mode="normal",
        rounding="floor",
        decimals=4,
        ensemble_disagreement=False,
        ensemble_disagreement_scale=1.0,
        extrinsic_reward=False,
        extrinsic_reward_scale=1.0,
        **kwargs,
    ):
        self.compression = compression
        self.compression_ndim = compression_ndim
        self.distance = distance
        self.granularity = granularity
        self.precision = precision
        self.mode = mode
        self.decimals = decimals
        self._w_ensemble_disagreement = ensemble_disagreement
        self._maybe_ensemble_disagreement_scale = ensemble_disagreement_scale
        self._w_extrinsic_reward = extrinsic_reward
        self._maybe_extrinsic_reward_scale = extrinsic_reward_scale
        super().__init__(**kwargs)

        if not self._ensemble_size:
            self._w_ensemble_disagreement = False

        if rounding == "floor":
            self.rounding_fn_np = np.floor
            self.rounding_fn_torch = torch.floor
        else:
            self.rounding_fn_np = np.round
            self.rounding_fn_torch = torch.round

    @torch.no_grad()
    def preallocate_memory(self):
        """
        Preallocate memory for distribution parameters in addition
        """
        super().preallocate_memory()

        self._rair_cost_per_path = torch.zeros(
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

        self.mask = np.ones(self.env.nObj**2, dtype=bool)

        dummy = np.arange(self.env.nObj**2).reshape(self.env.nObj, self.env.nObj)
        ind = np.tril_indices(self.env.nObj)
        self.mask[dummy[ind]] = False

        self.mask_tensor = torch.from_numpy(self.mask).to(torch_helpers.device)

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
        return batched_object_dyn[..., : self.compression_ndim].reshape(
            *obs.shape[:-1], -1
        )

    def compress(self, values):
        compressed_keys = []
        compressed_values = []

        for v in values:
            if v in compressed_keys:
                compressed_values[compressed_keys.index(v)] += 1
            else:
                compressed_keys.append(v)
                compressed_values.append(1)
        return entropy(np.asarray(compressed_values))

    def fast_compress(self, values2compress, horizon, num_particles):
        v = values2compress
        n_obs = values2compress.shape[2]
        # this is a bit of a hack, we project the feature dimension to 1 by reverse base-100 encoding
        # 100 should be greater than the range of values
        # this might not speed up much
        new_vals = v.reshape(num_particles, horizon, n_obs, -1)
        new_vals = (
            new_vals
            * (100 ** torch.arange(new_vals.shape[-1], device=torch_helpers.device))
        ).sum(-1)
        new_vals = new_vals.reshape(-1, new_vals.shape[-1], 1)
        # if known a priori, can be just created withous scanning the array :D
        all_values = torch.unique(new_vals).reshape(1, 1, -1)
        occ = (torch.abs(all_values - new_vals) == 0).sum(1)
        occ = occ / n_obs
        # could be faster
        occ[occ == 0] = 1  # could mess things up numerically
        occ = -torch.sum(occ * torch.log(occ), axis=-1)
        return occ.reshape(num_particles, horizon)

    def _model_rair_costs(self, rollout_buffer: RolloutBuffer):

        next_obs = rollout_buffer.as_array(
            "next_observations"
        )  # shape: [p,e,h,obs_dim]

        next_obs = self.get_entity_positions(next_obs)
        if self._ensemble_size:
            ensemble_dim = 1
            # We mean the predictions over the ensemble dimension!
            next_obs_ensemble_mean = torch.mean(next_obs, dim=ensemble_dim)
            # next_obs_ensemble_mean shape: [p,h,dof*nObj]
        else:
            next_obs_ensemble_mean = next_obs

        # back to object centric view!
        next_obs_ensemble_mean = next_obs_ensemble_mean.view(
            *next_obs_ensemble_mean.shape[:-1], self.env.nObj, -1
        )
        num_particles = next_obs_ensemble_mean.shape[0]

        if torch_helpers.device == "cpu" or self.mode == "normal":
            next_obs_ensemble_mean = torch_helpers.to_numpy(next_obs_ensemble_mean)

            values2compress = get_values2compress_np(
                next_obs_ensemble_mean,
                self.compression_ndim,
                self.horizon,
                self.precision,
                self.granularity,
                self.mask,
                self.rounding_fn_np,
                self.distance,
            )

            values2compress = np.round(values2compress * 10**self.decimals).astype(
                int
            )

            self._rair_cost_per_path[:num_particles, : self.horizon] = torch.tensor(
                [
                    [
                        self.compress(values2compress[p, h, ...].tolist())
                        for h in range(self.horizon)
                    ]
                    for p in range(num_particles)
                ]
            ).to(torch_helpers.device)
        else:
            values2compress_torch = get_values2compress_torch(
                next_obs_ensemble_mean,
                self.compression_ndim,
                self.horizon,
                self.precision,
                self.granularity,
                self.mask_tensor,
                self.rounding_fn_torch,
                self.distance,
            )

            values2compress_torch = torch.round(
                values2compress_torch * 10**self.decimals
            ).to(torch.int32)

            self._rair_cost_per_path[
                :num_particles, : self.horizon
            ] = self.fast_compress(values2compress_torch, self.horizon, num_particles)

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
            self._model_rair_costs(rollout_buffer)
        else:
            torch.fill_(self._rair_cost_per_path, 0.0)

        if self._ensemble_size:
            costs_path = (
                self._rair_cost_per_path.unsqueeze(1)
                .expand(-1, self._ensemble_size, -1)[:num_particles, ...]
                .clone()
            )
        else:
            costs_path = self._rair_cost_per_path[:num_particles, ...].clone()

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
