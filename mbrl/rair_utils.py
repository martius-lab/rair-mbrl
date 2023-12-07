import bz2

import numpy as np
import torch


def entropy(xs):
    if isinstance(xs, torch.Tensor):
        xs = xs / torch.sum(xs)
        return -torch.sum(xs * torch.log(xs))
    else:
        xs = xs / np.sum(xs)
        return -np.sum(xs * np.log(xs))


def compress(values, return_dict=False):
    compressed_keys = []
    compressed_values = []

    for v in values:
        if v in compressed_keys:
            compressed_values[compressed_keys.index(v)] += 1
        else:
            compressed_keys.append(v)
            compressed_values.append(1)

    if return_dict:
        return entropy(np.asarray(compressed_values)), compressed_keys, compressed_values
    else:
        return entropy(np.asarray(compressed_values))


def get_relation_matrix(
    rep2compress,
    compression_ndim,
    precision=100,
    granularity=2.5,
    bidirectional=False,
    non_dist=True,
    distance="euclidean",
    rounding_fn=np.floor,
):
    if non_dist:
        # Absolute difference and difference vectorized cases
        relation_matrix = rep2compress[:, None, :] - rep2compress[None, :, :]
        relation_matrix = relation_matrix.reshape(-1, compression_ndim)

        if not bidirectional:
            relation_matrix = np.abs(relation_matrix)

        relation_matrix = granularity * rounding_fn(relation_matrix * precision / granularity)
    else:
        relation_matrix = rep2compress[:, None, :] - rep2compress[None, :, :]
        relation_matrix = relation_matrix.reshape(-1, compression_ndim)
        relation_matrix = granularity * rounding_fn(relation_matrix * precision / granularity)

        if distance == "euclidean":
            relation_matrix = np.sqrt(np.sum(relation_matrix**2, axis=-1, keepdims=True))
        elif distance == "manhattan":
            relation_matrix = np.sum(np.abs(relation_matrix), axis=-1, keepdims=True)
        elif distance == "inf":
            relation_matrix = np.amax(np.abs(relation_matrix), axis=-1, keepdims=True)

    return relation_matrix


def get_mask(nObj, bidirectional=False):
    mask = np.ones(nObj**2, dtype=bool)
    if bidirectional:
        mask[range(0, nObj**2, nObj + 1)] = False
    else:
        dummy = np.arange(nObj**2).reshape(nObj, nObj)
        ind = np.tril_indices(nObj)
        mask[dummy[ind]] = False
    return mask


def get_obs_ready(obs, nObj, agent_dim, object_dyn_dim, compression_ndim):
    if obs.ndim == 1:
        obs = obs[None]

    if isinstance(obs, torch.Tensor):
        flat_object_dyn = obs.narrow(-1, agent_dim, object_dyn_dim * nObj)
        # -> Reshape so that .... x nObj x object_dim
        batched_object_dyn = flat_object_dyn.view(*obs.shape[:-1], nObj, object_dyn_dim)
        # For now only return x-y
        return batched_object_dyn[..., :compression_ndim]
    else:
        flat_object_dyn = obs[..., agent_dim : agent_dim + nObj * object_dyn_dim]
        # -> Reshape so that .... x nObj x object_dim
        batched_object_dyn = flat_object_dyn.reshape(*obs.shape[:-1], nObj, object_dyn_dim)
        # return only the part of the observation to be compressed!
        return batched_object_dyn[..., :compression_ndim]


def model_relational_rair(
    obs,
    env,
    compression_ndim=2,
    bidirectional=True,
    non_dist=True,
    distance="euclidean",
    granularity=2.5,
    precision=100,
    rounding="floor",
):
    rounding_fn = np.floor if rounding == "floor" else np.round

    mask = get_mask(env.nObj, bidirectional)

    obs_to_compress = get_obs_ready(obs, env.nObj, env.agent_dim, env.object_dyn_dim, compression_ndim)
    # obs_to_compress shape [num_particles, num_entities, compression_ndim]

    num_particles = obs_to_compress.shape[0]
    compression_cost_per_sample = np.zeros((num_particles,))

    for p in range(num_particles):
        relation_matrix = get_relation_matrix(
            obs_to_compress[p, :, :],
            compression_ndim,
            precision,
            granularity,
            bidirectional,
            non_dist,
            distance,
            rounding_fn,
        )

        compression_cost_per_sample[p] = compress(relation_matrix[mask].tolist())

    return compression_cost_per_sample


def get_frequency_table(
    obs,
    env,
    compression_ndim=2,
    bidirectional=True,
    non_dist=True,
    distance="euclidean",
    granularity=2.5,
    precision=100,
    rounding="floor",
):
    rounding_fn = np.floor if rounding == "floor" else np.round

    mask = get_mask(env.nObj, bidirectional)

    obs_to_compress = get_obs_ready(obs, env.nObj, env.agent_dim, env.object_dyn_dim, compression_ndim)
    # obs_to_compress shape [num_particles, num_entities, compression_ndim]

    num_particles = obs_to_compress.shape[0]
    assert num_particles == 1

    relation_matrix = get_relation_matrix(
        obs_to_compress[0, :, :],
        compression_ndim,
        precision,
        granularity,
        bidirectional,
        non_dist,
        distance,
        rounding_fn,
    )

    entropy_val, c_keys, c_vals = compress(relation_matrix[mask].tolist(), return_dict=True)
    return entropy_val, c_keys, c_vals


# For compression with bzip2
def model_compression_costs_structured_obs(obs, env, compression_ndim=2, granularity=100):
    if obs.ndim == 1:
        obs = obs[None]

    flat_object_dyn = obs[..., env.agent_dim : env.agent_dim + env.nObj * env.object_dyn_dim]
    # -> Reshape so that .... x nObj x object_dim
    batched_object_dyn = flat_object_dyn.reshape(*obs.shape[:-1], env.nObj, env.object_dyn_dim)
    # For now only return x-y
    obs_to_compress = batched_object_dyn[..., :compression_ndim]

    obs_to_compress = np.round(obs_to_compress * granularity).astype(np.int32)

    num_particles = obs_to_compress.shape[0]
    num_dims = obs_to_compress.shape[-1]

    compression_cost_per_sample = np.zeros((num_particles,))
    for p in range(num_particles):
        buff = 0
        for d in range(num_dims):
            compressed_obs = bz2.compress(obs_to_compress[p, :, d].tobytes())
            buff += len(compressed_obs)
        compression_cost_per_sample[p] = buff
    return compression_cost_per_sample
