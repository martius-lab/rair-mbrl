import numpy as np
import torch
from dm_control.utils import rewards
from gym import spaces
from scipy.spatial.transform import Rotation

from mbrl import torch_helpers
from mbrl.environments.dm2gym import DmControlWrapper


def _flatten_obs(obs):
    obs_pieces = []
    for v in obs.values():
        flat = np.array([v]) if np.isscalar(v) else v.ravel()
        obs_pieces.append(flat)
    return np.concatenate(obs_pieces, axis=0)


# QUADRUPED!
_TOES = ["toe_front_left", "toe_back_left", "toe_back_right", "toe_front_right"]
_KNEES = ["knee_front_left", "knee_back_left", "knee_back_right", "knee_front_right"]
_ANKLES = [
    "ankle_front_left",
    "ankle_back_left",
    "ankle_back_right",
    "ankle_front_right",
]
_HIPS = ["hip_front_left", "hip_back_left", "hip_back_right", "hip_front_right"]


# noinspection PyProtectedMember
class QuadrupedSuite(DmControlWrapper):
    domain_name = "quadruped"
    supports_live_rendering = False

    def __init__(
        self,
        *,
        name,
        task_name,
        task_kwargs=None,
        visualize_reward=True,
        render_mode="human",
        overwrite_obs=True,
        include_toes=True,
        include_knees=True,
        include_ankles=True,
        include_hips=False,
        include_torso=False,
        goal_mode="walk",  # roboyoga or walk
        roboyoga_goal_id=None,
        seed=None,
        **kwargs,
    ):
        if seed is not None:
            if task_kwargs is None:
                task_kwargs = {}
            task_kwargs = {"random": seed}

        super().__init__(
            name=name,
            task_name=task_name,
            task_kwargs=task_kwargs,
            visualize_reward=visualize_reward,
            render_mode=render_mode,
            **kwargs,
        )
        self.store_init_arguments(locals())
        self.overwrite_obs = overwrite_obs
        self.include_toes = include_toes
        self.include_knees = include_knees
        self.include_ankles = include_ankles
        self.include_hips = include_hips
        self.include_torso = include_torso

        self.goal_mode = goal_mode
        self.roboyoga_goal_id = roboyoga_goal_id
        self.goals = get_dmc_benchmark_goals("quadruped")

        self.qpos_len = 23
        self.qvel_len = 22
        self.agent_dim = self.qpos_len + self.qvel_len + 3 + 1
        # 3 for torso velocity and 1 for torso upright!

        self.nObj = (
            4
            * (
                self.include_toes
                + self.include_knees
                + self.include_ankles
                + self.include_hips
            )
            + self.include_torso
        )
        self.object_dyn_dim = 3
        self.object_stat_dim = 0
        # Updating the observation space
        orig_dim = self.observation_space.shape[0]  # 78
        obs_dim = self.agent_dim + self.object_dyn_dim * self.nObj
        if not self.overwrite_obs:
            obs_dim += orig_dim
            self.agent_dim += orig_dim
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(obs_dim,), dtype="float32"
        )
        self.observation_space_size_preproc = self.obs_preproc(
            np.zeros(self.observation_space.shape[0])
        ).shape[0]

    def _upright_reward_from_obs(self, torso_upright, deviation_angle=0):
        """Returns a reward proportional to how upright the torso is.

        Args:
            deviation_angle: A float, in degrees. The reward is 0 when the torso is
            exactly upside-down and 1 when the torso's z-axis is less than
            `deviation_angle` away from the global z-axis.
        """
        deviation = np.cos(np.deg2rad(deviation_angle))
        return rewards.tolerance(
            torso_upright,
            bounds=(deviation, float("inf")),
            sigmoid="linear",
            margin=1 + deviation,
            value_at_margin=0,
        )

    def _move_reward_from_obs(self, torso_x_vel, desired_speed=0.5):
        return rewards.tolerance(
            torso_x_vel,
            bounds=(desired_speed, float("inf")),
            margin=desired_speed,
            value_at_margin=0.5,
            sigmoid="linear",
        )

    def _compute_reward_from_obs(self, obs, goal_idx):
        # task_type = self.task_type
        # ex = [0, 1]
        # distance = self.goals[goal_idx] - self._env.physics.data.qpos
        # distance = np.linalg.norm(distance) - np.linalg.norm(distance[ex])
        # reward = -distance
        pose = obs[: self.qpos_len]

        def get_su(state, goal):
            dist = np.abs(state - goal)
            dist[..., [1, 2, 3]] = shortest_angle(dist[..., [1, 2, 3]])
            if goal_idx in [0, 1, 2, 5, 6, 7, 8, 11]:
                dist = dist[..., [0, 1, 2, 3, 4, 8, 12, 16]]
            if goal_idx in [12, 13]:
                dist = dist[..., [0, 1, 2, 3]]
            return dist.max(-1)

        def rotate(s, times=1):
            # Invariance goes as follows: add 1.57 to azimuth, circle legs 0,1,2,3 -> 1,2,3,0
            s = s.copy()
            for i in range(times):
                s[..., 1] = s[..., 1] + 1.57
                s[..., -16:] = np.roll(s[..., -16:], 12)
            return s

        def normalize(s):
            return np.concatenate(
                (s[..., 2:3], quat2euler(s[..., 3:7]), s[..., 7:]), -1
            )

        state = normalize(pose)
        goal = normalize(self.goals[goal_idx])
        distance = min(
            get_su(state, goal),
            get_su(rotate(state, 1), goal),
            get_su(rotate(state, 2), goal),
            get_su(rotate(state, 3), goal),
        )
        return -distance, (distance < 0.7).astype(np.float32)

    def compute_reward_from_obs(self, obs, goal_idx=None):
        if goal_idx is None:
            goal_idx = self.roboyoga_goal_id

        reward, success = self._compute_reward_from_obs(obs, goal_idx)

        # info = {f"metric_success/goal_{goal_idx}": success, f"metric_reward/goal_{goal_idx}": reward}
        return reward

    def cost_fn(self, states, actions, next_states):
        # return np.zeros_like(states[..., 0])
        torch_flag = False
        if torch.is_tensor(states):
            states = torch_helpers.to_numpy(states)
            torch_flag = True
        if self.goal_mode == "walk":
            desired_speed = 0.5
            upright_reward = self._upright_reward_from_obs(
                states[..., self.agent_dim - 1]
            )
            costs = (
                (-1)
                * upright_reward
                * self._move_reward_from_obs(
                    states[..., self.agent_dim - 4], desired_speed
                )
            )
        elif self.goal_mode == "roboyoga":
            # # [p,e,h,obs_dim]
            # ------------ Multidim version ----------------
            pose = states[..., : self.qpos_len]

            def get_su(state, goal):
                dist = np.abs(state - goal)
                dist[..., [1, 2, 3]] = shortest_angle(dist[..., [1, 2, 3]])
                if self.roboyoga_goal_id in [0, 1, 2, 5, 6, 7, 8, 11]:
                    dist = dist[..., [0, 1, 2, 3, 4, 8, 12, 16]]
                if self.roboyoga_goal_id in [12, 13]:
                    dist = dist[..., [0, 1, 2, 3]]
                return dist.max(-1)

            def rotate(s, times=1):
                # Invariance goes as follows: add 1.57 to azimuth, circle legs 0,1,2,3 -> 1,2,3,0
                s = s.copy()
                for i in range(times):
                    s[..., 1] = s[..., 1] + 1.57
                    s[..., -16:] = np.roll(s[..., -16:], 12, axis=-1)
                return s

            def normalize(s):
                return np.concatenate(
                    (s[..., 2:3], quat2euler(s[..., 3:7]), s[..., 7:]), -1
                )

            robo_state = normalize(pose.reshape(-1, self.qpos_len))
            robo_state = robo_state.reshape(*states.shape[:-1], -1)
            goal = normalize(self.goals[self.roboyoga_goal_id])
            get_su_stacked = np.stack(
                (
                    get_su(robo_state, goal),
                    get_su(rotate(robo_state, 1), goal),
                    get_su(rotate(robo_state, 2), goal),
                    get_su(rotate(robo_state, 3), goal),
                ),
                axis=-1,
            )
            costs = np.min(get_su_stacked, axis=-1)
            # success = (distance < 0.7).astype(np.float32)
        else:
            raise NotImplementedError

        if torch_flag:
            return torch_helpers.to_tensor(costs).to(torch_helpers.device)
        else:
            return costs

    def targ_proc(self, observations, next_observations):
        return next_observations - observations

    def obs_preproc(self, observation):
        return observation

    def obs_postproc(self, obs, pred=None, out=None):
        if pred is not None:
            return obs + pred
        else:
            return obs

    # def get_reward(self, physics):
    #     """Returns a reward to the agent."""

    #     # Move reward term.
    #     move_reward = rewards.tolerance(
    #         physics.torso_velocity()[0],
    #         bounds=(self._desired_speed, float('inf')),
    #         margin=self._desired_speed,
    #         value_at_margin=0.5,
    #         sigmoid='linear')

    #     return _upright_reward(physics) * move_reward

    def set_state_from_observation(self, observation):
        # physics_state = observation[:self.agent_dim]
        # self.dmcenv._physics.set_state(physics_state)
        # self.dmcenv._physics.after_reset()
        raise NotImplementedError

    def _update_obs(self, obs):
        # QPOS SHAPE: 23
        # QPOS SHAPE: 22
        obs["state"] = self.dmcenv.physics.data.qpos.copy()
        obs["state_vel"] = self.dmcenv.physics.data.qvel.copy()
        obs["torso_vel"] = self.dmcenv.physics.torso_velocity()
        obs["torso_up"] = self.dmcenv.physics.torso_upright()

        # print("qpos: ", obs["state"])
        # print("qvel: ", obs["state_vel"])

        if self.include_toes:
            obs["toes"] = np.hstack(self.dmcenv.physics.named.data.xpos[_TOES])
        if self.include_knees:
            obs["knees"] = np.hstack(self.dmcenv.physics.named.data.xpos[_KNEES])
        if self.include_ankles:
            obs["ankles"] = np.hstack(self.dmcenv.physics.named.data.xpos[_ANKLES])
        if self.include_hips:
            obs["hips"] = np.hstack(self.dmcenv.physics.named.data.xpos[_HIPS])
        if self.include_torso:
            obs["torso"] = np.hstack(self.dmcenv.physics.named.data.xpos["torso"])

        # Goal could also be added here for future!
        return obs

    def _get_observation(self):
        if self.overwrite_obs:
            obs = dict()
        else:
            obs = dict(self.timestep.observation)
        obs = self._update_obs(obs)
        # print("-----------")
        # for k, v in obs.items():
        #     print("{}: {}, shape: {}".format(k,v,v.shape))
        # print("-----------")
        return _flatten_obs(obs)

        """
        Output of
            self.dmcenv._physics.named.data.xpos
            for quadruped!
                                x         y         z
        0             world [ 0         0         0       ]
        1             torso [-0.00741  -0.00238   0.566   ]
        2    hip_front_left [-0.168     0.131     0.756   ]
        3   knee_front_left [-0.407     0.287     0.9     ]
        4  ankle_front_left [-0.387     0.285     1.25    ]
        5    toe_front_left [-0.214     0.182     1.48    ]
        6   hip_front_right [-0.172    -0.232     0.587   ]
        7  knee_front_right [-0.422    -0.413     0.505   ]
        8 ankle_front_right [-0.427    -0.704     0.706   ]
        9   toe_front_right [-0.252    -0.808     0.927   ]
        10    hip_back_right [ 0.154    -0.136     0.375   ]
        11   knee_back_right [ 0.259    -0.198     0.0798  ]
        12  ankle_back_right [ 0.562    -0.379     0.0917  ]
        13    toe_back_right [ 0.735    -0.482     0.314   ]
        14     hip_back_left [ 0.157     0.227     0.544   ]
        15    knee_back_left [ 0.297     0.501     0.459   ]
        16   ankle_back_left [ 0.591     0.602     0.628   ]
        17     toe_back_left [ 0.766     0.499     0.848   ]
        """


# WALKER!
_FEET = ["left_foot", "right_foot"]
_THIGHS = ["left_thigh", "right_thigh"]
_LEGS = ["left_leg", "right_leg"]


# noinspection PyProtectedMember
class WalkerSuite(DmControlWrapper):
    domain_name = "walker"
    supports_live_rendering = False

    def __init__(
        self,
        *,
        name,
        task_name,
        task_kwargs=None,
        visualize_reward=True,
        render_mode="human",
        overwrite_obs=True,
        include_feet=True,
        include_thighs=False,
        include_legs=True,
        include_torso=True,
        keep_torso_up=False,
        keep_feet_up=False,
        **kwargs,
    ):
        super().__init__(
            name=name,
            task_name=task_name,
            task_kwargs=task_kwargs,
            visualize_reward=visualize_reward,
            render_mode=render_mode,
            **kwargs,
        )
        self.store_init_arguments(locals())
        self.overwrite_obs = overwrite_obs
        self.include_feet = include_feet
        self.include_thighs = include_thighs
        self.include_legs = include_legs
        self.include_torso = include_torso

        self.keep_torso_up = keep_torso_up
        self.keep_feet_up = keep_feet_up

        assert not (self.keep_torso_up and self.keep_feet_up)

        self.qpos_len = 9
        self.qvel_len = 9
        self.agent_dim = self.qpos_len + self.qvel_len
        self.nObj = (
            2 * (self.include_feet + self.include_thighs + self.include_legs)
            + self.include_torso
        )
        self.object_dyn_dim = 3

        if self.keep_feet_up:
            self.entity_id_for_cost_fn = np.array([0, 1]) if self.include_feet else []
        elif self.keep_torso_up:
            self.entity_id_for_cost_fn = self.nObj - 1 if self.include_torso else []
        else:
            self.entity_id_for_cost_fn = np.array([])
        self.entity_id_for_cost_fn_tensor = (
            torch_helpers.to_tensor(self.entity_id_for_cost_fn)
            .to(torch_helpers.device)
            .to(torch.int32)
        )

        # Updating the observation space
        orig_dim = self.observation_space.shape[0]  # 24
        obs_dim = self.agent_dim + self.object_dyn_dim * self.nObj
        if not self.overwrite_obs:
            obs_dim += orig_dim
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(obs_dim,), dtype="float32"
        )

    def cost_fn(self, obs, actions, next_obs):
        if isinstance(obs, torch.Tensor):
            if self.keep_feet_up or self.keep_torso_up:
                flat_object_dyn = obs.narrow(
                    -1, self.agent_dim, self.object_dyn_dim * self.nObj
                )
                # -> Reshape so that .... x nObj x object_dim
                batched_object_dyn = flat_object_dyn.view(
                    *obs.shape[:-1], self.nObj, self.object_dyn_dim
                )
                feet_or_torso_obs = torch.index_select(
                    batched_object_dyn, -2, self.entity_id_for_cost_fn_tensor
                )
                # even for single dimension of indices, index_select keeps the dimension!
                # return the sum of z dimensions for all
                return (-1) * torch.sum(feet_or_torso_obs[..., -1], dim=-1)
            else:
                return torch.zeros_like(obs[..., 0])
        else:
            return np.zeros_like(obs[..., 0])

    def set_state_from_observation(self, observation):
        # physics_state = observation[:self.agent_dim]
        # self.dmcenv._physics.set_state(physics_state)
        # self.dmcenv._physics.after_reset()
        raise NotImplementedError

    def _update_obs(self, obs):
        # QPOS SHAPE: 9
        # QVEL SHAPE: 9
        obs["state"] = self.dmcenv.physics.data.qpos
        obs["state_vel"] = self.dmcenv.physics.data.qvel

        # print("qpos: ", self.dmcenv.physics.data.qpos.shape)
        # print("qvel: ", self.dmcenv.physics.data.qvel.shape)

        if self.include_feet:
            obs["feet"] = np.hstack(self.dmcenv.physics.named.data.xpos[_FEET])
        if self.include_thighs:
            obs["thighs"] = np.hstack(self.dmcenv.physics.named.data.xpos[_THIGHS])
        if self.include_legs:
            obs["legs"] = np.hstack(self.dmcenv.physics.named.data.xpos[_LEGS])
        if self.include_torso:
            obs["torso"] = np.hstack(self.dmcenv.physics.named.data.xpos["torso"])

        # Goal could also be added here for future!
        return obs

    def _get_observation(self):
        # original observation contains the following
        # obs['orientations'] = physics.orientations()
        # obs['height'] = physics.torso_height()
        # obs['velocity'] = physics.velocity()
        # original dimension of observation: 24

        if self.overwrite_obs:
            obs = dict()
        else:
            obs = dict(self.timestep.observation)
        obs = self._update_obs(obs)
        # print("All geom: ", self.dmcenv._physics.named.data.xpos)
        return _flatten_obs(obs)

        """
        Output of
            self.dmcenv._physics.named.data.xpos
            for WALKER!
                        x         y         z
        0       world [ 0         0         0       ]
        1       torso [-0.00526   0         1.22    ]
        2 right_thigh [ 0.189    -0.05      1.45    ]
        3   right_leg [ 0.534    -0.05      1.78    ]
        4  right_foot [ 0.76     -0.05      1.73    ]
        5  left_thigh [ 0.189     0.05      1.45    ]
        6    left_leg [-0.148     0.05      1.91    ]
        7   left_foot [-0.143     0.05      2.12    ]
        """


def shortest_angle(angle):
    if not angle.shape:
        return shortest_angle(angle[None])[0]
    angle = angle % (2 * np.pi)
    angle[angle > np.pi] = 2 * np.pi - angle[angle > np.pi]
    return angle


def quat2euler(quat):
    rot = Rotation.from_quat(quat)
    return rot.as_euler("XYZ")


def get_dmc_benchmark_goals(task_type):
    if task_type == "walker":
        # pose[0] is height
        # pose[1] is x
        # pose[2] is global rotation
        # pose[3:6] - first leg hip, knee, ankle
        # pose[6:9] - second leg hip, knee, ankle
        # Note: seems like walker can't bend legs backwards

        lie_back = [-1.2, 0.0, -1.57, 0, 0.0, 0.0, 0, -0.0, 0.0]
        lie_front = [-1.2, -0, 1.57, 0, 0, 0, 0, 0.0, 0.0]
        legs_up = [-1.24, 0.0, -1.57, 1.57, 0.0, 0.0, 1.57, -0.0, 0.0]

        kneel = [-0.5, 0.0, 0, 0, -1.57, -0.8, 1.57, -1.57, 0.0]
        side_angle = [-0.3, 0.0, 0.9, 0, 0, -0.7, 1.87, -1.07, 0.0]
        stand_up = [-0.15, 0.0, 0.34, 0.74, -1.34, -0.0, 1.1, -0.66, -0.1]

        lean_back = [-0.27, 0.0, -0.45, 0.22, -1.5, 0.86, 0.6, -0.8, -0.4]
        boat = [-1.04, 0.0, -0.8, 1.6, 0.0, 0.0, 1.6, -0.0, 0.0]
        bridge = [-1.1, 0.0, -2.2, -0.3, -1.5, 0.0, -0.3, -0.8, -0.4]

        head_stand = [-1, 0.0, -3, 0.6, -1, -0.3, 0.9, -0.5, 0.3]
        one_feet = [-0.2, 0.0, 0, 0.7, -1.34, 0.5, 1.5, -0.6, 0.1]
        arabesque = [-0.34, 0.0, 1.57, 1.57, 0, 0.0, 0, -0.0, 0.0]
        # Other ideas: flamingo (hard), warrior (med), upside down boat (med), three legged dog

        goals = np.stack(
            [
                lie_back,
                lie_front,
                legs_up,
                kneel,
                side_angle,
                stand_up,
                lean_back,
                boat,
                bridge,
                one_feet,
                head_stand,
                arabesque,
            ]
        )

    if task_type == "quadruped":
        # pose[0,1] is x,y
        # pose[2] is height
        # pose[3:7] are vertical rotations in the form of a quaternion (i think?)
        # pose[7:11] are yaw pitch knee ankle for the front left leg
        # pose[11:15] same for the front right leg
        # pose[15:19] same for the back right leg
        # pose[19:23] same for the back left leg

        lie_legs_together = get_quadruped_pose(
            [0, 3.14, 0], 0.2, dict(out_up=[0, 1, 2, 3]), [-0.7, 0.7, -0.7, 0.7]
        )
        lie_rotated = get_quadruped_pose([0.8, 3.14, 0], 0.2, dict(out_up=[0, 1, 2, 3]))
        lie_two_legs_up = get_quadruped_pose(
            [0.8, 3.14, 0], 0.2, dict(out_up=[1, 3], down=[0, 2])
        )

        lie_side = get_quadruped_pose(
            [0.0, 0, -1.57], 0.3, dict(out=[0, 1, 2, 3]), [-0.7, 0.7, -0.7, 0.7]
        )
        lie_side_back = get_quadruped_pose(
            [0.0, 0, 1.57], 0.3, dict(out=[0, 1, 2, 3]), [-0.7, 0.7, -0.7, 0.7]
        )
        stand = get_quadruped_pose([1.57, 0, 0], 0.2, dict(up=[0, 1, 2, 3]))
        stand_rotated = get_quadruped_pose([0.8, 0, 0], 0.2, dict(up=[0, 1, 2, 3]))

        stand_leg_up = get_quadruped_pose(
            [1.57, 0, 0.0], 0.7, dict(down=[0, 2, 3], out_up=[1])
        )
        attack = get_quadruped_pose([1.57, 0.0, -0.4], 0.7, dict(out=[0, 1, 2, 3]))
        balance_front = get_quadruped_pose(
            [1.57, 0.0, 1.57], 0.7, dict(up=[0, 1, 2, 3])
        )
        balance_back = get_quadruped_pose(
            [1.57, 0.0, -1.57], 0.7, dict(up=[0, 1, 2, 3])
        )
        balance_diag = get_quadruped_pose(
            [1.57, 0, 0.0], 0.7, dict(down=[0, 2], out_up=[1, 3])
        )

        goals = np.stack(
            [
                lie_legs_together,  # 0
                lie_rotated,  # 1
                lie_two_legs_up,  # 2
                lie_side,  # 3
                lie_side_back,  # 4
                stand,  # 5
                stand_rotated,  # 6
                stand_leg_up,  # 7
                attack,  # 8
                balance_front,  # 9
                balance_back,  # 10
                balance_diag,  # 11
            ]
        )

    return goals


def get_quadruped_pose(global_rot, global_pos=0.5, legs={}, legs_rot=[0, 0, 0, 0]):
    """

    :param angles: along height, along depth, along left-right
    :param height:
    :param legs:
    :return:
    """
    if not isinstance(global_pos, list):
        global_pos = [0, 0, global_pos]
    pose = np.zeros([23])
    pose[0:3] = global_pos
    pose[3:7] = Rotation.from_euler("XYZ", global_rot).as_quat()

    pose[[7, 11, 15, 19]] = legs_rot
    for k, v in legs.items():
        for leg in v:
            if k == "out":
                pose[[8 + leg * 4]] = 0.5  # pitch
                pose[[9 + leg * 4]] = -1.0  # knee
                pose[[10 + leg * 4]] = 0.5  # ankle
            if k == "inward":
                pose[[8 + leg * 4]] = -0.35  # pitch
                pose[[9 + leg * 4]] = 0.9  # knee
                pose[[10 + leg * 4]] = -0.5  # ankle
            elif k == "down":
                pose[[8 + leg * 4]] = 1.0  # pitch
                pose[[9 + leg * 4]] = -0.75  # knee
                pose[[10 + leg * 4]] = -0.3  # ankle
            elif k == "out_up":
                pose[[8 + leg * 4]] = -0.2  # pitch
                pose[[9 + leg * 4]] = -0.8  # knee
                pose[[10 + leg * 4]] = 1.0  # ankle
            elif k == "up":
                pose[[8 + leg * 4]] = -0.35  # pitch
                pose[[9 + leg * 4]] = -0.2  # knee
                pose[[10 + leg * 4]] = 0.6  # ankle

    return pose
