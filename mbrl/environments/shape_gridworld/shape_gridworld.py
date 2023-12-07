"""
Code taken and modified from: https://github.com/tkipf/c-swm
"""
"""Gym environment for pushing shape tasks (2D Shapes and 3D Cubes)."""

import gym
import matplotlib as mpl
import numpy as np
from gym import spaces

from mbrl.seeding import np_random_seeding

# mpl.use("TkAgg")
mpl.use("Agg")
import matplotlib.pyplot as plt
import skimage
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image


def square(r0, c0, width, im_size):
    rr, cc = [r0, r0 + width, r0 + width, r0], [c0, c0, c0 + width, c0 + width]
    return skimage.draw.polygon(rr, cc, im_size)


def triangle(r0, c0, width, im_size):
    rr, cc = [r0, r0 + width, r0 + width], [c0 + width // 2, c0, c0 + width]
    return skimage.draw.polygon(rr, cc, im_size)


def fig2rgb_array(fig):
    fig.canvas.draw()
    buffer = fig.canvas.tostring_rgb()
    width, height = fig.canvas.get_width_height()
    return np.fromstring(buffer, dtype=np.uint8).reshape(height, width, 3)


def render_cubes(positions, width):
    voxels = np.zeros((width, width, width), dtype=np.bool)
    colors = np.empty(voxels.shape, dtype=object)

    cols = ["purple", "green", "orange", "blue", "brown"]

    for i, pos in enumerate(positions):
        voxels[pos[0], pos[1], 0] = True
        colors[pos[0], pos[1], 0] = cols[i]

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.w_zaxis.set_pane_color((0.5, 0.5, 0.5, 1.0))
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.line.set_lw(0.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.voxels(voxels, facecolors=colors, edgecolor="k")

    im = fig2rgb_array(fig)
    plt.close(fig)
    im = np.array(
        Image.fromarray(im[215:455, 80:570]).resize((50, 50), Image.ANTIALIAS)
    )  # Crop and resize
    return im / 255.0


class ShapeGridworld(gym.Env):
    """Gym environment for shape pushing task."""

    def __init__(
        self,
        width=5,
        height=5,
        render_type="circles",
        render_delta=10,
        num_objects=5,
        object_persistency=10,
        render_w_grid=True,
        seed=None,
    ):
        self.width = width
        self.height = height
        self.render_type = render_type
        self.render_delta = render_delta
        self.render_w_grid = render_w_grid

        # No grid can be printed due to the slanted view with cubes!
        if self.render_type == "cubes":
            self.render_w_grid = False

        self.num_objects = num_objects
        self.num_actions = 2  # Move in x-y directions!
        self.object_persistency = object_persistency

        # self.colors = utils.get_colors(num_colors=max(9, self.num_objects))
        self.colors = [
            (0.21568627450980393, 0.49411764705882355, 0.7215686274509804, 1.0)
            for _ in range(max(9, self.num_objects))
        ]

        self.np_random = None
        self.game = None
        self.target = None

        # Initialize to pos outside of env for easier collision resolution.
        self.objects = [[-1, -1] for _ in range(self.num_objects)]

        # If True, then check for collisions and don't allow two
        #   objects to occupy the same position.
        self.collisions = True

        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.num_actions,), dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_objects * 2 + 2,),
            dtype=np.float32,
        )

        self.seed(seed)
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = np_random_seeding(seed)
        return [seed]

    def render(self, mode="rgb_array"):
        im = self._render_callback()
        im = np.transpose(im, (1, 2, 0))

        if self.render_w_grid:
            dx, dy = self.render_delta, self.render_delta
            # Custom (rgb) grid color
            grid_color = [1, 1, 1]
            # Modify the image to include the grid
            im[:, ::dy, :] = grid_color
            im[::dx, :, :] = grid_color

        if mode == "rgb_array":
            return im
        else:
            raise NotImplementedError

    def _render_callback(self):
        if self.render_type == "grid":
            im = np.zeros((3, self.width, self.height))
            for idx, pos in enumerate(self.objects):
                im[:, pos[0], pos[1]] = self.colors[idx][:3]
            return im
        elif self.render_type == "circles":
            im = np.zeros(
                (self.width * self.render_delta, self.height * self.render_delta, 3),
                dtype=np.float32,
            )
            for idx, pos in enumerate(self.objects):
                rr, cc = skimage.draw.circle(
                    pos[0] * self.render_delta + self.render_delta / 2,
                    pos[1] * self.render_delta + self.render_delta / 2,
                    self.render_delta / 2,
                    im.shape,
                )
                im[rr, cc, :] = self.colors[idx][:3]
            return im.transpose([2, 0, 1])
        elif self.render_type == "shapes":
            im = np.zeros(
                (self.width * self.render_delta, self.height * self.render_delta, 3),
                dtype=np.float32,
            )
            for idx, pos in enumerate(self.objects):
                if idx % 3 == 0:
                    rr, cc = skimage.draw.circle(
                        pos[0] * self.render_delta + self.render_delta / 2,
                        pos[1] * self.render_delta + self.render_delta / 2,
                        self.render_delta / 2,
                        im.shape,
                    )
                    im[rr, cc, :] = self.colors[idx][:3]
                elif idx % 3 == 1:
                    rr, cc = triangle(
                        pos[0] * self.render_delta,
                        pos[1] * self.render_delta,
                        self.render_delta,
                        im.shape,
                    )
                    im[rr, cc, :] = self.colors[idx][:3]
                else:
                    rr, cc = square(
                        pos[0] * self.render_delta,
                        pos[1] * self.render_delta,
                        self.render_delta,
                        im.shape,
                    )
                    im[rr, cc, :] = self.colors[idx][:3]
            return im.transpose([2, 0, 1])
        elif self.render_type == "cubes":
            im = render_cubes(self.objects, self.width)
            return im.transpose([2, 0, 1])

    def reset(self):

        self.objects = [[-1, -1] for _ in range(self.num_objects)]

        # Randomize object position.
        for i in range(len(self.objects)):

            # Resample to ensure objects don't fall on same spot.
            while not self.valid_pos(self.objects[i], i):
                self.objects[i] = [
                    self.np_random.choice(np.arange(self.width)),
                    self.np_random.choice(np.arange(self.height)),
                ]

        state_obs = np.asarray(self.objects, dtype=np.float32).flatten()

        self.current_object = 0
        self.current_object_t = 0
        state_obs = np.hstack((state_obs, self.current_object, self.current_object_t))

        return state_obs

    def valid_pos(self, pos, obj_id):
        """Check if position is valid."""
        if pos[0] < 0 or pos[0] >= self.width:
            return False
        if pos[1] < 0 or pos[1] >= self.height:
            return False

        if self.collisions:
            for idx, obj_pos in enumerate(self.objects):
                if idx == obj_id:
                    continue

                if pos[0] == obj_pos[0] and pos[1] == obj_pos[1]:
                    return False

        return True

    def valid_move(self, obj_id, offset):
        """Check if move is valid."""
        old_pos = self.objects[obj_id]
        new_pos = [p + o for p, o in zip(old_pos, offset)]
        return self.valid_pos(new_pos, obj_id)

    def translate(self, obj_id, offset):
        """ "Translate object pixel.

        Args:
            obj_id: ID of object.
            offset: (x, y) tuple of offsets.
        """

        if self.valid_move(obj_id, offset):
            self.objects[obj_id][0] += offset[0]
            self.objects[obj_id][1] += offset[1]

    def discretize_action(self, action_x):
        if action_x < -1 / 3:
            return -1
        elif action_x >= -1 / 3 and action_x < 1 / 3:
            return 0
        else:
            return 1

    def step(self, action):

        done = False
        reward = 0

        # directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

        direction = [self.discretize_action(a) for a in action]
        obj = self.current_object
        # print(f"Stepping for object {obj} in direction {direction}")

        self.translate(obj, direction)

        state_obs = np.asarray(self.objects, dtype=np.float32).flatten()

        self.current_object_t += 1

        if self.current_object_t >= self.object_persistency:
            self.current_object = (self.current_object + 1) % self.num_objects
            self.current_object_t = 0

        state_obs = np.hstack((state_obs, self.current_object, self.current_object_t))

        return state_obs, reward, done, None
