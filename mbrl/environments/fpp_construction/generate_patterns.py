import math

import numpy as np


def generate_patterns(base_location, num_blocks, pattern_type, height_offset):

    assert pattern_type in ["line", "spaced_line", "polygon", "singletower", "multitower", "pyramid"]
    # Generate pattern in the given base location, out of the manipulability range of the robot arm
    block_locations = []
    if "line" in pattern_type:
        block_locations.append(base_location)
        axis = 0 if np.random.uniform() < 0.5 else 1  # Line in x direction or y direction
        spacing = np.random.uniform(0.0, 0.12, size=1) if pattern_type == "spaced_line" else 0
        for i in range(1, num_blocks):
            new_block_loc = block_locations[-1].copy()
            new_block_loc[axis] += 0.05 + spacing
            block_locations.append(new_block_loc)
    elif pattern_type == "polygon":
        radius = np.random.uniform(0.09, 0.16, size=1)
        for i in range(num_blocks):
            new_x = base_location[0] + radius * math.sin(i * 2 * math.pi / num_blocks)
            new_y = base_location[1] + radius * math.cos(i * 2 * math.pi / num_blocks)
            block_locations.append(np.array([new_x, new_y, base_location[-1]]))
    elif pattern_type == "singletower":
        # block_locations.append(base_location)
        # block_locations[0][2] = height_offset

        target_range = 0.1
        height_offset = base_location[-1]

        goal_object0 = base_location + np.random.uniform(-target_range, target_range, size=3)
        goal_object0[2] = height_offset

        # Start off goals array with the first block
        block_locations.append(goal_object0)

        # These below don't have goal object0 because only object0+ can be used for towers in PNP stage. In stack stage,
        previous_xys = [goal_object0[:2]]
        current_tower_heights = [goal_object0[2]]

        num_configured_blocks = num_blocks - 1

        for i in range(num_configured_blocks):
            # If stack only, use the object0 position as a base
            goal_objecti = goal_object0[:2]
            objecti_xy = goal_objecti

            # Check if any of current block xy matches any previous xy's
            for _ in range(len(previous_xys)):
                previous_xy = previous_xys[_]
                if np.linalg.norm(previous_xy - objecti_xy) < 0.071:
                    goal_objecti = previous_xy

                    new_height_offset = current_tower_heights[_] + 0.05
                    current_tower_heights[_] = new_height_offset
                    goal_objecti = np.append(goal_objecti, new_height_offset)

            # If we didn't find a previous height at the xy.. just put the block at table height and update the previous xys array
            if len(goal_objecti) == 2:
                goal_objecti = np.append(goal_objecti, height_offset)
                previous_xys.append(objecti_xy)
                current_tower_heights.append(height_offset)

            block_locations.append(goal_objecti)

    elif pattern_type == "multitower":
        if num_blocks < 3:
            num_towers = 1
        elif 3 <= num_blocks <= 5:
            num_towers = 2
        else:
            num_towers = 3
        tower_bases = []
        tower_heights = []
        target_range = 0.1
        height_offset = base_location[-1]
        for i in range(num_towers):
            base_xy = base_location[:2] + np.random.uniform(-target_range, target_range, size=2)
            while not np.all([np.linalg.norm(base_xy - other_xpos) >= 0.06 for other_xpos in tower_bases]):
                base_xy = base_location[:2] + np.random.uniform(-target_range, target_range, size=2)

            tower_bases.append(base_xy)
            tower_heights.append(height_offset)

            goal_objecti = np.zeros(3)
            goal_objecti[:2] = base_xy
            goal_objecti[2] = height_offset
            block_locations.append(goal_objecti.copy())
        # for _ in range(tower_height):
        for _ in range(num_blocks - num_towers):
            goal_objecti = np.zeros(3)
            goal_objecti[:2] = tower_bases[_ % num_towers][:2]
            goal_objecti[2] = tower_heights[_ % num_towers] + 0.05
            tower_heights[_ % num_towers] = tower_heights[_ % num_towers] + 0.05
            block_locations.append(goal_objecti.copy())

    elif pattern_type == "pyramid":
        height_offset = base_location[-1]

        def skew(x):
            return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])

        def rot_matrix(A, B):
            """
            Rotate A onto B
            :param A:
            :param B:
            :return:
            """
            A = A / np.linalg.norm(A)
            B = B / np.linalg.norm(B)
            v = np.cross(A, B)
            s = np.linalg.norm(v)
            c = np.dot(A, B)

            R = np.identity(3) + skew(v) + np.dot(skew(v), skew(v)) * ((1 - c) / s**2)
            return R

        def get_xs_zs(start_point):
            start_point[2] = height_offset
            xs = [0, 1, 0.5]
            zs = [0, 0, 1]
            diagonal_start = 2
            x_bonus = 0
            z = 0
            while len(xs) <= num_blocks:
                next_x = diagonal_start + x_bonus
                xs.append(next_x)
                zs.append(z)
                x_bonus -= 0.5
                z += 1
                if x_bonus < -0.5 * diagonal_start:
                    diagonal_start += 1
                    x_bonus = 0
                    z = 0
            return xs, zs

        x_scaling = 0.06
        target_range = 0.1
        start_point = base_location + np.random.uniform(-target_range, target_range, size=3)
        xs, zs = get_xs_zs(start_point)  # Just temporary
        # xs is actually ys, because we rotate

        attempt_count = 0
        while start_point[1] + max(xs) * x_scaling > base_location[1] + target_range:
            start_point = base_location[:3] + np.random.uniform(-target_range, target_range, size=3)
            if attempt_count > 10:
                start_point[1] = base_location[1] - target_range

            xs, zs = get_xs_zs(start_point)  # Just temporary
            attempt_count += 1

        for i in range(num_blocks):
            new_goal = start_point.copy()
            new_goal[0] += xs[i] * x_scaling
            new_goal[2] += zs[i] * 0.05

            if i > 0:
                target_dir_vec = np.zeros(3)
                target_dir_vec[:2] = base_location[:2] - block_locations[0][:2]

                target_dir_vec = np.array([0, 1, 0])

                new_goal_vec = np.zeros(3)
                new_goal_vec[:2] = new_goal[:2] - block_locations[0][:2]

                new_goal = (
                    rot_matrix(new_goal_vec, target_dir_vec) @ (new_goal - block_locations[0]) + block_locations[0]
                )

            block_locations.append(new_goal)

    return block_locations


def resolve_conflicts(obs, blocks_in_pattern, total_num_blocks, obj_idx, initial_gripper_xpos, obj_range):

    obs_in_pattern = obs[obj_idx[(total_num_blocks - blocks_in_pattern) * 3 :]].reshape(-1, 3)[:, :2].tolist()

    obs_random = obs[obj_idx[: (total_num_blocks - blocks_in_pattern) * 3]].reshape(-1, 3).copy()

    for obj_i in range(total_num_blocks - blocks_in_pattern):
        object_xyzpos = obs_random[obj_i, :3]

        while not (
            (np.linalg.norm(object_xyzpos[:2] - initial_gripper_xpos[:2]) >= 0.1)
            and np.all([np.linalg.norm(object_xyzpos[:2] - other_xpos) >= 0.08 for other_xpos in obs_in_pattern])
        ):
            object_xyzpos[:2] = initial_gripper_xpos[:2] + np.random.uniform(-obj_range, obj_range, size=2)

        obs_in_pattern.append(object_xyzpos[:2])
        obs[obj_idx[obj_i * 3 : obj_i * 3 + 3]] = object_xyzpos

    return obs
