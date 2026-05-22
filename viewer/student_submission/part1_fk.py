from __future__ import annotations

import numpy as np


def forward_kinematics(
    joints: list[object],
    local_rotations: list[np.ndarray],
    root_offset: np.ndarray,
    topological_order: tuple[int, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Student part-1 implementation.

    Expected inputs:
    - joints: each joint has `.parent` and `.translation`
    - local_rotations: one 3x3 local rotation matrix per joint
    - root_offset: global translation applied to the root
    - topological_order: optional parent-before-child traversal order

    Expected outputs:
    - world_rotations: shape (J, 3, 3)
    - world_positions: shape (J, 3)
    """
    joint_count = len(joints)
    world_rotations = np.tile(
        np.eye(3, dtype=np.float32)[None, :, :],
        (joint_count, 1, 1),
    )
    world_positions = np.zeros((joint_count, 3), dtype=np.float32)

    # Default traversal order
    if topological_order is None:
        children = [[] for i in range(joint_count)]
        roots = []

        for i, joint in enumerate(joints):

            parent = joint.parent

            if parent == -1 or parent is None:
                roots.append(i)
            else:
                children[parent].append(i)

        order = []

        while roots:
            current = roots.pop()
            order.append(current)

            for child in children[current]:
                roots.append(child)

        topological_order = tuple(order)

    for joint_number in topological_order:

        joint = joints[joint_number]
        parent_number = joint.parent

        # Root joint
        if parent_number is None or parent_number == -1:

            world_rotations[joint_number] = local_rotations[joint_number]

            world_positions[joint_number] = root_offset + joint.translation

        # Child joint
        else:

            parent_rotation = world_rotations[parent_number]
            parent_position = world_positions[parent_number]

            # Propagate rotation
            world_rotations[joint_number] = parent_rotation @ local_rotations[joint_number]

            # Propagate position
            world_positions[joint_number] = parent_position + parent_rotation @ joint.translation

    return world_rotations, world_positions
