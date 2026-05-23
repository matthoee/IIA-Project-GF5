from __future__ import annotations

import numpy as np


def make_one_hot_skinning_weights(weights: np.ndarray) -> np.ndarray:
    """Student part-2 task: convert a dense weight matrix into one-hot weights."""
    #Initialize 2D matrix with all 0 entries
    one_hot = np.zeros_like(weights)
    
    #Returns 1D Array of dominant joints for each vertex 
    dominant_joints = np.argmax(weights, axis=1)
    # 1D Array [1,2,...,N], used to index joints in next step
    vertex_indices = np.arange(weights.shape[0])

    # Set the dominant joint's weight to 1.0
    one_hot[vertex_indices, dominant_joints] = 1.0


def skin_smpl_mesh(
    model_data: object,
    world_rotations: np.ndarray,
    world_positions: np.ndarray,
    *,
    use_blended_weights: bool,
) -> np.ndarray:
    """Student part-2 task: pose the SMPL mesh with one-hot or blended weights."""
    rest_vertices = np.asarray(model_data.rest_vertices, dtype=np.float32)
    rest_joints = np.asarray(model_data.rest_joints, dtype=np.float32)
    
    if use_blended_weights:
        weights = np.asarray(model_data.skinning_weights, dtype=np.float32)
    else:
        weights = np.asarray(model_data.one_hot_skinning_weights, dtype=np.float32)

    # Initialize empty array same shape as vertices
    skinned_vertices = np.zeros_like(rest_vertices)
    #number of joints
    num_joints = rest_joints.shape[0]
    
    # Vectorized loop: Process ALL vertices for ONE joint at a time
    for j in range(num_joints):
        #weight column for joint j
        w_j = weights[:, j:j+1]  # Shape: (V, 1)
        
        #coordinates wrt joint j
        local_pos = rest_vertices - rest_joints[j]  # Shape: (V, 3)
        
        #Rotate and translate into world space
        transformed_pos = (local_pos @ world_rotations[j].T) + world_positions[j]  # Shape: (V, 3)
        
        #Multiply by weights
        skinned_vertices += w_j * transformed_pos

    return skinned_vertices
