import logging

logger = logging.getLogger(__name__)

import json
import numpy as np
from scipy.linalg import lstsq

# Load recorded poses data from JSON file
# with open('env/kuka/data/T_shape_point_recordings/T_shape_record_1.json', 'r') as f:
#     recorded_poses = json.load(f)


with open('env/kuka/data/T_shape_point_recordings/U_shape_record_1.json', 'r') as f:
    recorded_poses = json.load(f)

# Define the local L1 positions (constant across all poses)
# for T-shape object
# positions_local1 = {
#     'Object_local1_p_w_to_o': [0, 0, 0],
#     'Object_local1_p_w_1': [-0.015, 0.13, 0],
#     'Object_local1_p_w_2': [0.015, 0.13, 0],
#     'Object_local1_p_w_3': [-0.015, 0.03, 0],
#     'Object_local1_p_w_4': [0.015, 0.03, 0],
#     'Object_local1_p_w_5': [0.055, 0.03, 0],
#     'Object_local1_p_w_6': [0.055, 0.0, 0],
#     'Object_local1_p_w_7': [-0.055, 0.03, 0],
#     'Object_local1_p_w_8': [-0.055, 0.0, 0],
# }

# for U-shape object
positions_local1 = {
    'Object_local1_p_w_to_o': [0, 0, 0],
    'Object_local1_p_w_1': [-0.039, 0.105, 0],
    'Object_local1_p_w_2': [-0.019, 0.105, 0],
    'Object_local1_p_w_3': [0.019, 0.105, 0],
    'Object_local1_p_w_4': [0.039, 0.105, 0],
    'Object_local1_p_w_5': [-0.019, 0.02, 0],
    'Object_local1_p_w_6': [0.019, 0.02, 0],
    'Object_local1_p_w_7': [-0.039, 0.0, 0],
    'Object_local1_p_w_8': [0.039, 0.0, 0],
}

def estimate_rotation_matrix(P_world, P_local):
    """Estimate rotation matrix from P_local to P_world using least squares."""
    P_world_centered = P_world - np.mean(P_world, axis=1, keepdims=True)
    P_local_centered = P_local - np.mean(P_local, axis=1, keepdims=True)
    H = P_world_centered @ P_local_centered.T
    U, _, Vt = np.linalg.svd(H)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R

def quaternion_to_matrix(q):
    """Convert a quaternion (x, y, z, w) to a 3x3 rotation matrix."""
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2)]
    ])

# Define data holders for concatenated results
R_w_to_L1_all = []
R_w_to_L2_all = []
P_w_to_L1_all = []
P_w_to_L2_all = []

# Process each pose in recorded_poses
for pose_data in recorded_poses.values():
    # Convert positions_local1 to a matrix for computation
    P_local1 = np.array([positions_local1[key] for key in positions_local1 if key != 'Object_local1_p_w_to_o']).T
    
    # Extract P_world for local1 points from pose data
    P_world1 = np.array([pose_data[key] for key in positions_local1 if key != 'Object_local1_p_w_to_o']).T
    
    # Estimate R_w_to_L1
    R_w_to_L1 = estimate_rotation_matrix(P_world1, P_local1)
    R_w_to_L1_all.append(R_w_to_L1)
    
    # Extract P_w_to_L1 and P_w_to_L2
    P_w_to_L1 = pose_data['Object_local1_p_w_to_o']
    P_w_to_L1_all.append(P_w_to_L1)
    
    P_w_to_L2 = pose_data['Object_local2_p_w_to_L2']
    P_w_to_L2_all.append(P_w_to_L2)
    
    # Convert R_w_to_L2 quaternion to rotation matrix
    R_w_to_L2_quaternion = pose_data['Object_local2_R_w_to_L2']  # Assume quaternion format (x, y, z, w)
    R_w_to_L2 = quaternion_to_matrix(R_w_to_L2_quaternion)
    R_w_to_L2_all.append(R_w_to_L2)

# Convert lists to numpy arrays for least-squares solution
R_w_to_L1_all = np.array(R_w_to_L1_all)
R_w_to_L2_all = np.array(R_w_to_L2_all)
P_w_to_L1_all = np.array(P_w_to_L1_all)
P_w_to_L2_all = np.array(P_w_to_L2_all)

# Solve for R_L2_to_L1
# Solve for R_L2_to_L1
# R_L2_to_L1 = np.mean([np.linalg.inv(R_w_to_L2) @ R_w_to_L1 for R_w_to_L1, R_w_to_L2 in zip(R_w_to_L1_all, R_w_to_L2_all)], axis=0)

# Initialize A and B matrices for least-squares formulation
A = []
B = []

# Populate A and B matrices for least squares
for R_w_to_L1, R_w_to_L2 in zip(R_w_to_L1_all, R_w_to_L2_all):
    # Kronecker product I ⊗ R_w_to_L2
    kron_product = np.kron(np.eye(3), R_w_to_L2)
    A.append(kron_product)
    B.append(R_w_to_L1.flatten())  # Flatten to match the shape for vec(R_w_to_L1)

# Stack A and B for least-squares solution
A = np.vstack(A)  # Shape: (3 * n_poses, 9)
B = np.hstack(B).reshape(-1, 1)  # Shape: (3 * n_poses, 1)

# Solve for vec(R_L2_to_L1) using least squares
R_L2_to_L1_vector, _, _, _ = lstsq(A, B)

# Reshape the vector result back to a 3x3 matrix
R_L2_to_L1 = R_L2_to_L1_vector.reshape(3, 3)

# Enforce the rotation matrix constraint using SVD
U, _, Vt = np.linalg.svd(R_L2_to_L1)
R_L2_to_L1 = U @ Vt
if np.linalg.det(R_L2_to_L1) < 0:
    R_L2_to_L1[:, -1] *= -1  # Ensure a positive determinant (det = 1)

logger.debug('%s %s', "Estimated R_L2_to_L1 (rotation matrix):", R_L2_to_L1)

# Correct A and B to have matching dimensions
A = []
B = []

for i, R_w_to_L2 in enumerate(R_w_to_L2_all):
    # Calculate expected P_w_to_L1 from the estimated R_L2_to_L1 and observed P_w_to_L2
    A.append(R_w_to_L2 @ R_L2_to_L1)  # 3x3 result per pose
    # B.append(P_w_to_L1_all[i] - P_w_to_L2_all[i])  # 3x1 result per pose
    diff = P_w_to_L1_all[i] - P_w_to_L2_all[i]
    B.append(diff.reshape(3, 1))  # Each is 3x1, per pose

# Stack A and B correctly
A = np.vstack(A)  # Shape (n_poses * 3, 3)
B = np.vstack(B)  # Shape (n_poses * 3, 1)

logger.debug('%s %s %s %s', "A: ", A.shape, " b: ", B.shape)
# Solve for P_L2_to_L1 using least squares
P_L2_to_L1, _, _, _ = lstsq(A, B)

logger.debug('%s %s', "Estimated R_L2_to_L1:", R_L2_to_L1)
logger.debug('%s %s', "Estimated P_L2_to_L1:", P_L2_to_L1)
