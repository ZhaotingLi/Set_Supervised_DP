import numpy as np
import quaternion # pip install numpy-quaternion
from geometry_msgs.msg import PoseStamped, Pose


from scipy.spatial.transform import Rotation as R


def euler_to_axis_angle(euler, seq: str = "xyz"):
    """
    Convert Euler angles → axis-angle (rotation vector).

    Parameters
    ----------
    euler : array-like, shape (..., 3)
        Euler angles in radians, e.g. [roll, pitch, yaw] or [x, y, z]
        depending on `seq`.
    seq : str, optional
        Euler angle convention (default: 'xyz').
        Must match whatever your current Franka code uses.

    Returns
    -------
    rotvec : np.ndarray, shape (..., 3)
        Axis-angle rotation vector. Direction = rotation axis (unit vector),
        norm = rotation angle in radians.
    """
    euler = np.asarray(euler, dtype=np.float32)
    rot = R.from_euler(seq, euler, degrees=False)
    rotvec = rot.as_rotvec()  # axis * angle
    return rotvec.astype(np.float32)


def axis_angle_to_euler(axis_angle, seq: str = "xyz"):
    """
    Convert axis-angle (rotation vector) → Euler angles.

    Parameters
    ----------
    axis_angle : array-like, shape (..., 3)
        Axis-angle rotation vector, as used in robosuite.
        Direction = rotation axis (unit), norm = angle (rad).
    seq : str, optional
        Euler angle convention (default: 'xyz').
        Must be consistent with `euler_to_axis_angle`.

    Returns
    -------
    euler : np.ndarray, shape (..., 3)
        Euler angles in radians, using the given sequence.
    """
    axis_angle = np.asarray(axis_angle, dtype=np.float32)
    rot = R.from_rotvec(axis_angle)
    euler = rot.as_euler(seq, degrees=False)
    return euler.astype(np.float32)


def get_quaternion_from_euler(roll, pitch, yaw):
  """
  Convert an Euler angle to a quaternion.
   
  Input
    :param roll: The roll (rotation around x-axis) angle in radians.
    :param pitch: The pitch (rotation around y-axis) angle in radians.
    :param yaw: The yaw (rotation around z-axis) angle in radians.
 
  Output
    :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
  """
  qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
  qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
  qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
  qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
 
  return [ qw ,qx, qy, qz]


def get_euler_from_quaternion(Q):
    """
    Convert a quaternion to Euler angles (roll, pitch, yaw).
    Supports input as list [w, x, y, z] or quaternion object.
    """
    # Check if input is a quaternion object and extract components
    if hasattr(Q, 'w'): 
        w, x, y, z = Q.w, Q.x, Q.y, Q.z
    else:
        # Assume it's a list/array [w, x, y, z]
        w, x, y, z = Q

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    # Clamp the value to the range [-1, 1] to handle numerical errors
    if np.abs(sinp) >= 1:
        # Use 90 degrees if out of range (Gimbal lock)
        pitch = np.sign(sinp) * (np.pi / 2) 
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

def orientation_2_quaternion(orientation):
    return np.quaternion(orientation.w, orientation.x, orientation.y, orientation.z)

def position_2_array(position):
    return np.array([position.x, position.y, position.z])

def pose_2_transformation(pose: Pose):
    quaternion_orientation = orientation_2_quaternion(pose.orientation)
    translation = position_2_array(pose.position)
    rotation_matrix = quaternion.as_rotation_matrix(quaternion_orientation)
    transformation_matrix = np.identity(4)
    transformation_matrix[0:3, 0:3] = rotation_matrix
    transformation_matrix[0:3, 3] = translation
    return transformation_matrix

def array_quat_2_pose(pos_array, quat):
    pose_st = PoseStamped()
    pose_st.pose.position.x = pos_array[0]
    pose_st.pose.position.y = pos_array[1]
    pose_st.pose.position.z = pos_array[2]
    pose_st.pose.orientation.x = quat.x
    pose_st.pose.orientation.y = quat.y
    pose_st.pose.orientation.z = quat.z
    pose_st.pose.orientation.w = quat.w
    return pose_st

def transformation_2_pose(transformation_matrix):
    pos_array = transformation_matrix[0:3, 3]
    rotation_matrix = transformation_matrix[0:3, 0:3]
    quat = quaternion.from_rotation_matrix(rotation_matrix)
    pose_st = array_quat_2_pose(pos_array, quat)
    return pose_st

def pose_st_2_transformation(pose_st: PoseStamped):
    transformation_matrix = pose_2_transformation(pose_st.pose)
    return transformation_matrix

def transform_pose(pose: PoseStamped, transformation_matrix):
    pose_as_matrix = pose_st_2_transformation(pose)
    transformed_pose_matrix = transformation_matrix @ pose_as_matrix
    transformed_pose = transformation_2_pose(transformed_pose_matrix)
    return transformed_pose

def transform_pos_ori(pos: np.array, ori, transform):
    ori_quat = list_2_quaternion(ori)
    ori_rot_matrix = quaternion.as_rotation_matrix(ori_quat)
    transformed_ori_rot_matrix = transform[:3,:3] @ ori_rot_matrix
    pos = np.hstack((pos, 1))
    transformed_pos = transform @ pos
    transformed_ori_quat = quaternion.from_rotation_matrix(transformed_ori_rot_matrix)
    transformed_ori_array = np.array([transformed_ori_quat.w, transformed_ori_quat.x, transformed_ori_quat.y, transformed_ori_quat.z])
    return transformed_pos[:3], transformed_ori_array

def list_2_quaternion(quaternion_list: list):
    return np.quaternion(quaternion_list[0], quaternion_list[1], quaternion_list[2], quaternion_list[3])