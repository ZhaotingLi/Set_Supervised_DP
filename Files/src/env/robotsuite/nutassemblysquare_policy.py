import logging

logger = logging.getLogger(__name__)

import numpy as np
import robosuite.utils.transform_utils as T
from env.robotsuite.env_robosuite import axisangle_to_rot6d

from scipy.spatial.transform import Rotation as R

def quaternion_multiply(quat1, quat2):
    """Return multiplication of two quaternions."""
    x1, y1, z1, w1 = quat1
    x2, y2, z2, w2 = quat2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return np.array([x, y, z, w])

def quaternion_conjugate(quat):
    """Return the conjugate of a quaternion."""
    x, y, z, w = quat
    return np.array([-x, -y, -z, w])

def quaternion_to_angular_velocity(quat, dt=1.0):
    """Convert quaternion (representing a rotation) to angular velocity."""
    norm = np.linalg.norm(quat[0:3])
    if norm > 0:
        theta = 2 * np.arctan2(norm, quat[3])
        return quat[0:3] / norm * theta / dt
    else:
        return np.array([0.0, 0.0, 0.0])
    
def quaternion_to_axis_angle(quat):
    """Convert a quaternion to axis-angle representation."""
    x, y, z, w = quat
    w = max(min(w, 1), -1)
    angle = 2 * np.arccos(w)
    s = np.sqrt(1 - w**2)
    if s < 0.01:  # To avoid division by zero
        axis = np.array([x, y, z])  # If s is close to zero, direction of rotation is not important
    else:
        axis = np.array([x, y, z]) / s  # Normalize axis
    # print("w: ", w, " s: ", s, " x: ",  x, " axis: ", axis, " angle: ", angle)
    return axis, angle

 # Helper function to calculate angular velocity to a target orientation
def calculate_angular_velocity_to_target(current_ori, target_ori, dt):
    relative_rotation = quaternion_multiply(target_ori, quaternion_conjugate(current_ori))
    axis, angle = quaternion_to_axis_angle(relative_rotation)
    
    # Normalize angle to the range [0, π]
    angle = np.mod(angle + np.pi, 2 * np.pi) - np.pi
    
    angular_velocity =  axis * angle / dt

    # angular_velocity = 0.5 * angular_velocity / np.linalg.norm(angular_velocity) if np.linalg.norm(angular_velocity) > 0.5 else angular_velocity
    angular_velocity = 0.5 * angular_velocity / np.linalg.norm(angular_velocity) if np.linalg.norm(angular_velocity) > 0.5 else angular_velocity

    return 0.5* angular_velocity, angle
    # return angular_velocity, angle
    
def calculate_orientation_velocity(current_ori, target_ori_1, target_ori_2, dt=1.0):
    """Calculate orientation velocity from current orientation to the closest target orientation."""
   
    # Calculate angular velocity and angle for both target orientations
    angular_velocity_1, angle_1 = calculate_angular_velocity_to_target(current_ori, target_ori_1, dt)
    angular_velocity_2, angle_2 = calculate_angular_velocity_to_target(current_ori, target_ori_2, dt)

    # Choose the target orientation with the smallest absolute angle
    if abs(angle_1) < abs(angle_2):
        return angular_velocity_1
    else:
        return angular_velocity_2
    
# def calculate_orientation_velocity(current_ori, target_ori,  dt=1.0):
#     """Calculate orientation velocity from current orientation to target orientation."""
#     # First, find the relative orientation needed
#     relative_rotation = quaternion_multiply(target_ori, quaternion_conjugate(current_ori))
#     print('relative_rotation: ', relative_rotation)
#     # Convert this quaternion into axis-angle representation
#     axis, angle = quaternion_to_axis_angle(relative_rotation)
    
#     # The angular velocity (in radians per second) can be derived from the angle
#     # divided by the desired time interval dt
#     angular_velocity = 0.1 * axis * angle / dt
    
#     return angular_velocity

class NutAssemblyPolicy:
    def __init__(self, use_abs_action = False):
        self.use_abs_action = use_abs_action
        self.current_state = 0
        self.count = 0
        self.state_machine = [
            "MOVE_TOWARDS_NUT",
            "ADJUST_FOR_GRASP",
            "ADJUST_FOR_GRASP_lower",
            "GRASP_NUT",
            "LIFT_NUT",
            "MOVE_TO_PEG",
            "ALIGN_OVER_PEG",
            "ASSEMBLE_NUT",
            "RELEASE_NUT_RETREAT"
        ]

        # used to record and caculate the action statistics that are used to do normalization
        self.record_action_statistics = False

        # self.grasp_orientation_transform = np.array([0.7071068, 0.7071068, 0, 0 ]) 
        self.grasp_orientation_transform = np.array([1, 0, 0, 0 ]) 
        # self.grasp_orientation_transform = np.array([0.8315295, 0.550739, 0.0721858, -0.0058706 ]) 

        
    def reset(self):
        self.count = 0
        self.current_state = 0

    def calculate_position_delta(self, current_pos, target_pos):
        """Calculate the delta to move from current position to target position."""
        return target_pos - current_pos

    def calculate_orientation_delta(self, current_ori, target_ori):
        """Calculate the delta to adjust orientation from current to target."""
        # This is simplified; real implementation should involve quaternion math
        return target_ori - current_ori

    def move_end_effector(self, delta_pos, delta_ori):
        """Move the end effector by the specified deltas."""
        # This method would use the robot's control API to move the end effector
        pass

    def set_gripper(self, state):
        """Open or close the gripper. State: True (close) or False (open)."""
        # This method would use the robot's control API to control the gripper
        pass

    
    def get_action(self, obs, env = None):
        """Determine the action to take based on the current state and observation."""
        if self.current_state < len(self.state_machine):
            state = self.state_machine[self.current_state]
            action = None  # Placeholder for actual action logic
            logger.debug('%s %s', "state: ", state)
            if state == "MOVE_TOWARDS_NUT":
                # Example of using observation to decide action
                action = self.handle_move_towards_nut(obs, env)
                # Additional states would be handled similarly...
            elif state == 'ADJUST_FOR_GRASP':
                action = self.handle_adjust_for_grasp(obs, height= np.array([0, 0, -0.02]), state=state)
            elif state == 'ADJUST_FOR_GRASP_lower':
                action = self.handle_adjust_for_grasp(obs, height= np.array([0, 0, 0.01]), state=state)
            elif state == 'GRASP_NUT':
                action = self.handle_grasp_nut(obs)
            elif state == 'LIFT_NUT':
                action = self.handle_lift_nut(obs)
            elif state == 'MOVE_TO_PEG':
                action = self.handle_move_to_peg_uper01(obs)
            elif state == 'ALIGN_OVER_PEG':
                action = self.handle_move_to_peg_uper02(obs)
            elif state == 'ASSEMBLE_NUT':
                action = self.handle_move_to_peg_lower(obs)
            elif state == 'RELEASE_NUT_RETREAT':
                action = self.release_nut(obs)
            
            # Execute the action determined by the current state
            # This is a placeholder to illustrate concept
            # In practice, action execution would involve interacting with the robot's API
            self.execute_action(action)
            
            self.count = self.count + 1

            if self.use_abs_action:
                # convert the relative action to absolute
                current_pos = obs[0:3]
                # current_ori = T.mat2quat(obs[6:15].reshape(3, 3))  # [x,y,z,w]

                r = R.from_matrix(obs[6:15].reshape(3, 3))
                current_ori = r.as_quat()
     
                # print("------------------------------------")
                # print("current pos: ", current_pos, " current ori quat: ", current_ori)

                action_max = np.max(abs(action))
                if action_max > 1:
                    action =  action / action_max
                
                pos_vel, ori_vel, gripper_action = action[:3], action[3:-1], action[-1]
                # print("pos_vel: ", pos_vel, " ori_vel:", ori_vel)
                dt = 0.05

                # 1) integrate position
                next_pos = current_pos  + pos_vel * dt  # check the coordinate ... [To do]

                # 2) integrate orientation via quaternion delta
                omega = ori_vel
                theta = np.linalg.norm(omega)  *10 * dt
                if theta > 1e-10:
                    axis = omega / np.linalg.norm(omega)
                    half_theta = 0.5 * theta
                    delta_q = np.concatenate([axis * np.sin(half_theta), [np.cos(half_theta)]])
                    
                else:
                    delta_q = np.array([0.0, 0.0, 0.0, 1.0])

                # apply rotation and normalize
                next_q = quaternion_multiply(delta_q, current_ori)
                next_q /= np.linalg.norm(next_q)
                # print("desired next_q: ", next_q)
                # print("desired next pos: ", next_pos)
                # 3) convert quaternion → axis-angle vector (axis * angle)
                # next_ori_mat = T.quat2mat(next_q)
                r = R.from_quat(next_q)
                next_ori_mat = r.as_matrix()
                # next_ori_mat_world = np.linalg.inv(current_ori_mat).dot(next_ori_mat)
                # eef_xmat = env.env._eef_xmat
                # next_ori_mat_world = eef_xmat.dot(next_ori_mat_world)
                # axis_angle_vec = T.quat2axisangle(T.mat2quat(next_ori_mat_world))

                # Rot_A_to_B_raw = eef_xmat.dot(current_ori_mat.T)

                Rot_A_to_B = np.array([
                    [ 0.0, -1.0,  0.0],
                    [ 1.0,  0.0,  0.0],
                    [ 0.0,  0.0,  1.0]
                ])

                # qaut1 = T.mat2quat(Rot_A_to_B_raw)
                # quat2 = T.mat2quat(Rot_A_to_B)
                # quat_avg = 0.2 * qaut1 + 0.8 * quat2
                # quat_avg /= np.linalg.norm(quat_avg)
                # Rot_A_to_B_avg = T.quat2mat(quat_avg)
                # print('Rot_A_to_B_raw: ', Rot_A_to_B_raw)
                # # if self.count < 50:
                # #     next_ori_mat_world = Rot_A_to_B_raw.dot(next_ori_mat)
                # # else:
                # #     next_ori_mat_world = Rot_A_to_B.dot(next_ori_mat)
                next_ori_mat_world = Rot_A_to_B.dot(next_ori_mat)
                # axis_angle_vec = T.quat2axisangle(T.mat2quat(next_ori_mat_world))
                r = R.from_matrix(next_ori_mat_world)
                axis_angle_vec = r.as_rotvec()

                # 4) build your final action: [x, y, z, ax*θ, ay*θ, az*θ, gripper]

                rot6d = axisangle_to_rot6d(axis_angle_vec)
                abs_action = np.concatenate([next_pos, rot6d, [gripper_action]])
                # print("next_pos: ", next_pos, " axis_angle_vec: ", axis_angle_vec)
            

                # abs_action[0:3] = abs_action[0:3] / 1.5
                # abs_action[3:7] = abs_action[3:7] / 3.5  # normalize it to [-1, 1], approximately TODO refine this, can do pos and ori seperately
                # print("current_pos: ", current_pos, " next_pos: ", next_pos, " current_ori: ", axis_angle_vec)

                if self.record_action_statistics:
                    if not hasattr(self, 'action_max'):
                        # first-ever action: initialize max & min
                        self.action_max = list(abs_action)
                        self.action_min = list(abs_action)
                    else:
                        # update per-dimension
                        for i, v in enumerate(abs_action):
                            if v > self.action_max[i]:
                                self.action_max[i] = v
                            if v < self.action_min[i]:
                                self.action_min[i] = v
                    logger.debug('%s %s %s %s', "self.action_max: ", self.action_max, " self.action_min: ", self.action_min)

                action = env.normalize_abs_action(abs_action, action_max=env.action_max_list[0], action_min=env.action_min_list[0])  # TODO for now we only normialzie abs_action, for velocity, also need normalized?
            else:
                action[0:-1] = 5 * action[0:-1]  # the effects is similar to normalizing the teacher action; keep the range close to [-1, 1]



            action_max = np.max(abs(action))
            if action_max > 1:
                action =  action / action_max
            
            # print("action teacher: ", action, " use_abs_action: ", self.use_abs_action)
            return action
        
    def obtain_gripper_goal_for_moving_to_NutHanlde(self, obs):
        # gripper_ori_w = np.array([0.733, 0.678, 0.060, -0.018])
        # nut_ori_w = np.array([-0.019, -0.019, 0.679, 0.734])
        # nut_mat = T.quat2mat(nut_ori_w)
        # # nut_grasp_mat = T.quat2mat(gripper_ori_w)
        # mat_handle_to_nut = (nut_mat.T).dot(nut_grasp_mat)
        # mat_handle_to_nut = T.quat2mat(np.array([ 1, 0, 0, 0 ]))
        mat_handle_to_nut = R.from_quat(np.array([ 1, 0, 0, 0 ])).as_matrix()
        # print("mat_handle_to_nut: ", mat_handle_to_nut) 

        # current_pos = obs['robot0_eef_pos']
        # nut_pos = obs['SquareNut_pos']
        # current_ori = obs['robot0_eef_quat']
        # nut_ori = obs['SquareNut_quat']

        current_pos = obs[0:3]
        nut_pos = obs[3:6]
        # nut_ori = T.mat2quat(obs[15:24].reshape(3, 3))
        nut_ori_mat = obs[15:24].reshape(3, 3)
        

        mat_handle_w = nut_ori_mat.dot(mat_handle_to_nut)
        # quat_handle_w = T.mat2quat(mat_handle_w)
        quat_handle_w = R.from_matrix(mat_handle_w).as_quat()
        # print("quat_handle_w: ", quat_handle_w)

        # pos_handle_to_nutcenter = np.array([0.065, 0, 0])
        # pos_handle_to_nutcenter = np.array([0.057, 0, 0])
        pos_handle_to_nutcenter = np.array([0.059, 0, 0])
        pos_handle_to_world = nut_pos + mat_handle_w.dot(pos_handle_to_nutcenter)
        # print("pos_handle_to_world: ", pos_handle_to_world)

        quat_rotz180 = np.array([0, 0, 1, 0])
        # mat_rotz180 = T.quat2mat(quat_rotz180)
        mat_rotz180 = R.from_quat(quat_rotz180).as_matrix()
        mat_handle_w2 = mat_rotz180.dot(mat_handle_w)
        # quat_handle_w2 = T.mat2quat(mat_handle_w2)
        quat_handle_w2 = R.from_matrix(mat_handle_w2).as_quat()
        # print("quat_handle_w2: ", quat_handle_w2)

        return pos_handle_to_world, quat_handle_w, quat_handle_w2
        # return pos_handle_to_world, quat_handle_w2, quat_handle_w2
    
    # helper function for the policy of state 5: MOVE_TO_PEG
    def obtain_gripper_goal_for_moving_to_Peg(self, obs, dt = 1.0):
        # gripper_ori_w = np.array([0.733, 0.678, 0.060, -0.018])
        # nut_ori_w = np.array([-0.019, -0.019, 0.679, 0.734])
        # nut_mat = T.quat2mat(nut_ori_w)
        # # nut_grasp_mat = T.quat2mat(gripper_ori_w)
        # mat_handle_to_nut = (nut_mat.T).dot(nut_grasp_mat)

        target_orientatoin_1 = np.array([-0.000, 0.000, 0.7071068, 0.7071068])  # rot_Z_90
        target_orientatoin_2 = np.array([0, 0, 1, 0])  # rot_Z_180
        target_orientatoin_3 = np.array([ 0, 0, -0.7071068, 0.7071068 ])  # rot_Z_negative90
        # orientation 2 [0, 0, 1, 0]   # rot_Z_180
        # orientation 3 [0, 0, 0, 1 ]  # rot_Z_0
        # print("mat_handle_to_nut: ", mat_handle_to_nut) 

        # current_pos = obs['robot0_eef_pos']
        # nut_pos = obs['SquareNut_pos']
        # current_ori = obs['robot0_eef_quat']
        # nut_ori = obs['SquareNut_quat']

        # nut_ori = T.mat2quat(obs[15:24].reshape(3, 3))
        nut_ori = R.from_matrix(obs[15:24].reshape(3, 3)).as_quat()

        # mat_handle_w = (T.quat2mat(nut_ori)).dot(mat_handle_to_nut)
        # quat_handle_w = T.mat2quat(mat_handle_w)
        # print("quat_handle_w: ", quat_handle_w)

        # pos_handle_to_nutcenter = np.array([0.054, 0, 0])
        # pos_handle_to_world = nut_pos + mat_handle_w.dot(pos_handle_to_nutcenter)
        # print("pos_handle_to_world: ", pos_handle_to_world)

        # quat_rotz180 = np.array([0, 0, 1, 0])
        # mat_rotz180 = T.quat2mat(quat_rotz180)
        # mat_handle_w2 = mat_rotz180.dot(mat_handle_w)
        # quat_handle_w2 = T.mat2quat(mat_handle_w2)
        # print("quat_handle_w2: ", quat_handle_w2)

        angular_velocity_1, angle_1 = calculate_angular_velocity_to_target(nut_ori, target_orientatoin_1, dt)
        angular_velocity_2, angle_2 = calculate_angular_velocity_to_target(nut_ori, target_orientatoin_2, dt)
        angular_velocity_3, angle_3 = calculate_angular_velocity_to_target(nut_ori, target_orientatoin_3, dt)

        # Initialize minimum angle to angle_1 and corresponding angular velocity to angular_velocity_1
        min_angle = angle_1
        min_angular_velocity = angular_velocity_1

        # Check if angle_2 is smaller than the current minimum angle
        if abs(angle_2) < abs(min_angle):
            min_angle = angle_2
            min_angular_velocity = angular_velocity_2

        # Check if angle_3 is smaller than the current minimum angle
        if abs(angle_3) < abs(min_angle):
            min_angle = angle_3
            min_angular_velocity = angular_velocity_3

        # Return the angular velocity corresponding to the smallest absolute angle
        return min_angular_velocity

        

    # policy for state 1: MOVE_TOWARDS_NUT,pos goal of eef is pos_handle_to_world + np.array([0, 0, 0.05])
    def handle_move_towards_nut(self, obs, env = None):
        """Handle the logic for moving towards the nut."""
        # current_pos = obs['robot0_eef_pos']
        # nut_pos = obs['SquareNut_pos']
        # current_ori = obs['robot0_eef_quat']
        # nut_ori = obs['SquareNut_quat']
        # nut_ori_to_robot = obs['SquareNut_to_robot0_eef_quat']


        current_pos = obs[0:3]
        nut_pos = obs[3:6]
        # current_ori = T.mat2quat(obs[6:15].reshape(3, 3))
        r = R.from_matrix(obs[6:15].reshape(3, 3))
        current_ori = r.as_quat()

        # from env.robotsuite.env_robosuite import quat_to_yaw
        # nut_z_angle = quat_to_yaw(nut_ori) * 180.0 / np.pi
        # # print("current_ori: ", current_ori)
        # print('nut_pos: ', nut_pos, 'nut_ori: ', nut_ori, " nut_z_angle: ", nut_z_angle)


        # target_grasp_ori = quaternion_multiply(nut_ori, self.grasp_orientation_transform)
        pos_handle_to_world, target_grasp_ori, target_grasp_ori02 = self.obtain_gripper_goal_for_moving_to_NutHanlde(obs)
        
        # Compute position delta for velocity
        # pos_delta = self.calculate_position_delta(current_pos, pos_handle_to_world + np.array([0, 0, 0.05]))
        pos_delta = self.calculate_position_delta(current_pos, pos_handle_to_world + np.array([0, 0, 0.08]))
        # print("pos_delta: ", pos_delta)
        # Normalize delta to obtain a direction vector
        pos_direction = pos_delta / np.linalg.norm(pos_delta) if np.linalg.norm(pos_delta) > 0.01 else pos_delta
        
        # Assume a fixed speed towards the nut; adjust as necessary
        speed = 0.3  # This speed value might need tuning
        
        # Position velocity: Moving towards the nut at a constant speed in the direction calculated
        pos_velocity = pos_direction * speed
        
        # ori_velocity = self.calculate_orientation_velocity(nut_ori, target_grasp_ori)
        ori_velocity =  calculate_orientation_velocity(current_ori, target_grasp_ori, target_grasp_ori02)

        # if np.linalg.norm(ori_velocity) > 0.1:
        #     pos_velocity = 0.2 * pos_velocity
        # if np.linalg.norm(pos_velocity) > 0.1:
        #     ori_velocity = 0.5 * ori_velocity
        # The gripper command is set to open (assuming -1 is open, 1 is close)
        gripper_command = np.array([-1.0])  # Adjust based on your robot's gripper control specifics
        
        # Check if the end effector is above the nut to transition to the next state
        if self.is_above_target(current_pos, pos_handle_to_world) and np.linalg.norm(ori_velocity) < 0.05:
            self.current_state += 1  # Advance to the next state
        
        # The action is a concatenation of position velocity, orientation velocity, and gripper command
        action = np.concatenate([pos_velocity, ori_velocity, gripper_command])
        # print('action: ', action)
        return action
    
    # policy for state 2: ADJUST_FOR_GRASP, pos goal of eef is pos_handle_to_world
    def handle_adjust_for_grasp(self, obs, height = np.array([0, 0, 0.01]), state = 'ADJUST_FOR_GRASP'):
        """Handle the logic for moving towards the nut."""
        # current_pos = obs['robot0_eef_pos']
        # nut_pos = obs['SquareNut_pos']
        # current_ori = obs['robot0_eef_quat']
        # nut_ori = obs['SquareNut_quat']
        # nut_ori_to_robot = obs['SquareNut_to_robot0_eef_quat']

        current_pos = obs[0:3]
        nut_pos = obs[3:6]
        # current_ori = T.mat2quat(obs[6:15].reshape(3, 3))
        current_ori = R.from_matrix(obs[6:15].reshape(3, 3)).as_quat()
        # nut_ori = T.mat2quat(obs[15:24].reshape(3, 3))

        # print("current_ori: ", current_ori)
        # print('nut_ori: ', nut_ori)

        # target_grasp_ori = quaternion_multiply(nut_ori, self.grasp_orientation_transform)
        pos_handle_to_world, target_grasp_ori, target_grasp_ori02 = self.obtain_gripper_goal_for_moving_to_NutHanlde(obs)
        
        # Compute position delta for velocity
        pos_delta = self.calculate_position_delta(current_pos, pos_handle_to_world -height)
        
        # Normalize delta to obtain a direction vector
        pos_direction = pos_delta / np.linalg.norm(pos_delta) if np.linalg.norm(pos_delta) > 0.01 else pos_delta
        
        # Assume a fixed speed towards the nut; adjust as necessary
        speed = 0.25  # This speed value might need tuning
        
        # Position velocity: Moving towards the nut at a constant speed in the direction calculated
        pos_velocity = pos_direction * speed
        
        # ori_velocity = self.calculate_orientation_velocity(nut_ori, target_grasp_ori)
        ori_velocity = 2.0 * calculate_orientation_velocity(current_ori, target_grasp_ori, target_grasp_ori02)

        # The gripper command is set to open (assuming -1 is open, 1 is close)
        gripper_command = np.array([-1.0])  # Adjust based on your robot's gripper control specifics
        
        # # Check if the end effector is above the nut to transition to the next state
        # print("pos_delta ", pos_delta)
        #TODO caculate the difference to the grasping point. Project the gripper position to the direction of the handle
        mat_handle_to_nut = R.from_quat(np.array([ 1, 0, 0, 0 ])).as_matrix()
        # nut_ori = T.mat2quat(obs[15:24].reshape(3, 3))
        nut_ori_mat = obs[15:24].reshape(3, 3)
        mat_handle_w = nut_ori_mat.dot(mat_handle_to_nut)

        # pos_handle_to_nutcenter = np.array([0.065, 0, 0])
        # pos_handle_to_nutcenter = np.array([0.057, 0, 0])
        handle_direction = np.array([1, 0, 0])
        handle_direction_inWorld = mat_handle_w.dot(handle_direction)


        # project current pos into handle_direction_inWorld:
        pos_delta_projected = abs(np.dot(pos_delta, handle_direction_inWorld))
        logger.debug('%s %s', "pos_delta_projected: ", pos_delta_projected)

        if self.is_position_aligned(current_pos[:2], pos_handle_to_world[:2], threshold=0.008) and abs(pos_delta[2])< 0.01 and np.linalg.norm(ori_velocity) < 0.05:
            self.current_state += 1  # Advance to the next state

        if not self.is_position_aligned(current_pos[:2], pos_handle_to_world[:2], threshold=0.01)and state == 'ADJUST_FOR_GRASP_lower':
            self.current_state -= 1
            # print("1")
        elif pos_delta_projected > 0.007 and state == 'ADJUST_FOR_GRASP_lower':
            self.current_state -= 1
            # print("2")
        
        # The action is a concatenation of position velocity, orientation velocity, and gripper command
        action = np.concatenate([pos_velocity, ori_velocity, gripper_command])
        return action

    # policy for state 3: GRASP_NUT
    def handle_grasp_nut(self, obs):
        # No movement, just close the gripper
        pos_velocity = np.array([0, 0, 0])
        ori_velocity = np.array([0, 0, 0])

        pos_handle_to_world, target_grasp_ori, target_grasp_ori02 = self.obtain_gripper_goal_for_moving_to_NutHanlde(obs)

        # gripper_q_state = 2 * obs['robot0_gripper_qpos'][0]  # 2 * 0.04 for open; 2 * 0.015 when close
        current_pos = obs[0:3]
        gripper_q_state = 2 * obs[24]
        grasped = obs[-1]
        # Close the gripper
        gripper_command = 1.0

        # if self.is_gripper_closed(gripper_state) and self.is_nut_grasped(nut_state):
        #     self.current_state += 1

        if gripper_q_state < 0.031 and gripper_q_state > 0.02 and grasped == 1.0:
            self.current_state += 1
        elif gripper_q_state < 0.02:
            self.current_state -= 1
        # elif grasped != 1.0 and abs(current_pos[2] - 0.828) > 0.05:
        elif grasped != 1.0 and abs(current_pos[2] - 0.828) > 0.01:
            self.current_state -= 1
        elif not self.is_position_aligned(current_pos[:2], pos_handle_to_world[:2], threshold=0.008):
            self.current_state -= 1


        action = np.concatenate([pos_velocity, ori_velocity, [gripper_command]])
        return action

    # policy for state 4: LIFT_NUT
    def handle_lift_nut(self, obs):
        # current_pos = obs['robot0_eef_pos']
        # nut_pos = obs['SquareNut_pos']
        # current_ori = obs['robot0_eef_quat']
        # nut_ori = obs['SquareNut_quat']
        # nut_ori_to_robot = obs['SquareNut_to_robot0_eef_quat']

        current_pos = obs[0:3]
        nut_pos = obs[3:6]

        grasped = obs[-1]
        # print("current_ori: ", current_ori)
        # print('nut_ori: ', nut_ori)


        # # target_grasp_ori = quaternion_multiply(nut_ori, self.grasp_orientation_transform)
        # pos_handle_to_world, target_grasp_ori, target_grasp_ori02 = self.obtain_gripper_goal_for_moving_to_NutHanlde(obs)


        # Lift the nut upwards by a small amount
        pos_velocity = np.array([0, 0, 0.3])  # Adjust the speed and direction as needed

        ori_velocity = np.array([0, 0, 0]) 
        gripper_command = 1.0  # Gripper remains closed

        if nut_pos[2] > 0.98 and grasped == 1:
            self.current_state += 1
        elif grasped != 1:
            self.current_state = 0
        action = np.concatenate([pos_velocity, ori_velocity, [gripper_command]])
        return action
    
    # policy for state 5: MOVE_TO_PEG
    def handle_move_to_peg_uper01(self, obs):
        current_pos = obs[0:3]
        nut_pos = obs[3:6]

        peg_up_pos = np.array([0.227, 0.101, 0.98])
        grasped = obs[-1]

        pos_delta = self.calculate_position_delta(nut_pos, peg_up_pos)
        pos_direction = pos_delta / np.linalg.norm(pos_delta) if np.linalg.norm(pos_delta) > 0.01 else pos_delta
        # speed = 0.25 
        speed = 0.3
        pos_velocity = pos_direction * speed

        ori_velocity = 0.4 * self.obtain_gripper_goal_for_moving_to_Peg(obs)  # Orientation adjustment may be necessary based on the task
        gripper_command = 1.0  # Keep the gripper closed
        
        if np.linalg.norm(pos_delta) < 0.01 and np.linalg.norm(ori_velocity) < 0.02:
             self.current_state += 1
        elif grasped != 1:
            self.current_state = 0

        action = np.concatenate([pos_velocity, ori_velocity, [gripper_command]])
        return action
    
    # policy for state 6: ALIGN_OVER_PEG 
    def handle_move_to_peg_uper02(self, obs):
        current_pos = obs[0:3]
        nut_pos = obs[3:6]

        peg_up_pos = np.array([0.227, 0.101, 0.96])

        pos_delta = self.calculate_position_delta(nut_pos, peg_up_pos)
        pos_direction = pos_delta / np.linalg.norm(pos_delta) if np.linalg.norm(pos_delta) > 0.01 else pos_delta
        speed = 0.25 
        pos_velocity = pos_direction * speed

        ori_velocity = 0.4 * self.obtain_gripper_goal_for_moving_to_Peg(obs)  # Orientation adjustment may be necessary based on the task
        gripper_command = 1.0  # Keep the gripper closed
        
        if np.linalg.norm(pos_delta) < 0.01 and np.linalg.norm(ori_velocity) < 0.015:
             self.current_state += 1

        action = np.concatenate([pos_velocity, ori_velocity, [gripper_command]])
        return action
    
    # policy for state 7: ALIGN_OVER_PEG 
    def handle_move_to_peg_lower(self, obs):
        current_pos = obs[0:3]
        nut_pos = obs[3:6]

        # peg_up_pos = np.array([0.227, 0.101, 0.89])
        peg_up_pos = np.array([0.227, 0.101, 0.91])

        pos_delta = self.calculate_position_delta(nut_pos, peg_up_pos)
        pos_direction = pos_delta / np.linalg.norm(pos_delta) if np.linalg.norm(pos_delta) > 0.01 else pos_delta
        speed = 0.25 
        pos_velocity = pos_direction * speed

        ori_velocity = 0.1 * self.obtain_gripper_goal_for_moving_to_Peg(obs)  # Orientation adjustment may be necessary based on the task
        gripper_command = 1.0  # Keep the gripper closed
        
        if np.linalg.norm(pos_delta) < 0.01 and np.linalg.norm(ori_velocity) < 0.05:
             self.current_state += 1

        action = np.concatenate([pos_velocity, ori_velocity, [gripper_command]])
        return action

    # policy for state 8
    def release_nut(self, obs):
        pos_velocity = np.array([0, 0, 0])
        ori_velocity = np.array([0, 0, 0])

        # Close the gripper
        gripper_command = -1.0
        action = np.concatenate([pos_velocity, ori_velocity, [gripper_command]])
        return action 

    def is_above_target(self, current_pos, target_pos, threshold=0.01):
        """Check if the current position is above the target within a certain threshold."""
        return np.linalg.norm(current_pos[:2] - target_pos[:2]) < threshold and abs(current_pos[2] - target_pos[2]) > threshold

    def is_position_aligned(self, current_pos, target_pos, threshold=0.012):
        # Check if the current position is within the threshold distance of the target position.
        distance = np.linalg.norm(current_pos - target_pos)
        # print("distance: ", distance)
        return distance < threshold

    def execute_action(self, action):
        """Execute the given action."""
        # Placeholder for executing an action, such as moving the end effector or operating the gripper
        pass

# Example usage
# policy = NutAssemblyPolicy()
# observation = {
#     'robot0_eef_pos': np.array([-0.15318896,  0.12206221,  1.06116653]),
#     'SquareNut_pos': np.array([-0.11467006,  0.16964621,  0.82997895]),
#     # Include other necessary parts of the observation here
# }
# action = policy.get_action(observation)
