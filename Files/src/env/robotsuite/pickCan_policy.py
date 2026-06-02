import logging

logger = logging.getLogger(__name__)

import numpy as np
import robosuite.utils.transform_utils as T
from robosuite.utils.sim_utils import check_contact

from env.robotsuite.env_robosuite import axisangle_to_rot6d

from scipy.spatial.transform import Rotation as R

'''Note, to make it easier for our hand-crafted policy (the tool sometimes stuck at the frame and not dropping down),
we changed the tool config to  '''
# self.real_tool_args = dict( 

#             name="tool", 

#             handle_size=( 

#                 (16.5 / 200.0), 

#                 (1.75 / 200.0), 

#                 (0.32 / 200.0), 

#             ),  # 16.5 cm length, 1.75 cm width, 0.32 cm thick (1.5 cm with foam) 

#             outer_radius_1=(3.5 / 200.0),  # larger hole 3.5 cm outer diameter 

#             inner_radius_1=(2.6 / 200.0),  # reduced larger hole 2.1 cm inner diameter (from real world 2.3 cm) 

#             height_1=(0.5 / 200.0),  # 0.7 cm height 

#             outer_radius_2=(3.0 / 200.0),  # smaller hole 3 cm outer diameter 

#             inner_radius_2=(2.0 / 200.0),  # smaller hole 2 cm outer diameter 

#             height_2=(0.7 / 200.0),  # 0.7 cm height 

#             ngeoms=8, 

#             grip_size=((3 / 200.0), (8.0 / 200.0)),  # 8 cm length, 3 cm thick 

#             density=2000.0, 

#             solref=(0.02, 1.0), 

#             solimp=(0.998, 0.998, 0.001), 

#             friction=(0.95, 0.3, 0.1), 

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

    angular_velocity = 0.5 * angular_velocity / np.linalg.norm(angular_velocity) if np.linalg.norm(angular_velocity) > 0.5 else angular_velocity


    return angular_velocity, angle
    
def calculate_orientation_velocity(current_ori, target_ori_1, target_ori_2, dt=1.0):
    """Calculate orientation velocity from current orientation to the closest target orientation."""
   
    # Calculate angular velocity and angle for both target orientations
    angular_velocity_1, angle_1 = calculate_angular_velocity_to_target(current_ori, target_ori_1, dt)
    angular_velocity_2, angle_2 = calculate_angular_velocity_to_target(current_ori, target_ori_2, dt)
    # print("angular_velocity_1: ", angular_velocity_1, " angle_1: ", angle_1)
    # print("angular_velocity_2: ", angular_velocity_2, " angle_2: ", angle_2)
    # Choose the target orientation with the smallest absolute angle
    if abs(angle_1) < abs(angle_2):
        return angular_velocity_1
    else:
        return angular_velocity_2
    

def calculate_orientation_velocity_withAngles(current_ori, target_ori_1, target_ori_2, dt=1.0):
    """Calculate orientation velocity from current orientation to the closest target orientation."""
   
    # Calculate angular velocity and angle for both target orientations
    angular_velocity_1, angle_1 = calculate_angular_velocity_to_target(current_ori, target_ori_1, dt)
    angular_velocity_2, angle_2 = calculate_angular_velocity_to_target(current_ori, target_ori_2, dt)
    # print("angular_velocity_1: ", angular_velocity_1, " angle_1: ", angle_1)
    # print("angular_velocity_2: ", angular_velocity_2, " angle_2: ", angle_2)
    # Choose the target orientation with the smallest absolute angle
    if abs(angle_1) < abs(angle_2):
        return angular_velocity_1, angle_1
    else:
        return angular_velocity_2, angle_2
    

class PickCanPolicy:
    def __init__(self, use_abs_action = False):
        self.use_abs_action = use_abs_action
        self.current_state = 0
        self.state_machine = [
            "MOVE_TOWARDS_CAN",
            "ADJUST_FOR_GRASP_CAN",
            "GRASP_CAN",
            "LIFT_CAN",
            "MOVE_TO_BASE",
            "ALIGN_OVER_BASE",
            "RELEASE_FRAME", 
            'MOVE_TO_SAFE_REGION',
            "MOVE_TOWARDS_TOOL",
            "ADJUST_FOR_GRASP_TOOL",
            "GRASP_TOOL",
            "LIFT_TOOL", 
            "MOVE_TO_FRAME", 
            "ALIGN_OVER_FRAME",
            "RELEASE_TOOL",
            "Done"
        ]

        # used to record and caculate the action statistics that are used to do normalization
        self.record_action_statistics = False
        # self.record_action_statistics = True

        # self.grasp_orientation_transform_quat = T.quat_conjugate(np.array([1, 0, 0, 0 ]) )
        self.grasp_orientation_transform_quat =np.array([-1.1431241e-04, -3.5661459e-04,  7.0725894e-01,  7.0695448e-01]) 
        
        self.grasp_orientation_transform_mat = T.quat2mat(self.grasp_orientation_transform_quat)

        self.x_axis_gripper_in_world = None

    def reset(self):
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

    
    def get_action(self, obs, env):
        obs = env.env._get_observations()
        # print("obs: ", obs)
        """Determine the action to take based on the current state and observation."""
        if self.current_state < len(self.state_machine):
            state = self.state_machine[self.current_state]
            action = None  # Placeholder for actual action logic
            logger.debug('%s %s', "state: ", state)
            if state == "MOVE_TOWARDS_CAN":
                self.x_axis_gripper_in_world = None
                action = self.handle_move_towards_frame(obs, env)
                # action = self.handle_move_towards_tool(obs, env)
            elif state == 'ADJUST_FOR_GRASP_CAN':
                action = self.handle_adjust_for_grasp_frame(obs, env)
            elif state == 'GRASP_CAN':
                action = self.handle_grasp_frame(obs, env)
            elif state == 'LIFT_CAN':
                action = self.handle_lift_frame(obs, env)
            elif state == 'MOVE_TO_BASE':
                action = self.handle_frame_move_to_base(obs, env)
            elif state == 'ALIGN_OVER_BASE':
                action = self.handle_frame_move_to_base_lower(obs, env)
            elif state == 'RELEASE_FRAME':
                action = self.release_frame(obs, env)
            elif state == 'MOVE_TO_SAFE_REGION':
                action = self.move_to_safe_region(obs)
            elif state == 'MOVE_TOWARDS_TOOL':
                action = self.handle_move_towards_tool(obs, env)
            elif state == "ADJUST_FOR_GRASP_TOOL":
                action = self.handle_adjust_for_grasp_tool(obs, env)
            elif state == "GRASP_TOOL":
                action = self.handle_grasp_tool(obs, env)
            elif state == "LIFT_TOOL":
                action = self.handle_lift_tool(obs, env)
            elif state == "MOVE_TO_FRAME":
                action = self.move_to_frame_with_tool(obs, env, height=np.array([0, 0, 0.06]), speed=0.25)
            elif state == "ALIGN_OVER_FRAME":
                action = self.move_to_frame_with_tool(obs, env, height=np.array([0, 0, 0.005]), speed= 0.1)
            elif state == "RELEASE_TOOL":
                action = self.release_tool(obs, env, height=np.array([0, 0, -0.03]), speed= 0.1)
            elif state == "Done":
                action = np.zeros(7)
                action[-1] = -1
            
            # Execute the action determined by the current state
            # This is a placeholder to illustrate concept
            # In practice, action execution would involve interacting with the robot's API
            self.execute_action(action)

            if self.use_abs_action:
                # convert the relative action to absolute
                current_pos = obs['robot0_eef_pos']
                current_ori = obs['robot0_eef_quat']

     
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

                next_ori_mat_world = Rot_A_to_B.dot(next_ori_mat)
                # axis_angle_vec = T.quat2axisangle(T.mat2quat(next_ori_mat_world))
                r = R.from_matrix(next_ori_mat_world)
                axis_angle_vec = r.as_rotvec()

                # 4) build your final action: [x, y, z, ax*θ, ay*θ, az*θ, gripper]

                rot6d = axisangle_to_rot6d(axis_angle_vec)
                abs_action = np.concatenate([next_pos, rot6d, [gripper_action]])
                # print("next_pos: ", next_pos, " axis_angle_vec: ", axis_angle_vec)
            
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
            
            # print("action teacher: ", action)

            return action

    def move_to_frame_with_tool(self, obs, env, height, speed = 0.25):
        current_pos = obs['robot0_eef_pos']
        # current_ori = T.mat2quat(obs[6:15].reshape(3, 3))
        current_ori = obs['robot0_eef_quat']
        gripper_q_state = 2 * obs['robot0_gripper_qpos'][0]
        grasped = env.env._check_grasp(gripper=env.env.robots[0].gripper, object_geoms=env.env.tool.contact_geoms)
        # print("grasped: ", grasped)

        tool_hole1_center = env.env.sim.data.site_xpos[env.env.obj_site_id["tool_hole1_center"]]
        frame_hook_endpoint_pos = env.env.sim.data.site_xpos[env.env.obj_site_id["frame_hang_site"]] + height

        hook_endpoint = env.env.sim.data.site_xpos[env.env.obj_site_id["frame_mount_site"]]
        frame_hook_vec =frame_hook_endpoint_pos - hook_endpoint
        frame_hook_vec[2] = 0
        frame_hook_length = np.linalg.norm(frame_hook_vec)
        frame_hook_vec = -frame_hook_vec / frame_hook_length
        
        pos_delta = self.calculate_position_delta(tool_hole1_center, frame_hook_endpoint_pos)
        # pos_delta = np.zeros(3)
        pos_direction = pos_delta / np.linalg.norm(pos_delta) if np.linalg.norm(pos_delta) > 0.01 else pos_delta / (np.linalg.norm(pos_delta)+0.01)
        # speed = 0.25
        # speed = 0.15 
        pos_velocity = pos_direction * speed
        # print("frame_tip_pos: ", frame_tip_pos, " base_walls_geom_positions_average: ", base_walls_geom_positions_average)

        # x = self.x_axis_gripper_in_world
        z = np.array([0, 0, -1])
        y = frame_hook_vec
        # Calculate x as the cross product of y and z
        x = np.cross(y, z)
        # Normalize x to make it a unit vector
        x = x / np.linalg.norm(x)
        # Construct the orientation matrix
        gripper_assemble_frame_pose_ori_matrix = np.column_stack((x, y, z))
        gripper_assemble_frame_pose_ori_matrix = gripper_assemble_frame_pose_ori_matrix.dot(self.grasp_orientation_transform_mat)
        gripper_grasp_frame_pose_ori_quat = T.mat2quat(gripper_assemble_frame_pose_ori_matrix)

        # print("gripper_grasp_frame_pose_ori_matrix: ", gripper_grasp_frame_pose_ori_matrix)
        quat_rotx180 = np.array([0.8939967, 0, 0, -0.4480736])
        mat_rotz180 = T.quat2mat(quat_rotx180)
        gripper_assemble_frame_pose02_ori_matrix = mat_rotz180.dot(gripper_assemble_frame_pose_ori_matrix)
        gripper_assemble_frame_pose02_ori_matrix = gripper_assemble_frame_pose02_ori_matrix.dot(self.grasp_orientation_transform_mat)
        gripper_grasp_frame_pose02_ori_quat = T.mat2quat(gripper_assemble_frame_pose02_ori_matrix)

        ori_velocity = calculate_orientation_velocity(current_ori, gripper_grasp_frame_pose_ori_quat, gripper_grasp_frame_pose_ori_quat)

        # ori_velocity = 0.1 * np.array([-1, 0, 0])
        gripper_command = 1.0  # Keep the gripper closed
        
        if np.linalg.norm(pos_delta[0:2]) > 0.003 and tool_hole1_center[2] > frame_hook_endpoint_pos[2]:
            pos_velocity[2] = 0

        if np.linalg.norm(pos_delta[0:2]) > 0.003 and tool_hole1_center[2] < frame_hook_endpoint_pos[2]:
            self.current_state -= 1
        if np.linalg.norm(pos_delta) < 0.01 and np.linalg.norm(pos_delta[0:2]) < 0.003 and np.linalg.norm(ori_velocity) < 0.03:
            self.current_state += 1
        elif grasped != 1:
            self.current_state = 9  # MOVE_TOWARDS_TOOL

        action = np.concatenate([pos_velocity, ori_velocity, [gripper_command]])
        return action
    
    def release_tool(self, obs, env, height=np.array([0,0,0]), speed = 0.25):
        current_pos = obs['robot0_eef_pos']
        # current_ori = T.mat2quat(obs[6:15].reshape(3, 3))
        current_ori = obs['robot0_eef_quat']
        gripper_q_state = 2 * obs['robot0_gripper_qpos'][0]
        grasped = env.env._check_grasp(gripper=env.env.robots[0].gripper, object_geoms=env.env.tool.contact_geoms)

        tool_hole1_center = env.env.sim.data.site_xpos[env.env.obj_site_id["tool_hole1_center"]]
        frame_hook_endpoint_pos = env.env.sim.data.site_xpos[env.env.obj_site_id["frame_hang_site"]] + height

        hook_endpoint = env.env.sim.data.site_xpos[env.env.obj_site_id["frame_mount_site"]]
        frame_hook_vec = env.env.sim.data.site_xpos[env.env.obj_site_id["frame_intersection_site"]] - hook_endpoint
        frame_hook_length = np.linalg.norm(frame_hook_vec)
        frame_hook_vec = -frame_hook_vec / frame_hook_length   

        pos_delta = self.calculate_position_delta(tool_hole1_center, frame_hook_endpoint_pos)
        # pos_delta = np.zeros(3)
        pos_direction = pos_delta / np.linalg.norm(pos_delta) if np.linalg.norm(pos_delta) > 0.01 else pos_delta / (np.linalg.norm(pos_delta)+0.01)
        # speed = 0.25
        # speed = 0.15 
        pos_velocity = pos_direction * speed
        # print("frame_tip_pos: ", frame_tip_pos, " base_walls_geom_positions_average: ", base_walls_geom_positions_average)

        # x = self.x_axis_gripper_in_world
        z = np.array([0, 0, -1])
        y = np.array([0, 1, 0])
        # Calculate x as the cross product of y and z
        x = np.cross(y, z)
        # Normalize x to make it a unit vector
        x = x / np.linalg.norm(x)
        # Construct the orientation matrix
        gripper_assemble_frame_pose_ori_matrix = np.column_stack((x, y, z))
        gripper_assemble_frame_pose_ori_matrix = gripper_assemble_frame_pose_ori_matrix.dot(self.grasp_orientation_transform_mat)
        gripper_grasp_frame_pose_ori_quat = T.mat2quat(gripper_assemble_frame_pose_ori_matrix)

        # print("gripper_grasp_frame_pose_ori_matrix: ", gripper_grasp_frame_pose_ori_matrix)
        quat_rotx180 = np.array([0.8939967, 0, 0, -0.4480736])
        mat_rotz180 = T.quat2mat(quat_rotx180)
        gripper_assemble_frame_pose02_ori_matrix = mat_rotz180.dot(gripper_assemble_frame_pose_ori_matrix)
        gripper_assemble_frame_pose02_ori_matrix = gripper_assemble_frame_pose02_ori_matrix.dot(self.grasp_orientation_transform_mat)
        gripper_grasp_frame_pose02_ori_quat = T.mat2quat(gripper_assemble_frame_pose02_ori_matrix)

        ori_velocity = calculate_orientation_velocity(current_ori, gripper_grasp_frame_pose_ori_quat, gripper_grasp_frame_pose_ori_quat)

        # ori_velocity = 0.1 * np.array([-1, 0, 0])
        gripper_command = -1.0  # Keep the gripper closed
        
        if np.linalg.norm(pos_delta) < 0.01 and np.linalg.norm(pos_delta[0:2]) < 0.003 and np.linalg.norm(ori_velocity) < 0.03:
             self.current_state += 1

        action = np.concatenate([pos_velocity, ori_velocity, [gripper_command]])
        return action


    def handle_lift_tool(self, obs, env):
        current_pos = obs['robot0_eef_pos']
        # current_ori = T.mat2quat(obs[6:15].reshape(3, 3))
        current_ori = obs['robot0_eef_quat']
        gripper_q_state = 2 * obs['robot0_gripper_qpos'][0]
        grasped = env.env._check_grasp(gripper=env.env.robots[0].gripper, object_geoms=env.env.tool.contact_geoms)
        tool_handle = env.env.sim.data.body_xpos[env.env.obj_body_id['tool']] 

        # target_grasp_ori = quaternion_multiply(nut_ori, self.grasp_orientation_transform)
        # pos_handle_to_world, target_grasp_ori, target_grasp_ori02 = self.obtain_gripper_goal_for_moving_to_NutHanlde(obs)


        # Lift the nut upwards by a small amount
        pos_velocity = np.array([0, 0, 0.4])  # Adjust the speed and direction as needed

        ori_velocity = np.array([0, 0, 0]) 
        gripper_command = 1.0  # Gripper remains closed

        if tool_handle[2] > 1.05 and grasped == 1:
        # if tool_handle[2] > 0.9 and grasped == 1:
            self.current_state += 1
        elif grasped != 1:
            self.current_state -= 2
        action = np.concatenate([pos_velocity, ori_velocity, [gripper_command]])
        return action
        
    def handle_adjust_for_grasp_tool(self, obs, env):
        current_pos = obs['robot0_eef_pos']
        # current_ori = T.mat2quat(obs[6:15].reshape(3, 3))
        current_ori = obs['robot0_eef_quat']    

        tool_hole1_center = env.env.sim.data.site_xpos[env.env.obj_site_id["tool_hole1_center"]]
        # print("obj_site_id: ", env.env.obj_site_id.keys())
        # print("obj_geom_id: ", env.env.obj_geom_id.keys())
        # print("obj_body_Id: ", env.env.obj_body_id.keys())
        tool_handle = env.env.sim.data.body_xpos[env.env.obj_body_id['tool']] + np.array([0, 0, 0])

        tool_vec = tool_hole1_center - tool_handle
        tool_vec[2] = 0
        tool_vec_length = np.linalg.norm(tool_vec)
        tool_vec = -tool_vec / tool_vec_length
        z = np.array([0, 0, -1])
        # Calculate x as the cross product of y and z
        x = np.cross(tool_vec, z)
        # Normalize x to make it a unit vector
        x = x / np.linalg.norm(x)
        # Construct the orientation matrix
        gripper_grasp_tool_pose_ori_matrix = np.column_stack((x, tool_vec, z))
        gripper_grasp_tool_pose_ori_matrix = gripper_grasp_tool_pose_ori_matrix.dot(self.grasp_orientation_transform_mat)
        gripper_grasp_tool_pose_ori_quat = T.mat2quat(gripper_grasp_tool_pose_ori_matrix)

        quat_rotz180 = np.array([0, 0, 1, 0])
        mat_rotz180 = T.quat2mat(quat_rotz180)
        # gripper_grasp_tool_pose02_ori_matrix = gripper_grasp_tool_pose_ori_matrix.dot(self.grasp_orientation_transform_mat)
        gripper_grasp_tool_pose02_ori_matrix = mat_rotz180.dot(gripper_grasp_tool_pose_ori_matrix)
        gripper_grasp_tool_pose02_ori_quat = T.mat2quat(gripper_grasp_tool_pose02_ori_matrix)
        
        # Compute position delta for velocity
        goal_pos = tool_handle
        pos_delta = self.calculate_position_delta(current_pos, goal_pos)
        
        # Normalize delta to obtain a direction vector
        pos_direction = pos_delta / np.linalg.norm(pos_delta) if np.linalg.norm(pos_delta) > 0.01 else pos_delta

        speed = 0.25  # This speed value might need tuning
        
        # Position velocity: Moving towards the nut at a constant speed in the direction calculated
        pos_velocity = pos_direction * speed
        
        # ori_velocity = np.zeros(3)
        ori_velocity = calculate_orientation_velocity(current_ori, gripper_grasp_tool_pose_ori_quat, gripper_grasp_tool_pose02_ori_quat)

        # The gripper command is set to open (assuming -1 is open, 1 is close)
        gripper_command = np.array([-1.0])  # Adjust based on your robot's gripper control specifics
        
        # # # Check if the end effector is above the nut to transition to the next state
        # if self.is_above_target(current_pos, goal_pos -np.array([0, 0, 0.01]) ):
        #     self.current_state += 1  # Advance to the next state
        
        if self.is_position_aligned(current_pos, goal_pos) and abs(current_pos[2] - 0.818)< 0.01 and np.linalg.norm(ori_velocity) < 0.05:
            self.current_state += 1 
        
        # The action is a concatenation of position velocity, orientation velocity, and gripper command
        action = np.concatenate([pos_velocity, ori_velocity, gripper_command])
        # action = np.zeros(7)
        return action
    
    def handle_move_towards_tool(self, obs, env):
        current_pos = obs['robot0_eef_pos']
        # current_ori = T.mat2quat(obs[6:15].reshape(3, 3))
        current_ori = obs['robot0_eef_quat']    

        # pos_handle_to_world, target_grasp_ori, target_grasp_ori02 = self.obtain_gripper_goal_for_moving_to_NutHanlde(obs)
        
        tool_hole1_center = env.env.sim.data.site_xpos[env.env.obj_site_id["tool_hole1_center"]]
        # print("obj_site_id: ", env.env.obj_site_id.keys())
        # print("obj_geom_id: ", env.env.obj_geom_id.keys())
        # print("obj_body_Id: ", env.env.obj_body_id.keys())
        tool_handle = env.env.sim.data.body_xpos[env.env.obj_body_id['tool']] + np.array([0, 0, 0.05])

        tool_vec = tool_hole1_center - tool_handle
        tool_vec[2] = 0
        tool_vec_length = np.linalg.norm(tool_vec)
        tool_vec = -tool_vec / tool_vec_length
        z = np.array([0, 0, -1])
        # Calculate x as the cross product of y and z
        x = np.cross(tool_vec, z)
        # Normalize x to make it a unit vector
        x = x / np.linalg.norm(x)
        # Construct the orientation matrix
        gripper_grasp_tool_pose_ori_matrix = np.column_stack((x, tool_vec, z))
        gripper_grasp_tool_pose_ori_matrix = gripper_grasp_tool_pose_ori_matrix.dot(self.grasp_orientation_transform_mat)
        gripper_grasp_tool_pose_ori_quat = T.mat2quat(gripper_grasp_tool_pose_ori_matrix)

        quat_rotz180 = np.array([0, 0, 1, 0])
        mat_rotz180 = T.quat2mat(quat_rotz180)
        # gripper_grasp_tool_pose02_ori_matrix = gripper_grasp_tool_pose_ori_matrix.dot(self.grasp_orientation_transform_mat)
        gripper_grasp_tool_pose02_ori_matrix = mat_rotz180.dot(gripper_grasp_tool_pose_ori_matrix)
        gripper_grasp_tool_pose02_ori_quat = T.mat2quat(gripper_grasp_tool_pose02_ori_matrix)
        

        # Compute position delta for velocity
        goal_pos = tool_handle
        pos_delta = self.calculate_position_delta(current_pos, goal_pos)
        # print("pos_delta: ", pos_delta)
        
        # Normalize delta to obtain a direction vector
        pos_direction = pos_delta / np.linalg.norm(pos_delta) if np.linalg.norm(pos_delta) > 0.01 else pos_delta
        
        # Assume a fixed speed towards the nut; adjust as necessary
        speed = 0.35  # This speed value might need tuning
        
        # Position velocity: Moving towards the nut at a constant speed in the direction calculated
        pos_velocity = pos_direction * speed
        
        # ori_velocity = np.zeros(3)
        ori_velocity = calculate_orientation_velocity(current_ori, gripper_grasp_tool_pose_ori_quat, gripper_grasp_tool_pose02_ori_quat)

        # The gripper command is set to open (assuming -1 is open, 1 is close)
        gripper_command = np.array([-1.0])  # Adjust based on your robot's gripper control specifics
        
        # # # Check if the end effector is above the nut to transition to the next state
        if self.is_above_target(current_pos, goal_pos -np.array([0, 0, 0.01]) ):
            self.current_state += 1  # Advance to the next state
        
        # The action is a concatenation of position velocity, orientation velocity, and gripper command
        action = np.concatenate([pos_velocity, ori_velocity, gripper_command])
        # action = np.zeros(7)
        return action
    
    def handle_grasp_tool(self, obs, env):
        # No movement, just close the gripper
        pos_velocity = np.array([0, 0, 0])
        ori_velocity = np.array([0, 0, 0])

        # gripper_q_state = 2 * obs['robot0_gripper_qpos'][0]  # 2 * 0.04 for open; 2 * 0.015 when close
        current_pos = obs['robot0_eef_pos']
        # current_ori = T.mat2quat(obs[6:15].reshape(3, 3))
        current_ori = obs['robot0_eef_quat']
        gripper_q_state = 2 * obs['robot0_gripper_qpos'][0]
        grasped = env.env._check_grasp(gripper=env.env.robots[0].gripper, object_geoms=env.env.tool.contact_geoms)
        # print("grasped: ", grasped)
        # Close the gripper
        gripper_command = 1.0

        # if self.is_gripper_closed(gripper_state) and self.is_nut_grasped(nut_state):
        #     self.current_state += 1

        if gripper_q_state < 0.031 and gripper_q_state > 0.02 and grasped == 1.0:
            self.current_state += 1
        elif gripper_q_state < 0.02:
            self.current_state -= 1
        elif grasped != 1.0 and abs(current_pos[2] - 0.828) > 0.05:
            self.current_state -= 1

        action = np.concatenate([pos_velocity, ori_velocity, [gripper_command]])
        return action
    
    # policy for state 1: MOVE_TOWARDS_NUT,pos goal of eef is pos_handle_to_world + np.array([0, 0, 0.05])
    def handle_move_towards_frame(self, obs, env):
        current_pos = obs['robot0_eef_pos']
        # current_ori = T.mat2quat(obs[6:15].reshape(3, 3))
        current_ori = obs['robot0_eef_quat']

        can_pose = obs['Can_pos']
        can_quat = obs['Can_quat']

        # print("current_ori: ", current_ori)  # current_ori:  [0.99563103 0.02639328 0.08953752 0.00229967]
        # print('nut_ori: ', nut_ori)
        # hook_endpoint = env.env.sim.data.site_xpos[env.env.obj_site_id["frame_mount_site"]]
        # frame_hook_vec = env.env.sim.data.site_xpos[env.env.obj_site_id["frame_intersection_site"]] - hook_endpoint
        # frame_hook_length = np.linalg.norm(frame_hook_vec)
        # frame_hook_vec = -frame_hook_vec / frame_hook_length
        # # x is frame_hook_vec, z is 0 0 1, create the orienation matrix for me
        y = np.array([1, 0, 0])
        # y = np.array([0.5, 0.5, 0])
        z = np.array([0, 0, -1])
        # Calculate x as the cross product of y and z
        x = np.cross(y, z)
        # Normalize x to make it a unit vector
        x = x / np.linalg.norm(x)
        # Construct the orientation matrix
        gripper_grasp_frame_pose_ori_matrix = np.column_stack((x, y, z))
        gripper_grasp_frame_pose_ori_matrix = gripper_grasp_frame_pose_ori_matrix.dot(self.grasp_orientation_transform_mat)
        gripper_grasp_frame_pose_ori_quat = T.mat2quat(gripper_grasp_frame_pose_ori_matrix)
        ori_velocity = calculate_orientation_velocity(current_ori, gripper_grasp_frame_pose_ori_quat, gripper_grasp_frame_pose_ori_quat)



        
        goal_pos = can_pose + np.array([0, 0, 0.075])
        # pos_handle_to_world, target_grasp_ori, target_grasp_ori02 = self.obtain_gripper_goal_for_moving_to_NutHanlde(obs)
        
        # Compute position delta for velocity
        pos_delta = self.calculate_position_delta(current_pos, goal_pos)
        # print("pos_delta: ", pos_delta)
        
        # Normalize delta to obtain a direction vector
        pos_direction = pos_delta / np.linalg.norm(pos_delta) if np.linalg.norm(pos_delta) > 0.005 else pos_delta * 20
        
        # Assume a fixed speed towards the nut; adjust as necessary
        speed = 0.45  # This speed value might need tuning
        
        # Position velocity: Moving towards the nut at a constant speed in the direction calculated
        pos_velocity = pos_direction * speed
        
        # ori_velocity = self.calculate_orientation_velocity(nut_ori, target_grasp_ori)
        # ori_velocity = calculate_orientation_velocity(current_ori, gripper_grasp_frame_pose_ori_quat, gripper_grasp_frame_pose_ori_quat)
        # ori_velocity = np.zeros(3)

        # The gripper command is set to open (assuming -1 is open, 1 is close)
        gripper_command = np.array([-1.0])  # Adjust based on your robot's gripper control specifics
        
        # # Check if the end effector is above the nut to transition to the next state
        if self.is_above_target(current_pos, goal_pos -np.array([0, 0, 0.02]) ):
            self.current_state += 1  # Advance to the next state
        
        # The action is a concatenation of position velocity, orientation velocity, and gripper command
        action = np.concatenate([pos_velocity, ori_velocity, gripper_command])
        # action = np.zeros(7)
        return action
    
    # policy for state 2: ADJUST_FOR_GRASP, pos goal of eef is pos_handle_to_world
    def handle_adjust_for_grasp_frame(self, obs, env):
        current_pos = obs['robot0_eef_pos']
        # current_ori = T.mat2quat(obs[6:15].reshape(3, 3))
        current_ori = obs['robot0_eef_quat']

        can_pose = obs['Can_pos']
        can_quat = obs['Can_quat']

        y = np.array([1, 0, 0])
        # y = np.array([0.5, 0.5, 0])
        z = np.array([0, 0, -1])
        # Calculate x as the cross product of y and z
        x = np.cross(y, z)
        # Normalize x to make it a unit vector
        x = x / np.linalg.norm(x)
        # Construct the orientation matrix
        gripper_grasp_frame_pose_ori_matrix = np.column_stack((x, y, z))
        gripper_grasp_frame_pose_ori_matrix = gripper_grasp_frame_pose_ori_matrix.dot(self.grasp_orientation_transform_mat)
        gripper_grasp_frame_pose_ori_quat = T.mat2quat(gripper_grasp_frame_pose_ori_matrix)
        ori_velocity = calculate_orientation_velocity(current_ori, gripper_grasp_frame_pose_ori_quat, gripper_grasp_frame_pose_ori_quat)



        # frame_intersection_pos = env.env.sim.data.site_xpos[env.env.obj_site_id["frame_intersection_site"]] + frame_hook_vec * 0.025
        goal_pos = can_pose + np.array([0, 0, +0.015])
        # pos_handle_to_world, target_grasp_ori, target_grasp_ori02 = self.obtain_gripper_goal_for_moving_to_NutHanlde(obs)
        
        # Compute position delta for velocity
        pos_delta = self.calculate_position_delta(current_pos, goal_pos)
        
        # Normalize delta to obtain a direction vector
        pos_direction = pos_delta / np.linalg.norm(pos_delta) if np.linalg.norm(pos_delta) > 0.01 else 100 * pos_delta
        
        # Assume a fixed speed towards the nut; adjust as necessary
        speed = 0.35  # This speed value might need tuning
        
        # Position velocity: Moving towards the nut at a constant speed in the direction calculated
        pos_velocity = pos_direction * speed
        
        # ori_velocity = self.calculate_orientation_velocity(nut_ori, target_grasp_ori)
        # ori_velocity = np.zeros(3)
        # ori_velocity = calculate_orientation_velocity(current_ori, gripper_grasp_frame_pose_ori_quat, gripper_grasp_frame_pose02_ori_quat)

        # The gripper command is set to open (assuming -1 is open, 1 is close)
        gripper_command = np.array([-1.0])  # Adjust based on your robot's gripper control specifics
        
        # # Check if the end effector is above the nut to transition to the next state
        if self.is_position_aligned(current_pos, goal_pos) and np.linalg.norm(pos_delta[1]) < 0.005:
            self.current_state += 1  # Advance to the next state
        

        # The action is a concatenation of position velocity, orientation velocity, and gripper command
        action = np.concatenate([pos_velocity, ori_velocity, gripper_command])
        return action

    # policy for state 3: GRASP_NUT
    def handle_grasp_frame(self, obs, env):
        # No movement, just close the gripper
        pos_velocity = np.array([0, 0, 0])
        ori_velocity = np.array([0, 0, 0])

        # gripper_q_state = 2 * obs['robot0_gripper_qpos'][0]  # 2 * 0.04 for open; 2 * 0.015 when close
        current_pos = obs['robot0_eef_pos']
        # current_ori = T.mat2quat(obs[6:15].reshape(3, 3))
        current_ori = obs['robot0_eef_quat']
        can_pose = obs['Can_pos']
        can_quat = obs['Can_quat']
        goal_pos = can_pose + np.array([0, 0, +0.015])
        
        # print("objects: ", env.env.objects)
        can_obj = None
        for i, obj in enumerate(env.env.objects):
            if env.env.objects_in_bins[i]:
                continue
            can_obj = obj
        gripper_q_state = 2 * obs['robot0_gripper_qpos'][0]
        grasped = env.env._check_grasp(gripper=env.env.robots[0].gripper, object_geoms=can_obj.contact_geoms)
        # print("grasped: ", grasped)
        # Close the gripper
        gripper_command = 1.0

        # if self.is_gripper_closed(gripper_state) and self.is_nut_grasped(nut_state):
        #     self.current_state += 1
        logger.debug('%s %s', "gripper_q_state: ", gripper_q_state)
        if gripper_q_state < 0.062 and gripper_q_state > 0.02 and grasped == 1.0:
            self.current_state += 1
        elif gripper_q_state < 0.02:
            self.current_state -= 1
        elif grasped != 1.0 and abs(current_pos[2] - goal_pos[2]) > 0.05:
            self.current_state -= 1

        action = np.concatenate([pos_velocity, ori_velocity, [gripper_command]])
        return action

    # policy for state 4: LIFT_NUT
    def handle_lift_frame(self, obs, env):
        current_pos = obs['robot0_eef_pos']
        # current_ori = T.mat2quat(obs[6:15].reshape(3, 3))
        current_ori = obs['robot0_eef_quat']

        can_pose = obs['Can_pos']
        can_quat = obs['Can_quat']

        can_obj = None
        for i, obj in enumerate(env.env.objects):
            if env.env.objects_in_bins[i]:
                continue
            can_obj = obj
        gripper_q_state = 2 * obs['robot0_gripper_qpos'][0]
        grasped = env.env._check_grasp(gripper=env.env.robots[0].gripper, object_geoms=can_obj.contact_geoms)
        # print("grasped: ", grasped)
        # print("current_ori: ", current_ori)
        # print('nut_ori: ', nut_ori)


        # target_grasp_ori = quaternion_multiply(nut_ori, self.grasp_orientation_transform)
        # pos_handle_to_world, target_grasp_ori, target_grasp_ori02 = self.obtain_gripper_goal_for_moving_to_NutHanlde(obs)


        # Lift the nut upwards by a small amount
        pos_velocity = np.array([0, 0, 0.4])  # Adjust the speed and direction as needed

        y = np.array([1, 0, 0])
        # y = np.array([0.5, 0.5, 0])
        z = np.array([0, 0, -1])
        # Calculate x as the cross product of y and z
        x = np.cross(y, z)
        # Normalize x to make it a unit vector
        x = x / np.linalg.norm(x)
        # Construct the orientation matrix
        gripper_grasp_frame_pose_ori_matrix = np.column_stack((x, y, z))
        gripper_grasp_frame_pose_ori_matrix = gripper_grasp_frame_pose_ori_matrix.dot(self.grasp_orientation_transform_mat)
        gripper_grasp_frame_pose_ori_quat = T.mat2quat(gripper_grasp_frame_pose_ori_matrix)
        ori_velocity = calculate_orientation_velocity(current_ori, gripper_grasp_frame_pose_ori_quat, gripper_grasp_frame_pose_ori_quat)

        gripper_command = 1.0  # Gripper remains closed

        if can_pose[2] > 0.97 and grasped == 1:
        # if frame_hook_pos[2] > 1.0 and grasped == 1:
            self.current_state += 1
        elif grasped != 1:
            self.current_state = 0
        action = np.concatenate([pos_velocity, ori_velocity, [gripper_command]])
        return action
    
    # policy for state 5: MOVE_TO_PEG
    def handle_frame_move_to_base(self, obs, env):
        return self.handle_frame_move_to_base_with_height(obs, env, np.array([0,0, 0.05]), velocity_threshold=0.04)
    
    # policy for state 6: ALIGN_OVER_PEG 
    def handle_frame_move_to_base_lower(self, obs, env):
        action = self.handle_frame_move_to_base_with_height(obs, env, np.array([0,0, -0.05]), velocity_threshold=0.02)
        return action 

    def obtain_rotation_matrix_hand_wrt_standardGraspPose(self, obs, env):

        # rotation matrix of frame w.r.t world
        current_ori = obs['robot0_eef_quat']
        grasped = env.env._check_grasp(gripper=env.env.robots[0].gripper, object_geoms=env.env.frame.contact_geoms)
        # print("grasped: ", grasped)

    
        hook_endpoint = env.env.sim.data.site_xpos[env.env.obj_site_id["frame_mount_site"]]
        frame_hook_vec = env.env.sim.data.site_xpos[env.env.obj_site_id["frame_intersection_site"]] - hook_endpoint
        frame_hook_length = np.linalg.norm(frame_hook_vec)
        frame_hook_vec = -frame_hook_vec / frame_hook_length
        
        frame_tip_pos = env.env.sim.data.site_xpos[env.env.obj_site_id["frame_tip_site"]] 
        frame_hook_endpoint_pos = env.env.sim.data.site_xpos[env.env.obj_site_id["frame_hang_site"]] 
        frame_hang_vec =frame_hook_endpoint_pos - env.env.sim.data.site_xpos[env.env.obj_site_id["frame_intersection_site"]]
        frame_hang_length = np.linalg.norm(frame_hang_vec)
        frame_hang_vec = -frame_hang_vec / frame_hang_length

        x = frame_hang_vec
        y = frame_hook_vec
    

        # y = frame_hook_vec
        # Calculate x as the cross product of y and z
        z = np.cross(x, y)
        # Normalize x to make it a unit vector
        z = z / np.linalg.norm(z)
        frame_pose_ori_matrix_wrtWorld = np.column_stack((x, y, z))
        # frame_pose_ori_matrix_wrtWorld = frame_pose_ori_matrix_wrtWorld.dot(self.grasp_orientation_transform_mat)


        gripper_pose_ori_matrix_wrtWorld = T.quat2mat(current_ori)
        # gripper_pose_ori_matrix_wrtWorld = np.linalg.inv(self.grasp_orientation_transform_mat).dot(gripper_pose_ori_matrix_wrtWorld)
        Rotation_object_wrt_gripper = np.linalg.inv(frame_pose_ori_matrix_wrtWorld).dot(gripper_pose_ori_matrix_wrtWorld)
        # print("Rotation_object_wrt_gripper: ", Rotation_object_wrt_gripper)
        # rotation matrix of hand w.r.t world
        return Rotation_object_wrt_gripper

    def handle_frame_move_to_base_with_height(self, obs, env, height, velocity_threshold = 0.03):
        # current_ori = T.mat2quat(obs[6:15].reshape(3, 3))
        current_pos = obs['robot0_eef_pos']
        current_ori = obs['robot0_eef_quat']
        can_pose = obs['Can_pos']
        can_quat = obs['Can_quat']

        can_obj = None
        for i, obj in enumerate(env.env.objects):
            if env.env.objects_in_bins[i]:
                continue
            can_obj = obj
        gripper_q_state = 2 * obs['robot0_gripper_qpos'][0]
        grasped = env.env._check_grasp(gripper=env.env.robots[0].gripper, object_geoms=can_obj.contact_geoms)

        bin_pose = np.array([0.145, 0.36, 0.95]) + height
        pos_delta = self.calculate_position_delta(can_pose, bin_pose)
        # pos_delta = np.zeros(3)
        pos_direction = pos_delta / np.linalg.norm(pos_delta) if np.linalg.norm(pos_delta) > 0.01 else pos_delta / (np.linalg.norm(pos_delta)+0.01)
        speed = 0.45
        # speed = 0.15 
        pos_velocity = pos_direction * speed
        # print("frame_tip_pos: ", frame_tip_pos, " base_walls_geom_positions_average: ", base_walls_geom_positions_average)

        y = np.array([1, 0, 0])
        # y = np.array([0.5, 0.5, 0])
        z = np.array([0, 0, -1])
        # Calculate x as the cross product of y and z
        x = np.cross(y, z)
        # Normalize x to make it a unit vector
        x = x / np.linalg.norm(x)
        # Construct the orientation matrix
        gripper_grasp_frame_pose_ori_matrix = np.column_stack((x, y, z))
        gripper_grasp_frame_pose_ori_matrix = gripper_grasp_frame_pose_ori_matrix.dot(self.grasp_orientation_transform_mat)
        gripper_grasp_frame_pose_ori_quat = T.mat2quat(gripper_grasp_frame_pose_ori_matrix)
        ori_velocity = calculate_orientation_velocity(current_ori, gripper_grasp_frame_pose_ori_quat, gripper_grasp_frame_pose_ori_quat)

        gripper_command = 1.0  # Keep the gripper closed

        # check the difference between [0, 0, 1] and frame_hook_vec, which is the dot product between this two vectors
        # print("np.abs(frame_hook_vec[2]): ", np.abs(frame_hook_vec[2]))

        
        # print("contacts_frame_tool: ", contacts_frame_tool)
        # if np.linalg.norm(pos_delta[0:2]) > 0.01 and np.linalg.norm(ori_velocity) > velocity_threshold:
        #     pos_velocity[-1] = 0
        if np.linalg.norm(pos_delta) < 0.03:
            self.current_state += 1
        elif grasped != 1:
            self.current_state = 0  # MOVE_TOWARDS_TOOL

        

        action = np.concatenate([pos_velocity, ori_velocity, [gripper_command]])
        return action


    # policy for state 8
    def release_frame(self, obs, env):
        # (1) move away from the frame a little bit

        current_pos = obs['robot0_eef_pos']
        # current_ori = T.mat2quat(obs[6:15].reshape(3, 3))
        current_ori = obs['robot0_eef_quat']
        gripper_q_state = 2 * obs['robot0_gripper_qpos'][0]

        
        pos_direction =np.zeros(3)
        
        # Assume a fixed speed towards the nut; adjust as necessary
        speed = 0.35  # This speed value might need tuning
        
        # Position velocity: Moving towards the nut at a constant speed in the direction calculated
        pos_velocity = pos_direction * speed
        # Close the gripper
        gripper_command = -1.0

        ori_velocity = np.zeros(3)
        
        

        action = np.concatenate([pos_velocity, ori_velocity, [gripper_command]])
        return action 

    def move_to_safe_region(self, obs):
        # (1) move away from the frame a little bit

        current_pos = obs['robot0_eef_pos']
        # current_ori = T.mat2quat(obs[6:15].reshape(3, 3))
        current_ori = obs['robot0_eef_quat']
        goal_pose = np.array([0.10808056,  -0.2,  1.00744903])
        pos_delta = self.calculate_position_delta(current_pos, goal_pose)
        # print("pos_delta: ", pos_delta)
        
        # Normalize delta to obtain a direction vector
        pos_direction = pos_delta / np.linalg.norm(pos_delta) if np.linalg.norm(pos_delta) > 0.01 else pos_delta
        
        # Assume a fixed speed towards the nut; adjust as necessary
        speed = 0.35  # This speed value might need tuning
        
        # Position velocity: Moving towards the nut at a constant speed in the direction calculated
        pos_velocity = pos_direction * speed
        ori_velocity = np.zeros(3)
        # Close the gripper
        gripper_command = -1.0
        
        if np.linalg.norm(pos_delta) < 0.01 and np.linalg.norm(ori_velocity) < 0.01:
            self.current_state += 1

        action = np.concatenate([pos_velocity, ori_velocity, [gripper_command]])
        return action 

    def is_above_target(self, current_pos, target_pos, threshold=0.01):
        """Check if the current position is above the target within a certain threshold."""
        return np.linalg.norm(current_pos[:2] - target_pos[:2]) < threshold and abs(current_pos[2] - target_pos[2]) > threshold

    def is_position_aligned(self, current_pos, target_pos, threshold=0.012):
        # Check if the current position is within the threshold distance of the target position.
        distance = np.linalg.norm(current_pos - target_pos)
        return distance < threshold

    def execute_action(self, action):
        """Execute the given action."""
        # Placeholder for executing an action, such as moving the end effector or operating the gripper
        pass
