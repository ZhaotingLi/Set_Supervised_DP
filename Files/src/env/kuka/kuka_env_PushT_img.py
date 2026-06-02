import logging

logger = logging.getLogger(__name__)


from sensor_msgs.msg import JointState
from env.kuka.inverse_kinematics import inverse_kinematics_init
from spatialmath import SO3
import cor_tud_msgs.msg as cor_msg
import sensor_msgs.msg as sensor_msg
import numpy as np
import rospy
import sys
from geometry_msgs.msg import PoseStamped
from env.kuka.iiwa_robotics_toolbox import iiwa
import math

from env.realsense_Image_receiver import ImageReceiver

from env.robotsuite.env_robosuite import combine_obs_dicts

ROBOT = 'iiwa14'


class KUKAenv_pushT_img:
    def __init__(self, config):

        ## load shape meta
        shape_meta = config.shape_meta
        self.abs_action = config.use_abs_action

        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys


        # self.joint_impedance_controller = JointImpedanceController(robot_name=ROBOT, alpha=0.5)
        self.robot = iiwa(model=ROBOT)
        # self.cartesian_impedance_controller = CartesianImpedanceController(robot_name=ROBOT, robot=self.robot)
        self.IK = None
        
        # ROS
        # rospy.Subscriber('/spacenav/joy', sensor_msg.Joy, self._callback_spacenav, queue_size=10)
        rospy.Subscriber('/%s/end_effector_state' % ROBOT, cor_msg.CartesianState, self._callback_end_effector_state, queue_size=10)
        rospy.Subscriber('/%s/joint_states' % ROBOT, JointState, self._callback_joint_states, queue_size=10)
        self.request_pub = rospy.Publisher('/%s/control_request' % ROBOT, cor_msg.ControlRequest, queue_size=10)
        # read box pose
        # rospy.Subscriber("/vrpn_client_node/box/pose", PoseStamped, self._callback_object_pose, queue_size=10)


        self.receiver_img1 = ImageReceiver('/camera1/color/image_raw/compressed')
        self.receiver_img2 = ImageReceiver('/camera2/color/image_raw/compressed', enable_crop=True)

        # Init variables
        self.q = None
        self.ee_pose = None
        self.ee_velocity = None
        self.spacenav_state = None
        
        # Parameters
        self.controller = 'ik'  # options: cartesian_impedance, ik
        self.ik_type = 'ranged ik'  # options: track ik, ranged ik

        # self.control_orientation = True
        self.control_orientation = False
        # self.scale = 0.05 * 0.15
        self.scale = 0.05  * 0.5
        self.delta_q_max = np.array([0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08])  # maximum requested delta for safety

        # If self.control_orientation = False
        
        self.control_position_z = False # do not control the pose_z via space mouse
        self.poition_goal_z = None
        # self.orientation_goal = np.array([0, 0, 0])
        self.delta_t_orientation = 1  # can only be 1, you cannot scale the transform matrix directly
        self.gain_orientation = 0.1

        self.ee_goal_position = None
        # self.orientation_goal = np.array([-3.1366246985664645, 0.013679215381810106, 1.1278594985427546])
        # self.ee_pos_limit_xy = [0.3, 0.7, -0.4, 0.4]   # xlow xup ylow yup zlow zup

        # pose of object
        self.Tobject_pose = None
        self.Uobject_pose = None
        self.use_simulation = False  # if use simulation, set the object pose as zeros
        self.obs_dict_past = None

        self.count = 0
    
        # original state when reset the robot
        self.rest_controller_gain = 0.35

        # one nice joint config [-1.0021906347288463, 1.253672343939383, 0.6721988301722307, -1.554135497858412, -1.0877502828244043, 0.7306915548498796, 1.8073318499420343]
        self.q_goal_reset = np.array([0.3236136803194445, 1.147745278035441, -0.35717580180277175, -1.3570434144236874, 0.5105804238743348, 0.7109607269929713, 0.6005146495719441])
        # the intermidate one is set to keep safe, keep the robot from colliding into the table
        self.q_goal_middle_reset = np.array([0.3236136803194445, 1.047745278035441, -0.35717580180277175, -1.3570434144236874, 0.5105804238743348, 0.7109607269929713, 0.6005146495719441])

        # task specific parameters 
        self.orientation_goal = np.array([3.1277035959303134, 0.00882190027433434, 2.31675814335778])
        self.ee_pos_limit_xyz =  [0.2, 0.7, -0.4, 0.4, 0.142, 0.165]
        self.action_min = [0.2, -0.4]
        self.action_max = [0.7, 0.4]
        self.init_IK()

    ## get the observation dict
    def _get_obs_dict(self):
            
        obs_dict = dict()
        for key in self.rgb_keys:
            if key in ['image1']:
                image = self.receiver_img1.image
            if key in ['image2']:
                image = self.receiver_img2.image
            obs_dict[key] = np.moveaxis(image, -1,0).astype(np.float32) / 255.   # (C, H, W)
            obs_dict[key] = (2.0 * obs_dict[key] - 1.0).astype(np.float32)
        for key in self.lowdim_keys:
            if key in ['robot0_eef_pos_vel']:
                obs_dict[key] = self._print_end_effector_pose_velocity().astype(np.float32)
                # normalize the pose (2D)
                obs_dict[key][:2] = self.normalize_abs_action(obs_dict[key][:2], max_list=self.action_max, min_list=self.action_min)

            #TODO normalize eef_pose and unnormalize the output of the NN
    
            # if key.endswith('_pos') and self.abs_action:
            #     # print('pos : ', obs_dict[key] )
            #     if key == 'robot0_eef_pos':
            #         obs_dict[key] = self.normalize_eef_pos(obs_dict[key], max_list=self.eef_pos_max_list[0], min_list=self.eef_pos_min_list[0])
            #     if key == 'robot1_eef_pos':
            #         obs_dict[key] = self.normalize_eef_pos(obs_dict[key], max_list=self.eef_pos_max_list[1], min_list=self.eef_pos_min_list[1])
            #     # print('normalized obs_dict[key] : ', obs_dict[key] )
            # if key.endswith('qpos'):
            #     obs_dict[key] = self.normalize_gripper_qpose(obs_dict[key])

        
        if self.obs_dict_past is None:
            self.obs_dict_past = obs_dict
        
        obs_dict_combined = combine_obs_dicts(self.obs_dict_past, obs_dict) 
        # obs_dict_combined[self.rgb_keys] -> (2, C, H, W)
        # obs_dict_combined[self.lowdim_keys] -> (2, lowdim_keys)

        self.obs_dict_past = obs_dict

        return obs_dict_combined



    def _callback_end_effector_state(self, data):
        self.ee_pose = np.array(data.pose.data)
        self.ee_velocity = np.array(data.velocity.data)   


    def normalize_abs_action(self, action_, min_list, max_list):
        # arr = np.asarray(action, dtype=float)
        action = action_.copy()
        lo  = np.asarray(min_list, dtype=np.float32)
        hi  = np.asarray(max_list, dtype=np.float32)
        # centers and half-ranges
        center = (hi + lo) / 2.0
        half_range = (hi - lo) / 2.0
        # avoid division by zero
        half_range[half_range == 0] = 1.0
        action = ((action - center) / half_range)
        return action
    
    def unnormalize_abs_action(self, norm_action_, min_list, max_list):
        norm_action = norm_action_.copy()
        arr = np.asarray(norm_action, dtype=np.float32)
        lo  = np.asarray(min_list, dtype=np.float32)
        hi  = np.asarray(max_list, dtype=np.float32)
        # compute half-range and center
        half_range = (hi - lo) / 2.0
        center     = (hi + lo) / 2.0
        # avoid division-by-zero artifacts (if hi==lo, half_range==0)
        # but here we only need half_range for multiplication, so it’s fine:
        arr = (arr * half_range + center)
        return arr

    def _print_end_effector_pose_velocity(self):
        if self.q is None:
            logger.debug("Not receving q! Waiting...")
            logger.debug('No ee_pose data received yet.')
            return np.zeros(14)  # Return default position and orientation

        # ee_pose_transfromed =  self.ee_pose   # !!!!! Wrong code, should never do this, will yield some strange values!!!
        ee_pose_transfromed = np.zeros(2)
        ee_pose_transfromed[0] =  self.ee_pose[0] #+ 0.15
        ee_pose_transfromed[1] =  self.ee_pose[1] #+ 0.15

        return np.concatenate((ee_pose_transfromed, self.ee_velocity[0:2]), axis=0)
        
    def _callback_joint_states(self, data):
        self.q = np.array(data.position)


    def send_position_request(self, q_d):
        q_dot_d = [0, 0, 0, 0, 0, 0, 0]
        msg = cor_msg.ControlRequest()
        msg.header.stamp = rospy.Time.now()
        msg.q_d.data = q_d
        msg.q_dot_d.data = q_dot_d
        msg.control_type = 'joint impedance'
        self.request_pub.publish(msg)
        return True
    
    def render(self):
        return None
    
   
    def step(self, action):
        action_r = action.copy()
        if self.abs_action:
            action_r = self.unnormalize_abs_action(action_r, max_list=self.action_max, min_list=self.action_min)
        logger.debug('%s %s', "action in step: ", action_r)

        # this way of transfering abs_action to delta_action can cause some drifting, donot use.
        # if self.abs_action:
        #     ee_position, ee_orientation = self.ee_pose[:3], SO3.RPY(self.ee_pose[3:], order='zyx')
        #     # transform to delta_action
        #     if not self.control_position_z:
        #         ee_delta_position = (action_r - ee_position[:2])/ self.scale
        #     else:
        #         ee_delta_position = (action_r - ee_position)/ self.scale
        #     # add limits here to keep safe
        #     ee_delta_position = np.clip(ee_delta_position, -1 ,  1)
        
        #     action_r = ee_delta_position
        #     print("abs_action")

        # Run control
        # check the dimension of the action
        if self.ik_type == "track ik":
            # Start IK with current q
            self.init_IK()
        
        self.run(action_r)        # run the controller

        reward = 0
        self.count = self.count + 1
        # done = True if self.count > 20000 else False
        done = False   # not done unless human press the button
        terminated = False 

        info = {}

        # check the state is available
        
        # robot_state = self._print_end_effector_pose_velocity()  # 4, 1

        # state = np.concatenate((robot_state, objects_state), axis = 0)   # dim 48

        info['success'] = False

        # control the robot at the end of each epsiode, this is very very important!!! 
        # otherwise, the robot will execute the last action when the algorithm is still traninnig without controlling the robot directly!
        if done or terminated or info['success']:
            # stay at the current state
            self.hold_on_mode()
        
        obs_dict = self._get_obs_dict()
        
        # expanded_obs = np.expand_dims(state, axis=0)
        return [obs_dict, reward, done, terminated, info]
    
    ## hold on mode for the robot, let the robot stay still and wait for human input (in the main-file, through keyboards, buttons, etc.)
    def hold_on_mode(self):
        control_frequency = 500
        rate = rospy.Rate(control_frequency)
        last_current_q = self.q
        for i in range(500):
            # self.joint_impedance_controller.send_torque_request(last_current_q)
            self.send_position_request(last_current_q)
            rate.sleep()
        return 

    def reset(self):
        self.obs_dict_past = None
        self.count = 0
        control_frequency = 500
        rate = rospy.Rate(control_frequency)

        reached_middle_goal = False
        while not reached_middle_goal:
            q_delta = self.rest_controller_gain * (self.q_goal_middle_reset - self.q)
            q_d = self.q + q_delta
            # reached_middle_goal = False if np.linalg.norm(self.q_goal_middle_reset - self.q) > 0.3 else True
            reached_middle_goal = False if np.linalg.norm(self.q_goal_middle_reset - self.q) > 0.2 else True
            # self.joint_impedance_controller.send_torque_request(q_d)
            self.send_position_request(q_d)
            rate.sleep()

        while(np.linalg.norm(self.q_goal_reset - self.q) > 0.13): # set [1:] because we want the 0-joint to be quite accurate
        # while(np.linalg.norm(self.q_goal_reset - self.q) > 0.035):
            # print("np.linalg.norm(self.q_goal_reset - self.q) ", np.linalg.norm(self.q_goal_reset - self.q) )
            # if self.q is None:
            #     print("self.q is None")
            #     continue
            q_delta = self.rest_controller_gain * (self.q_goal_reset - self.q)
            q_d = self.q + q_delta
            # self.joint_impedance_controller.send_torque_request(q_d)
            self.send_position_request(q_d)
            rate.sleep()


        robot_state = self._print_end_effector_pose_velocity()  # 12, 1

        # state = np.concatenate((robot_state, objects_state), axis = 0)   # dim 9
        # state = np.concatenate((ee_pose_relative, self.ee_velocity, self.object_pose_relative), axis = 0)   # dim 18
        # state = np.concatenate((self.ee_pose, self.object_pose, self.goal_object_position), axis = 0) # dim 14
        # state = np.concatenate((self.ee_pose, self.object_pose), axis = 0)
        # print("state: ", state)

        self.ee_goal_position = None

        info = {}
        # expanded_obs = np.expand_dims(state, axis=0)
        obs_dict = self._get_obs_dict()
        return [obs_dict, info]



    def init_IK(self):
        self.IK = inverse_kinematics_init(self.ik_type, self.q, ROBOT)

    def _get_pose(self, joint_q, end_link='iiwa_link_7'):
        pose = self.robot.fkine(joint_q, end=end_link, start='iiwa_link_0')
        position = pose.t
        orientation = pose.R
        return position, orientation

    ## send control command to the robot
    def run(self, ee_action):
        # Get ee pose
        ee_position, ee_orientation = self.ee_pose[:3], SO3.RPY(self.ee_pose[3:], order='zyx')

        if not self.abs_action:
            # Get delta x, for kuka_env, the ee_delta_position is the action
            # ee_delta_position = np.array(self.spacenav_state)[:3]

            # constrain the input, for the box-pushing task, we first consider the easiest case, where the ee moves in a plane
            # in the future, this should be modified to a safety region
            # [1] set translation_z = 0
            ee_delta_position = np.zeros(3)
            ee_delta_position[0:2] = ee_action
            if not self.control_position_z:
                ee_position_z = ee_position[2].copy()
                if self.poition_goal_z is None:
                    self.poition_goal_z = ee_position_z
                # to do, check the value
                self.poition_goal_z = 0.125
                ee_delta_position[2] = (self.poition_goal_z - ee_position_z) / self.scale
            
            # print("ee_position: ", ee_position, " self.poition_goal_z: ", self.poition_goal_z, " ee_delta_position: ", ee_delta_position )


            # Get desired position
            # ee_goal_position = ee_position + ee_delta_position * self.scale
            # print("ee_goal_position: ", ee_goal_position)

            # only update the desired position when we receive the space mouse feedback
            if self.ee_goal_position is None:
                self.ee_goal_position = ee_position
            # if np.linalg.norm(np.array(self.spacenav_state)[:3]) > 0.01:
            if np.linalg.norm(np.array(ee_delta_position[:3])) > 0.01:
                self.ee_goal_position =  ee_position + ee_delta_position * self.scale
                # print("self.ee_goal_position: ", self.ee_goal_position)
        else:
            self.ee_goal_position = ee_position.copy()
            if not self.control_position_z:
                self.ee_goal_position[:2] = ee_action
                self.ee_goal_position[2] = 0.125
            else:
                self.ee_goal_position[:3] = ee_action
            
            # limit EEF displacement if we have a past observation in abs_action mode
            if self.obs_dict_past is not None:
                key = 'robot0_eef_pos_vel'
                # past_pos = self.unnormalize_abs_action(self.obs_dict_past[key], action_max=self.eef_pos_max_list[i], action_min=self.eef_pos_min_list[i])
                # past_pos = self.obs_dict_past[key][:2]
                past_pos = self.unnormalize_abs_action(self.obs_dict_past[key][:2], max_list=self.action_max, min_list=self.action_min)
                desired = self.ee_goal_position[:2]
                delta   = desired - past_pos
                dist    = np.linalg.norm(delta)
                max_step = 0.02
                if dist > max_step:
                    delta = delta / dist * max_step
                    logger.debug(f"robot exceeded max_step")
                self.ee_goal_position[:2] = past_pos + delta

            
        
        # check whether the goal violate the pos constraint
        # x lower bound
        self.ee_goal_position[0] = self.ee_pos_limit_xyz[0] if self.ee_goal_position[0] < self.ee_pos_limit_xyz[0] else self.ee_goal_position[0]
        # x higher bound
        self.ee_goal_position[0] = self.ee_pos_limit_xyz[1] if self.ee_goal_position[0] > self.ee_pos_limit_xyz[1] else self.ee_goal_position[0]
        # y lower bound
        self.ee_goal_position[1] = self.ee_pos_limit_xyz[2] if self.ee_goal_position[1] < self.ee_pos_limit_xyz[2] else self.ee_goal_position[1]
        # y higher bound
        self.ee_goal_position[1] = self.ee_pos_limit_xyz[3] if self.ee_goal_position[1] > self.ee_pos_limit_xyz[3] else self.ee_goal_position[1]
        
        # z lower bound 
        self.ee_goal_position[2] = self.ee_pos_limit_xyz[4] if self.ee_goal_position[2] < self.ee_pos_limit_xyz[4] else self.ee_goal_position[2]
        # z higher bound
        self.ee_goal_position[2] = self.ee_pos_limit_xyz[5] if self.ee_goal_position[2] > self.ee_pos_limit_xyz[5] else self.ee_goal_position[2]

        # print("self.ee_goal_position: ", self.ee_goal_position, " ee_pos_limit_xyz: ", self.ee_pos_limit_xyz)

        # check whether current robot is near the constrainted region, if yes, stay still and show errors
        tolenance_pos_constraint = 0.05 
        ee_violated_pos_constraint = (self.ee_goal_position[0] < self.ee_pos_limit_xyz[0] -tolenance_pos_constraint) and \
                                        (self.ee_goal_position[0] > self.ee_pos_limit_xyz[1] +tolenance_pos_constraint) and \
                                        (self.ee_goal_position[1] < self.ee_pos_limit_xyz[2] -tolenance_pos_constraint) and \
                                        (self.ee_goal_position[1] > self.ee_pos_limit_xyz[3] +tolenance_pos_constraint) and \
                                        (self.ee_goal_position[2] < self.ee_pos_limit_xyz[4] -tolenance_pos_constraint) and \
                                        (self.ee_goal_position[2] > self.ee_pos_limit_xyz[5] +tolenance_pos_constraint)
        if ee_violated_pos_constraint:
            logger.debug('%s %s', "ee_violated_pos_constraint: ", ee_violated_pos_constraint)
            self.ee_goal_position = ee_position

        if self.control_orientation:
            ee_delta_orientation = np.array(self.spacenav_state)[3:] * self.scale
            ee_delta_orientation = SO3.RPY(ee_delta_orientation, order='zyx')
            # Get desired orientation
            ee_goal_orientation = ee_delta_orientation @ ee_orientation
        else:
            # Get desired orientation
            ee_goal_orientation = SO3.RPY(self.orientation_goal, unit='rad', order='zyx')

        if self.controller == 'cartesian_impedance':
            # Send command to controller
            self.cartesian_impedance_controller.control_law(self.ee_goal_position, ee_goal_orientation)

        elif self.controller == 'ik':
            # Compute inverse kinematics
            q_d = self.IK.compute(self.ee_goal_position, ee_goal_orientation.rpy(), self.q)

            position_solved, orientation_solved = self._get_pose(joint_q= q_d)
            # print("np.linalg.norm(self.ee_goal_position - position_solved): ", np.linalg.norm(self.ee_goal_position - position_solved))
            if np.linalg.norm(self.ee_goal_position - position_solved) > 0.01:
            # if np.linalg.norm(self.ee_goal_position - position_solved) > 0.02:
                logger.debug('%s %s %s %s', "position_solved: ", position_solved, " self.ee_goal_position: ", self.ee_goal_position)
                q_d = 0.5 * (self.q + q_d)

            # Compute delta in joint space and store
            delta_q = q_d - self.q

            # Clip delta
            delta_q_clipped = np.clip(delta_q, a_min=-self.delta_q_max, a_max=self.delta_q_max)

            # Compute desired joint clipped
            q_d_clipped = self.q + delta_q_clipped

            # Send command to controller
            # self.joint_impedance_controller.send_torque_request(q_d_clipped)
            self.send_position_request(q_d_clipped)
        else:
            raise ValueError('Selected controller not valid, options: cartesian_impedance, ik.')

        return True








