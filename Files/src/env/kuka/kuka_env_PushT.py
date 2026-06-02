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

ROBOT = 'iiwa14'


class KUKAenv_pushT:
    def __init__(self):
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
        
        rospy.Subscriber('/vrpn_client_node/TObject/pose', PoseStamped, self._callback_TObject_pose, queue_size=10)

        rospy.Subscriber('/vrpn_client_node/UObject/pose', PoseStamped, self._callback_UObject_pose, queue_size=10)

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

        self.count = 0
    
        # original state when reset the robot
        self.rest_controller_gain = 0.35

        # one nice joint config [-1.0021906347288463, 1.253672343939383, 0.6721988301722307, -1.554135497858412, -1.0877502828244043, 0.7306915548498796, 1.8073318499420343]
        self.q_goal_reset = np.array([0.3236136803194445, 1.147745278035441, -0.35717580180277175, -1.3570434144236874, 0.5105804238743348, 0.7109607269929713, 0.6005146495719441])
        # the intermidate one is set to keep safe, keep the robot from colliding into the table
        self.q_goal_middle_reset = np.array([0.3236136803194445, 1.047745278035441, -0.35717580180277175, -1.3570434144236874, 0.5105804238743348, 0.7109607269929713, 0.6005146495719441])

        # task specific parameters 
        self.orientation_goal = np.array([3.1277035959303134, 0.00882190027433434, 2.31675814335778])
        self.ee_pos_limit_xyz =  [0.3, 0.7, -0.4, 0.2, 0.142, 0.165]
        self.init_IK()

    def _callback_end_effector_state(self, data):
        self.ee_pose = np.array(data.pose.data)
        self.ee_velocity = np.array(data.velocity.data)   


    def _print_end_effector_pose_velocity(self):
        if self.q is None:
            logger.debug("Not receving q! Waiting...")
            logger.debug('No ee_pose data received yet.')
            return np.zeros(14)  # Return default position and orientation

        # ee_pose_transfromed =  self.ee_pose   # !!!!! Wrong code, should never do this, will yield some strange values!!!
        ee_pose_transfromed = np.zeros(2)
        ee_pose_transfromed[0] =  self.ee_pose[1] + 0.15
        ee_pose_transfromed[1] = - self.ee_pose[0] + 0.15

        return np.concatenate((ee_pose_transfromed, self.ee_velocity[0:2]), axis=0)
        
    def _callback_joint_states(self, data):
        self.q = np.array(data.position)

    def _callback_TObject_pose(self, data):
        self.Tobject_pose = data

    def _callback_UObject_pose(self, data):
        self.Uobject_pose = data

    def _print_Tobject_pose(self):
        if self.Tobject_pose is None:
            logger.debug('No Tobject_pose data received yet.')
            return [0, 0, 0], [0, 0, 0, 0]  # Return default position and orientation

        position = self.Tobject_pose.pose.position
        orientation = self.Tobject_pose.pose.orientation
        return [position.x, position.y, position.z], [orientation.x, orientation.y, orientation.z, orientation.w] 

    def _print_Uobject_pose(self):
        if self.Uobject_pose is None:
            logger.debug('No Uobject_pose data received yet.')
            return [0, 0, 0], [0, 0, 0, 0]  # Return default position and orientation

        position = self.Uobject_pose.pose.position
        orientation = self.Uobject_pose.pose.orientation
        return [position.x, position.y, position.z], [orientation.x, orientation.y, orientation.z, orientation.w] 

    def obtain_keypoints_for_T_and_U_objects(self):
        # T_object_positions_local = {
        #     'Object_local1_p_w_to_o': [0, 0, 0],
        #     'Object_local1_p_w_1': [-0.015, 0.13, 0],   # meter
        #     'Object_local1_p_w_2': [0.015, 0.13, 0],
        #     'Object_local1_p_w_3': [-0.015, 0.03, 0],
        #     'Object_local1_p_w_4': [0.015, 0.03, 0],
        #     'Object_local1_p_w_5': [0.055, 0.03, 0],
        #     'Object_local1_p_w_6': [0.055, 0.0, 0],
        #     'Object_local1_p_w_7': [-0.055, 0.03, 0],
        #     'Object_local1_p_w_8': [-0.055, 0.0, 0],
        #     'Object_local1_p_w_9': [-0.015, 0.1, 0],
        #     'Object_local1_p_w_10': [0.015, 0.1, 0],
        #     'Object_local1_p_w_11': [0.0, 0.13, 0],
        # }

        # # U_object_positions_local = {
        # #     'Object_local1_p_w_to_o': [0, 0, 0],
        # #     'Object_local1_p_w_1': [-0.039, 0.105, 0],
        # #     'Object_local1_p_w_2': [-0.019, 0.105, 0],
        # #     'Object_local1_p_w_3': [0.019, 0.105, 0],
        # #     'Object_local1_p_w_4': [0.039, 0.105, 0],
        # #     'Object_local1_p_w_5': [-0.019, 0.02, 0],
        # #     'Object_local1_p_w_6': [0.019, 0.02, 0],
        # #     'Object_local1_p_w_7': [-0.039, 0.0, 0],
        # #     'Object_local1_p_w_8': [0.039, 0.0, 0],
        # #     'Object_local1_p_w_9': [-0.019, 0.05, 0],
        # #     'Object_local1_p_w_10': [0.019, 0.05, 0],
        # #     'Object_local1_p_w_11': [0.0, 0.02, 0],
        # # }

        # U_object_positions_local = {
        #     'Object_local1_p_w_to_o': [0, 0, 0],
        #     'Object_local1_p_w_5': [-0.019, 0.02, 0],
        #     'Object_local1_p_w_6': [0.019, 0.02, 0],
        #     'Object_local1_p_w_2': [-0.019, 0.105, 0],
        #     'Object_local1_p_w_3': [0.019, 0.105, 0],
        #     'Object_local1_p_w_4': [0.039, 0.105, 0],
        #     'Object_local1_p_w_8': [0.039, 0.0, 0],
        #     'Object_local1_p_w_1': [-0.039, 0.105, 0],
        #     'Object_local1_p_w_7': [-0.039, 0.0, 0],
        #     'Object_local1_p_w_9': [-0.019, 0.05, 0],
        #     'Object_local1_p_w_10': [0.019, 0.05, 0],
        #     'Object_local1_p_w_11': [0.0, 0.02, 0],
        # }

        T_object_positions_local = {
            #'Object_local1_p_w_to_o': [0, 0, 0],
            'Object_local1_p_w_1': [-0.015, 0.13, 0],   # meter
            'Object_local1_p_w_2': [0.015, 0.13, 0],
            'Object_local1_p_w_3': [-0.015, 0.03, 0],
            'Object_local1_p_w_4': [0.015, 0.03, 0],
            'Object_local1_p_w_5': [0.055, 0.03, 0],
            'Object_local1_p_w_6': [0.055, 0.0, 0],
            'Object_local1_p_w_7': [-0.055, 0.03, 0],
            'Object_local1_p_w_8': [-0.055, 0.0, 0],
        }

        # U_object_positions_local = {
        #     'Object_local1_p_w_to_o': [0, 0, 0],
        #     'Object_local1_p_w_1': [-0.039, 0.105, 0],
        #     'Object_local1_p_w_2': [-0.019, 0.105, 0],
        #     'Object_local1_p_w_3': [0.019, 0.105, 0],
        #     'Object_local1_p_w_4': [0.039, 0.105, 0],
        #     'Object_local1_p_w_5': [-0.019, 0.02, 0],
        #     'Object_local1_p_w_6': [0.019, 0.02, 0],
        #     'Object_local1_p_w_7': [-0.039, 0.0, 0],
        #     'Object_local1_p_w_8': [0.039, 0.0, 0],
        #     'Object_local1_p_w_9': [-0.019, 0.05, 0],
        #     'Object_local1_p_w_10': [0.019, 0.05, 0],
        #     'Object_local1_p_w_11': [0.0, 0.02, 0],
        # }

        U_object_positions_local = {
            #'Object_local1_p_w_to_o': [0, 0, 0],
            'Object_local1_p_w_5': [-0.019, 0.02, 0],
            'Object_local1_p_w_6': [0.019, 0.02, 0],
            'Object_local1_p_w_2': [-0.019, 0.105, 0],
            'Object_local1_p_w_3': [0.019, 0.105, 0],
            'Object_local1_p_w_4': [0.039, 0.105, 0],
            'Object_local1_p_w_8': [0.039, 0.0, 0],
            'Object_local1_p_w_1': [-0.039, 0.105, 0],
            'Object_local1_p_w_7': [-0.039, 0.0, 0],
        }

        # Initialize list to store the (2, 1) points
        keypoints_T = []
        keypoints_U = []

        robot_state = self._print_end_effector_pose_velocity()  # 4, 1
        robot_pos = robot_state[0:2]

        # Process T object points
        for point_key, point_value in T_object_positions_local.items():
            P7_L1 = np.array([point_value]).T  # Shape (3, 1)
            p_w = self.tranform_local_to_world_U_or_T_object(P7_L1=P7_L1, T_object=True)
            p_w_2d = p_w[:2]  # Remove the last item to get shape (2, 1)
            p_w_2d = p_w[:2].reshape(2,) - robot_pos  # Remove the last item and make it (2,)
            keypoints_T.append(p_w_2d)

        # Process U object points
        for point_key, point_value in U_object_positions_local.items():
            P7_L1 = np.array([point_value]).T  # Shape (3, 1)
            p_w = self.tranform_local_to_world_U_or_T_object(P7_L1=P7_L1, T_object=False)
            p_w_2d = p_w[:2]  # Remove the last item to get shape (2, 1)
            p_w_2d = p_w[:2].reshape(2,) - robot_pos # Remove the last item and make it (2,)
            keypoints_U.append(p_w_2d)

        # Concatenate all points to get a (16 * 3, 1) array
        keypoints_T = np.array(keypoints_T)
        keypoints_U = np.array(keypoints_U)
        # print("robot pos: ", robot_pos)
        # print("keypoints_T: ", keypoints_T + robot_pos)
        # print("keypoints_U: ", keypoints_U + robot_pos)
        # print("shape of keypoints_T: ", keypoints_T.shape)
        final_keypoints_array = np.concatenate((keypoints_T, keypoints_U, keypoints_T - keypoints_U), axis=0)
        # print("shape of final_keypoints_array: ", final_keypoints_array.shape)
        return final_keypoints_array.reshape(-1)

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
    
    def tranform_local_to_world_U_or_T_object(self, P7_L1 = np.array([[-0.055, 0.0, 0.0]]).T, T_object = True):
        def quaternion_to_matrix(q):
                """Convert a quaternion (x, y, z, w) to a 3x3 rotation matrix."""
                x, y, z, w = q
                return np.array([
                    [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                    [2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w)],
                    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2)]
                ])
            # Given values

        # T-shape object results, tested 
        if T_object:
            R_L2_to_L1 = np.array([[ 0.95211778, -0.30562785,  0.0079597 ],
                                [-0.30568869, -0.95209741,  0.00805851],
                                [ 0.00511551, -0.01010585, -0.99993585]]
                                )
            P_L2_to_L1 = np.array([[ 0.00029164],
                                [-0.05317678],
                                [ 0.0035383 ]])
        else:

            # U-shape object results
            R_L2_to_L1 = np.array( [[ 0.99044297 , 0.13789649 ,-0.00269775],
                                [-0.13787775 , 0.99042981,  0.00620666],
                                [ 0.00352781, -0.00577539 , 0.9999771 ]])
            P_L2_to_L1 = np.array([[ 0.00156376],
                                    [-0.04893837],
                                    [-0.00310259]]
                                )


        # Define P7 in L1 frame
        # P7_L1 = np.array([[-0.055, 0.0, 0.0]]).T  # Shape (3, 1)

        # # Inverse R_L2_to_L1 to use for L1 to L2 transformation
        R_L1_to_L2 = np.linalg.inv(R_L2_to_L1)

        # Transform P7 from L1 to L2 frame
        P_L2 = R_L1_to_L2 @ P7_L1 + ( R_L1_to_L2 @ P_L2_to_L1)

        # Given measured values of R_w_to_L2 and P_w_to_L2 (assuming quaternion conversion if needed)
        # R_w_to_L2 = np.array([[...], [...], [...]])  # Replace with measured 3x3 rotation matrix for R_w_to_L2
        # P_w_to_L2 = np.array([[...], [...], [...]]).T  # Replace with measured 3x1 position vector for P_w_to_L2
        if T_object:
            P_w_to_L2, R_w_to_L2 = self._print_Tobject_pose() 
        else:
            P_w_to_L2, R_w_to_L2 = self._print_Uobject_pose() 
        # print("P_w_to_L2: ", P_w_to_L2)
        R_w_to_L2 = quaternion_to_matrix(R_w_to_L2)
        # Transform P_L2 to world frame
        # print("R_w_to_L2 @ P_L2: ", R_w_to_L2 @ P_L2)
        R_w_to_L2 = np.array(R_w_to_L2)  # Ensure R_w_to_L2 is a NumPy array
        P_L2 = np.array(P_L2)            # Ensure P_L2 is a NumPy array
        P_w_to_L2 = np.array(P_w_to_L2)  # Ensure P_w_to_L2 is a NumPy array

        # Perform the transformation with reshaping
        P_w = (R_w_to_L2 @ P_L2.reshape(3, 1)) + P_w_to_L2.reshape(3, 1)

        return P_w

    def step(self, action):

        # Run control
        # check the dimension of the action
        if self.ik_type == "track ik":
            # Start IK with current q
            self.init_IK()
        action_r = action.copy()
        self.run(action_r)        # run the controller

        reward = 0
        self.count = self.count + 1
        # done = True if self.count > 20000 else False
        done = False   # not done unless human press the button
        terminated = False 

        info = {}

        # check the state is available
        
        robot_state = self._print_end_effector_pose_velocity()  # 4, 1
        objects_state = self.obtain_keypoints_for_T_and_U_objects() # 24 * 3 , 1

        state = np.concatenate((robot_state, objects_state), axis = 0)   # dim 48

        info['success'] = False

        # control the robot at the end of each epsiode, this is very very important!!! 
        # otherwise, the robot will execute the last action when the algorithm is still traninnig without controlling the robot directly!
        if done or terminated or info['success']:
            # stay at the current state
            self.hold_on_mode()

        
        expanded_obs = np.expand_dims(state, axis=0)
        return [expanded_obs, reward, done, terminated, info]
    
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
        objects_state = self.obtain_keypoints_for_T_and_U_objects() # 18 * 2 , 1

        state = np.concatenate((robot_state, objects_state), axis = 0)   # dim 9
        # state = np.concatenate((ee_pose_relative, self.ee_velocity, self.object_pose_relative), axis = 0)   # dim 18
        # state = np.concatenate((self.ee_pose, self.object_pose, self.goal_object_position), axis = 0) # dim 14
        # state = np.concatenate((self.ee_pose, self.object_pose), axis = 0)
        # print("state: ", state)

        self.ee_goal_position = None

        info = {}
        expanded_obs = np.expand_dims(state, axis=0)
        return [expanded_obs, info]




    def init_IK(self):
        self.IK = inverse_kinematics_init(self.ik_type, self.q, ROBOT)

    def _get_pose(self, joint_q, end_link='iiwa_link_7'):
        pose = self.robot.fkine(joint_q, end=end_link, start='iiwa_link_0')
        position = pose.t
        orientation = pose.R
        return position, orientation

    def run(self, ee_action):
        # Get ee pose
        ee_position, ee_orientation = self.ee_pose[:3], SO3.RPY(self.ee_pose[3:], order='zyx')

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
            self.poition_goal_z = 0.15
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








