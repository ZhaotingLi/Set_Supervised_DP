import logging

logger = logging.getLogger(__name__)


from sensor_msgs.msg import JointState
from env.kuka.inverse_kinematics import inverse_kinematics_init
from spatialmath import SO3
from spatialmath import UnitQuaternion
import cor_tud_msgs.msg as cor_msg
import sensor_msgs.msg as sensor_msg
import numpy as np
import rospy
import sys
from geometry_msgs.msg import PoseStamped

# message used for qb_hand
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import JointTrajectoryControllerState


import math
import random

from std_msgs.msg import Float32MultiArray


ROBOT = 'iiwa14'
# ROBOT = 'iiwa7'

# create a message for qbhand
def create_trajectory_message(position, velocity=0, duration=1):
    msg = JointTrajectory()
    msg.header.stamp = rospy.Time.now()
    msg.joint_names = ['qbhand1_synergy_joint']

    point = JointTrajectoryPoint()
    point.positions = [position]
    point.velocities = [velocity]
    point.time_from_start = rospy.Duration(duration)

    msg.points.append(point)
    return msg


class KUKAenv_6dEE:
    def __init__(self):
        # self.joint_impedance_controller = JointImpedanceController(robot_name=ROBOT, alpha=0.5)
        # self.robot = iiwa(model=ROBOT)
        # self.cartesian_impedance_controller = CartesianImpedanceController(robot_name=ROBOT, robot=self.robot)
        self.IK = None
        
        # ROS
        # rospy.Subscriber('/spacenav/joy', sensor_msg.Joy, self._callback_spacenav, queue_size=10)
        rospy.Subscriber('/%s/end_effector_state' % ROBOT, cor_msg.CartesianState, self._callback_end_effector_state, queue_size=10)
        rospy.Subscriber('/%s/joint_states' % ROBOT, JointState, self._callback_joint_states, queue_size=10)
        self.request_pub = rospy.Publisher('/%s/control_request' % ROBOT, cor_msg.ControlRequest, queue_size=10)
        # read box pose
        rospy.Subscriber("/vrpn_client_node/bottle/pose", PoseStamped, self._callback_object_pose, queue_size=10)

        rospy.Subscriber("/sensor_averages", Float32MultiArray, self._callback_tactile_sensor, queue_size=10)

        # control the qbhand
        self.pub_qbhand = rospy.Publisher('/qbhand1/control/qbhand1_synergy_trajectory_controller/command', JointTrajectory, queue_size=10)
        rospy.Subscriber('/qbhand1/control/qbhand1_synergy_trajectory_controller/state', JointTrajectoryControllerState, self._qb_hand_state_callback)
        self.qbhand_position = None

        
        # Init variables
        self.q = None
        self.ee_pose = None
        self.ee_velocity = None
        self.spacenav_state = None
        
        # Parameters
        self.controller = 'ik'  # options: cartesian_impedance, ik
        self.ik_type = 'ranged ik'  # options: track ik, ranged ik
        # self.control_orientation = True
        self.control_orientation = True
        self.scale = 0.015  # changed as we modify the spacemouse_feedback.py
        self.scale_ori = 0.05   # changed as we modify the spacemouse_feedback.py
        # self.scale = 0.05 * 0.1
        self.delta_q_max = np.array([0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08])  # maximum requested delta for safety

        
        self.control_position_z = False # do not control the pose_z via space mouse
        self.poition_goal_z = None
        # self.orientation_goal = np.array([0, 0, 0])
        self.delta_t_orientation = 1  # can only be 1, you cannot scale the transform matrix directly
        self.gain_orientation = 0.1

        self.ee_goal_position = None
        # self.orientation_goal = np.array([-1.18, 0.15, 3.07])
        # self.ee_pos_limit_xyz = [-0.32, 0.26, -0.72, -0.5, 0.05, 0.43]   # xlow xup ylow yup zlow zup
        # self.orientation_goal = np.array([-1.431268417238323, 0.48663172527731396, -1.5800998550262397])  # 12.14 exp
        self.orientation_goal = np.array([-1.1944169325418168, 1.5378410392851685, -1.2136691829618993])
        self.ee_pos_limit_xyz = [0.2, 0.8, -0.5, 0.5, 0.2, 0.8]
        self.ee_fixed_range_xyz = [0.58, 0.64, -0.3, 0.3, 0.6, 0.76]

        # pose of object
        self.object_pose = None  # x, y, z + roll-pitch-yaw
        self.object_pose_relative = None
        self.use_simulation = False  # if use simulation, set the object pose as zeros
        self.object_position_velocity = None
        self.last_object_position = None
        self.sensor_tactile_readings = None

        self.count = 0
    
        # original state when reset the robot
        self.rest_controller_gain = 0.3
        self.q_goal_reset = np.array([-0.3844571707497305, 0.8402530767167159, 0.7733644435997699, -1.2256496309028426, -0.5527975885687727, 1.3058055798056998, 0.30103516092859445])

        # the intermidate one is set to keep safe, keep the robot from colliding into the table
        self.q_goal_middle_reset = np.array([-0.3844571707497305, 0.8402530767167159, 0.7733644435997699, -1.2256496309028426, -0.5527975885687727, 1.3058055798056998, 0.30103516092859445])

        # task specific parameters
        # self.goal_object_position = np.array([0.21494782, -0.50807971]) # to modify
        self.goal_object_position = np.array([-0.54, 0.035, 0,117]) 

        self.quat_offset = UnitQuaternion([1, 0, 0, 0])
        self.first_quaternion_checked = False

        self.init_IK()

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
    
    # # kuka 14
    # def transform_box_to_robot_frame(self, box_pose): # input: box_pose 3 x 1 only position
    #     # robot_base in box frame: [0.04, 0.23, 0]
    #     # orientation: robot_x is box_x + 1.6, robot_y is box_y, robot_z is box_z

    #     # transform the box pose to robot frame
    #     box_pose_robot_frame = box_pose.copy()
    #     box_pose_robot_frame[0] = box_pose[0] + 1.6
    #     box_pose_robot_frame[1] = box_pose[1] 
    #     box_pose_robot_frame[2] = box_pose[2] 
    #     return box_pose_robot_frame
    
    def transform_box_to_robot_frame(self, box_pose): # input: box_pose 3 x 1 only position
        # robot_base in box frame: [0.04, 0.23, 0]
        # orientation: robot_x is ball_x  - 0.055, robot_y is  ball_y, robot_z is ball_z 
        # transform the box pose to robot frame
        box_pose_robot_frame = box_pose.copy()
        box_pose_robot_frame[0] = box_pose[1] -0.055
        box_pose_robot_frame[1] = box_pose[1]
        box_pose_robot_frame[2] = box_pose[2] 
        return box_pose_robot_frame
    
    def _qb_hand_state_callback(self, data):
        # print("data: ", data)
        self.qbhand_position = data.actual.positions  # [joint_pos]
        # self.qbhand_velocity = data.actual.velocities[0]

    def get_state(self):
        # check the state is available
        # if self.ee_pose is None or self.object_pose is None or self.qbhand_position is None:
        #     print("--------------------------error!---------------------------------------------------------------")
        #     print("no robot data: ", self.ee_pose is None, " no object data: ", self.object_pose is None, " no qbhand data: ", self.qbhand_position is None)
        #     # if self.use_simulation:
        #     if self.object_pose is None:
        #         self.object_pose = np.zeros(6)
        #         self.object_pose_relative = np.zeros(6)
        #         self.object_position_velocity = np.zeros(3)
        #     if self.qbhand_position is None:
        #         self.qbhand_position = np.array([0])
        # self.last_object_position = self.object_pose[0:3]
        # state = np.concatenate((self.ee_pose, self.object_pose), axis = 0)
        # state = np.concatenate((self.ee_pose, self.object_pose, self.goal_object_position), axis = 0)
        # print("state: ", state)
        # ee_pose_relative = self.ee_pose
        # ee_pose_relative[0:3] = ee_pose_relative[0:3] - self.transform_box_to_robot_frame(self.object_pose[0:3])
        # state = np.concatenate((self.ee_pose[0:3], self.qbhand_position), axis = 0)

        quaternion = UnitQuaternion.RPY(self.ee_pose[3:6]).A
        pose = np.concatenate([self.ee_pose[0:3], quaternion])

        x_processed = self._preprocess_x(pose)

        # UnitQuaternion
        # state = np.concatenate((self.ee_pose[0:3], self.qbhand_position), axis = 0)
        # print("state: ", x_processed)
        
        if self.sensor_tactile_readings is None:
            self.sensor_tactile_readings = np.zeros(5)
        state =  np.concatenate((x_processed, self.sensor_tactile_readings), axis = 0) # 7+5

        expanded_obs = np.expand_dims(state, axis=0)
        logger.debug('%s %s', "shape of expanded_obs: ", expanded_obs.shape)
        # state = np.concatenate((self.ee_pose[0:3], self.qbhand_position, self.ee_velocity[0:3], self.transform_box_to_robot_frame(self.object_pose[0:3]), self.object_position_velocity), axis = 0)
        # state = np.concatenate((ee_pose_relative, self.ee_velocity, self.object_pose_relative), axis = 0)   # dim 18
        return expanded_obs

    def _preprocess_x(self, x_raw):
        x = np.zeros(7)
        # Map from robot to optitrack
        # x[0] = x_raw[0] - self.x_offset
        # x[1] = x_raw[1] - self.y_offset
        # x[2] = x_raw[2] - self.z_offset
        x[0] = x_raw[0] 
        x[1] = x_raw[1] 
        x[2] = x_raw[2] 
        orientation = UnitQuaternion([x_raw[3], x_raw[4], x_raw[5], x_raw[6]])

        # Check that we start in hemisphere nearest to goal
        if not self.first_quaternion_checked:
            orientation_1 = orientation.A
            orientation_2 = -orientation.A

            # Calculate the dot product
            dot_product_1 = orientation_1.dot(self.quat_offset.A)
            dot_product_2 = orientation_2.dot(self.quat_offset.A)

            # Calculate the angular distance
            angular_distance_1 = math.acos(dot_product_1)
            angular_distance_2 = math.acos(dot_product_2)

            if angular_distance_1 < angular_distance_2:
                orientation = UnitQuaternion(orientation_1)
                logger.debug('%s %s', 'Not flipped', angular_distance_1)
            else:
                orientation = UnitQuaternion(orientation_2)
                logger.debug('%s %s', 'Flipped', angular_distance_2)
            
            # Change flag
            self.first_quaternion_checked = True

        orientation_new = (orientation / self.quat_offset).A    
        x[3] = orientation_new[0]
        x[4] = orientation_new[1]
        x[5] = orientation_new[2]
        x[6] = orientation_new[3]

        return x


    def step(self, action):

        # print("action: ", action)
        # Run control
        # check the dimension of the action
        if self.q is None:
            logger.debug("Not receving q! Waiting...")
            state = self.get_state()
            return [state, 0, False, False, []]
        if self.ik_type == "track ik":
            # Start IK with current q
            self.init_IK()
        # # the action now is the delta position of the end-effector + control the qbhand
        # # check the dimension of the action
        # # if len(action) != 7:
        # #     print("action dimension is not correct! action: ", action)
        # #     return None
        
        # # the one control the soft hand
        # action[-1] = 0.4
        
        self.run(action[0:3], action[3:6])        # run the controller
        ## control the qbhand, the output of nn is between [-1, 1], we need to scale it to [0, 1]
        # msg = create_trajectory_message((action[-1] + 1.0) / 2.0)
        # self.pub_qbhand.publish(msg)

        reward = 0
        self.count = self.count + 1
        # done = True if self.count > 20000 else False
        done = False   # not done unless human press the button
        terminated = False 

        info = {}

        state = self.get_state()
        ## check whether the task is finished
        # info['success'] = self.check_box_pushing_task_success()
        info['success'] = False

        # control the robot at the end of each epsiode, this is very very important!!! 
        # otherwise, the robot will execute the last action when the algorithm is still traninnig without controlling the robot directly!
        if done or terminated or info['success']:
            # stay at the current state
            self.hold_on_mode()

        return [state, reward, done, terminated, info]
    
    def hold_on_mode(self):
        logger.debug("hold on mode")
        control_frequency = 500
        rate = rospy.Rate(control_frequency)
        last_current_q = self.q
        for i in range(500):
            # self.joint_impedance_controller.send_torque_request(last_current_q)
            self.send_position_request(last_current_q)
            rate.sleep()
        return 
    
    
    def request_random_ee_position(self):
        def sample_3d_point(limits):
            x = random.uniform(limits[0], limits[1])
            y = random.uniform(limits[2], limits[3])
            z = random.uniform(limits[4], limits[5])
            return x, y, z

        sampled_point = sample_3d_point(self.ee_fixed_range_xyz)   # sample a position within the predefined valid region
        
        ee_goal_orientation = SO3.RPY(self.orientation_goal, unit='rad', order='zyx')

        q_goal_fixed_ee = self.IK.compute(sampled_point, ee_goal_orientation.rpy(), self.q)

        control_frequency = 500
        rate = rospy.Rate(control_frequency)
        while(np.linalg.norm(q_goal_fixed_ee[:] - self.q[:]) > 0.13): # set [1:] because we want the 0-joint to be quite accurate
            q_delta = self.rest_controller_gain * (q_goal_fixed_ee - self.q)
            q_d = self.q + q_delta
            # self.joint_impedance_controller.send_torque_request(q_d)
            self.send_position_request(q_d)
            rate.sleep()

        self.get_state()

        return


    def reset(self):
        self.count = 0
        control_frequency = 500
        rate = rospy.Rate(control_frequency)
        self.ee_goal_position = None

        logger.debug("reset the env [start]")

        # reached_middle_goal = False
        # while not reached_middle_goal:
        #     q_delta = self.rest_controller_gain * (self.q_goal_middle_reset - self.q)
        #     q_d = self.q + q_delta
        #     reached_middle_goal = False if np.linalg.norm(self.q_goal_middle_reset - self.q) > 0.3 else True
        #     # self.joint_impedance_controller.send_torque_request(q_d)
        #     self.send_position_request(q_d)
        #     rate.sleep()

        # first list the end effector up
        for i in range(10):
            self.run(np.array([0, 0, 1]), np.array([0, 0, 0]))
            rate.sleep()

        # while(np.linalg.norm(self.q_goal_reset[0:] - self.q[0:]) > 0.2): # set [1:] because we want the 0-joint to be quite accurate
        while(np.linalg.norm(self.q_goal_reset[0:] - self.q[0:]) > 0.075): # set [1:] because we want the 0-joint to be quite accurate
        # while(np.linalg.norm(self.q_goal_reset - self.q) > 0.035):
            # print("np.linalg.norm(self.q_goal_reset - self.q) ", np.linalg.norm(self.q_goal_reset - self.q) )
            # if self.q is None:
            #     print("self.q is None")
            #     continue
            q_delta = self.rest_controller_gain * (self.q_goal_reset - self.q)
            q_d = self.q + q_delta
            # self.joint_impedance_controller.send_torque_request(q_d)
            self.send_position_request(q_d)
            # print("q_delta: ", np.linalg.norm(self.q_goal_reset[0:] - self.q[0:]), " self.count: ", self.count)
            rate.sleep()

            self.count = self.count  + 1
            if self.count > 10000 and np.linalg.norm(self.q_goal_reset[:-1] - self.q[:-1]) < 0.6:
                break
        

        # msg = create_trajectory_message(0)
        msg = create_trajectory_message(0.7)
        self.pub_qbhand.publish(msg)

        self.count = 0
        self.ee_goal_position = None

        state = self.get_state()
        # state = np.concatenate((ee_pose_relative, self.ee_velocity, self.object_pose_relative), axis = 0)   # dim 18
        # state = np.concatenate((self.ee_pose, self.object_pose, self.goal_object_position), axis = 0) # dim 14
        # state = np.concatenate((self.ee_pose, self.object_pose), axis = 0)
        # print("state: ", state)

        info = {}
        logger.debug("reset the env [Done]")
        return [state, info]

    
    def check_box_pushing_task_success(self):
        if np.linalg.norm(self.goal_object_position[:2] - self.object_pose[:2]) < 0.03:
            return True
        return False

    def _callback_tactile_sensor(self, data):
        self.sensor_tactile_readings = np.clip(np.array(data.data), -100, 100) / 50.0
        # print("tactile sensor: ", self.sensor_tactile_readings)

    def _callback_end_effector_state(self, data):
        self.ee_pose = np.array(data.pose.data)
        self.ee_velocity = np.array(data.velocity.data)
        
        
    def _callback_joint_states(self, data):
        self.q = np.array(data.position)

    def _callback_object_pose(self, data):
        self.object_pose = np.zeros(6)
        # rospy.loginfo(rospy.get_caller_id() + "I heard pose %s", data.pose)
        self.object_pose[0] = data.pose.position.x
        self.object_pose[1] = data.pose.position.y  
        self.object_pose[2] = data.pose.position.z 
        x = data.pose.orientation.x
        y = data.pose.orientation.y
        z = data.pose.orientation.z
        w = data.pose.orientation.w
        roll, pitch, yaw = self.euler_from_quaternion(x, y, z, w)
        self.object_pose[3] = roll
        self.object_pose[4] = pitch
        self.object_pose[5] = yaw   

        # caclulate the velocity
        if self.last_object_position is None:
            self.last_object_position = self.object_pose[0:3] 
        self.object_position_velocity = self.object_pose[0:3] - self.last_object_position
        # self.last_object_position = self.object_pose[0:3]

        # self.object_pose_relative = self.object_pose.copy()
        # self.object_pose_relative[0:3] = self.object_pose[0:3] - self.goal_object_position[0:3]

    def euler_from_quaternion(self, x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)

        return roll_x, pitch_y, yaw_z  # in radians


    def init_IK(self):
        self.IK = inverse_kinematics_init(self.ik_type, self.q, ROBOT)

    def run(self, ee_delta_position, ee_delta_ori_command):
        # Get ee pose
        ee_position, ee_orientation = self.ee_pose[:3], SO3.RPY(self.ee_pose[3:], order='zyx')

        # Get delta x, for kuka_env, the ee_delta_position is the action
        # ee_delta_position = np.array(self.spacenav_state)[:3]

        # constrain the input, for the box-pushing task, we first consider the easiest case, where the ee moves in a plane
        # in the future, this should be modified to a safety region
        # [1] set translation_z = 0

        # if not self.control_position_z:
        #     if self.poition_goal_z is None:
        #         self.poition_goal_z = self.ee_pose[2]
        #     ee_delta_position[2] = (self.poition_goal_z - self.ee_pose[2]) / self.scale
        
        
        # only update the desired position when we receive the space mouse feedback
        if self.ee_goal_position is None:
            self.ee_goal_position = ee_position
        # if np.linalg.norm(np.array(self.spacenav_state)[:3]) > 0.01:
        if np.linalg.norm(np.array(ee_delta_position[:2])) > 0.01:
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
            ee_delta_orientation = ee_delta_ori_command * self.scale_ori
            ee_delta_orientation = SO3.RPY(ee_delta_orientation, order='zyx')
            # Get desired orientation
            ee_goal_orientation = ee_delta_orientation @ ee_orientation
        else:
            # Get desired orientation
            ee_goal_orientation = SO3.RPY(self.orientation_goal, unit='rad', order='zyx')

        # ###################### Check whether orientation violate the constraints ##############################
        # rotation_matrix_EE = ee_orientation.A
        # vector_x_axis_in_world_ = np.array([1, 0, 0])

        # vector_z_axis_in_local = np.array([0, 0, 1])
        # vector_z_axis_in_world = rotation_matrix_EE @ vector_z_axis_in_local
        # # caculate the angle
        # angle_diff = np.arccos(np.dot(vector_z_axis_in_world, vector_x_axis_in_world_)) * 180 / np.pi
        # # print("z angle_diff: ", angle_diff )
        # if abs(angle_diff) > 45:
        #     # first check whether the new goal leads to smaller angle diff

        #     rotation_matrix_EE = ee_goal_orientation.A
        #     vector_z_axis_in_world = rotation_matrix_EE @ vector_z_axis_in_local
        #     # caculate the angle
        #     angle_diff2 = np.arccos(np.dot(vector_z_axis_in_world, vector_x_axis_in_world_)) * 180 / np.pi
        #     if abs(angle_diff2) > abs(angle_diff):
        #         ee_goal_orientation = ee_orientation

        if self.controller == 'cartesian_impedance':
            # Send command to controller
            self.cartesian_impedance_controller.control_law(self.ee_goal_position, ee_goal_orientation)

        elif self.controller == 'ik':
            # Compute inverse kinematics
            q_d = self.IK.compute(self.ee_goal_position, ee_goal_orientation.rpy(), self.q)

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








