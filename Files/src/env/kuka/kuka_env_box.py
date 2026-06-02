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


class KUKAenv:
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
        rospy.Subscriber("/vrpn_client_node/box/pose", PoseStamped, self._callback_object_pose, queue_size=10)
        
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
        self.object_pose = None  # x, y, z + roll-pitch-yaw
        self.object_pose_relative = None
        self.use_simulation = False  # if use simulation, set the object pose as zeros

        self.count = 0
    
        # original state when reset the robot
        self.rest_controller_gain = 0.35

        # one nice joint config [-1.0021906347288463, 1.253672343939383, 0.6721988301722307, -1.554135497858412, -1.0877502828244043, 0.7306915548498796, 1.8073318499420343]
        self.q_goal_reset = np.array([-0.4283028356721598, 1.2909959540843803, -0.01831621002748346, -1.199736261168323, 0.013749740848191472, 0.6474497877985909, 1.941621681307243])
        # the intermidate one is set to keep safe, keep the robot from colliding into the table
        self.q_goal_middle_reset = np.array([-0.4283028356721598, 1.1909959540843803, -0.01831621002748346, -1.199736261168323, 0.013749740848191472, 0.6474497877985909, 1.941621681307243])

        # task specific parameters
        # self.goal_object_position = np.array([0.21494782, -0.50807971]) # to modify
        # self.goal_object_position = np.array([0.3729, -0.4155, 0.0585])  
        self.goal_object_position = np.array([0.242, -0.46, 0.079])  
        self.orientation_goal = np.array([3.129662993390037, 0.01070695587553136, 1.5216094657886965])
        self.ee_pos_limit_xyz =  [0.3, 0.7, -0.4, 0.4, 0.07, 0.16]
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
    
    def transform_box_to_robot_frame(self, box_pose): # input: box_pose 3 x 1 only position
        # robot_base in box frame: [0.04, 0.23, 0]
        # orientation: robot_x is -1 * box_y + 0.18, robot_y is box_x -0.14 , robot_z is box_z +0.06

        # self.goal_object_position = np.array([0.3729, -0.4155, 0.0585])  
        self.goal_object_position = np.array([0.242, -0.46, 0.079]) 

        # 0.398, -0.3846, 0.0636 goal in robot 0.2954502346493191, 0.5954097027747056, 0.2801362416677936 

        # transform the box pose to robot frame
        box_pose_robot_frame = box_pose.copy()
        box_pose_robot_frame[0] = - (box_pose[1] + 0.045) + 0.18
        box_pose_robot_frame[1] = (box_pose[0]) -0.1
        box_pose_robot_frame[2] = box_pose[2] 
        return box_pose_robot_frame

    def step(self, action):

        # Run control
        # check the dimension of the action
        if self.q is None:
            logger.debug("Not receving q! Waiting...")
            # state = np.concatenate((self.ee_pose, self.object_pose), axis = None)
            # state = np.concatenate((self.ee_pose, self.object_pose, self.goal_object_position), axis = 0)
            ee_pose_relative = self.ee_pose
            ee_pose_relative[0:3] = ee_pose_relative[0:3] - self.transform_box_to_robot_frame(self.object_pose[0:3])
            # state = np.concatenate((ee_pose_relative, self.ee_velocity, self.object_pose_relative), axis = 0)   # dim 18
            state = np.concatenate((ee_pose_relative[0:3], self.object_pose_relative), axis = 0)   # dim 9

            return [state, 0, False, False, []]
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
        if self.ee_pose is None or self.object_pose is None:
            logger.debug("--------------------------error!---------------------------------------------------------------")
            logger.debug('%s %s %s %s', "no robot data: ", self.ee_pose is None, " no object data: ", self.object_pose is None)
        # state = np.concatenate((self.ee_pose, self.object_pose), axis = 0)
        # state = np.concatenate((self.ee_pose, self.object_pose, self.goal_object_position), axis = 0)
        # print("state: ", state)
        ee_pose_relative = self.ee_pose.copy()
        ee_pose_relative[0:3] = ee_pose_relative[0:3] - self.transform_box_to_robot_frame(self.object_pose[0:3])
        # state = np.concatenate((ee_pose_relative, self.ee_velocity, self.object_pose_relative), axis = 0)   # dim 18
        state = np.concatenate((ee_pose_relative[0:3], self.object_pose_relative), axis = 0)   # dim 9

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
            reached_middle_goal = False if np.linalg.norm(self.q_goal_middle_reset - self.q) > 0.3 else True
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


        # check the state is available
        if self.ee_pose is None or self.object_pose is None:
            logger.debug("--------------------------error!---------------------------------------------------------------")
            logger.debug('%s %s %s %s', "no robot data: ", self.ee_pose is None, " no object data: ", self.object_pose is None)
            # if self.use_simulation:
            self.object_pose = np.zeros(6)
            self.object_pose_relative = np.zeros(6)

        ee_pose_relative = self.ee_pose
        ee_pose_relative[0:3] = ee_pose_relative[0:3] - self.transform_box_to_robot_frame(self.object_pose[0:3])
        state = np.concatenate((ee_pose_relative[0:3], self.object_pose_relative), axis = 0)   # dim 9
        # state = np.concatenate((ee_pose_relative, self.ee_velocity, self.object_pose_relative), axis = 0)   # dim 18
        # state = np.concatenate((self.ee_pose, self.object_pose, self.goal_object_position), axis = 0) # dim 14
        # state = np.concatenate((self.ee_pose, self.object_pose), axis = 0)
        # print("state: ", state)

        self.ee_goal_position = None

        info = {}
        return [state, info]

    
    def check_box_pushing_task_success(self):
        if np.linalg.norm(self.goal_object_position[:2] - self.object_pose[:2]) < 0.03:
            return True
        return False

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

        self.object_pose_relative = self.object_pose.copy()
        self.object_pose_relative[0:3] = self.object_pose[0:3] - self.goal_object_position[0:3]

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
            self.poition_goal_z = 0.08
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








