import logging
import rospy
import sensor_msgs.msg as sensor_msg
import numpy as np
from spatialmath import SE3, SO3
from scipy.spatial.transform import Rotation
import std_msgs.msg as std_msg

logger = logging.getLogger(__name__)

"""
Class that obtains the human feedback from the computer's keyboard.
"""


class Feedback_spaceNav_position:
    def __init__(self):
        super(Feedback_spaceNav_position, self).__init__()
        rospy.init_node('robot_control')
        # ROS
        rospy.Subscriber('/spacenav/joy', sensor_msg.Joy, self._callback_spacenav, queue_size=10)

        # Init variables
        self.spacenav_state = None
        
        # Parameters
        self.control_orientation = False
        self.h_translation = np.array([0, 0, 0])

        self.control_gripper = False  # if not control the gripper, 

        self.restart = False  # use buttons to control the restart
        self.time_count = 0
        # [To do] use buttons to control the gripper

        # [To do] you should also consider the error of the feedback signal at the beginning, when the human just touches the spacemouse

    def _callback_spacenav(self, data):
        self.spacenav_state = data.axes
        if data.buttons[0] or data.buttons[1]:
            if self.time_count > 100:  # avoid reset the env serval times while only pressing once
                self.restart = True
                self.time_count = 0
        self.time_count = self.time_count + 1

    def _spacenav_command(self):
        if self.spacenav_state is None:
            return None
        
        ## ABB stuff (1) ## for now removed
        # # Get ee pose
        # ee_position, ee_orientation = self._get_ee_pose()
        ee_position = np.array([0.0, 0.0, 0.0])
        ee_orientation = np.identity(3)

        # Get delta x
        # print("self.spacenav_state: ", self.spacenav_state)
        # ee_delta_position = np.array(self.spacenav_state)[:3]
        ######################### [only x-y action]
        ee_delta_position = np.array(self.spacenav_state)[:2]

        # AttributeError: 'numpy.ndarray' object has no attribute 'norm'

        # AttributeError: 'numpy.ndarray' object has no attribute 'norm'
        ee_delta_position_norm = np.linalg.norm(ee_delta_position)
        if ee_delta_position_norm < 0.25:
            ee_delta_position_processed = None
            # ee_delta_position_processed = ee_delta_position 
        else:
            # normalize
            # ee_delta_position_processed = ee_delta_position / ee_delta_position_norm
            ee_delta_position_processed = ee_delta_position          
        # print("ee_delta_position: ", ee_delta_position, "ee_delta_position_processed: ", ee_delta_position_processed)

        # if not self.control_gripper:
        #     # append 1 to the end of the array
        #     ee_delta_position_processed = np.append(ee_delta_position_processed, 1)

        # Get desired position
        # ee_goal_position = ee_position + ee_delta_position * self.scale

        # Get delta orientation
        #rotation_object = Rotation.from_matrix(ee_orientation)       
        #orientation_euler = rotation_object.as_euler('xyz')

        #try:
        #    self.desired_ee_orientation
        #except:
        #    self.desired_ee_orientation = orientation_euler

        
        if self.control_orientation:
            logger.warning("control_orientation not implemented yet")
            # ee_delta_orientation = np.array(self.spacenav_state)[3:] * self.scale
            # ee_delta_orientation = SE3.RPY(ee_delta_orientation, order='zyx').R
        # else:  
            # orientation_d_dot = self.linear_DS_orientation(self.orientation_goal, ee_orientation)
            # ee_delta_orientation =  orientation_d_dot * self.delta_t_orientation

        return ee_delta_position_processed 
    

    def get_h(self):
        self.h = self._spacenav_command()
        return self.h

    def ask_for_done(self):
        done = self.restart
        self.restart = False
        return done