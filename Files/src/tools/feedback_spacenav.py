import rospy
import sensor_msgs.msg as sensor_msg
import numpy as np
from spatialmath import SE3, SO3
from scipy.spatial.transform import Rotation
import std_msgs.msg as std_msg

"""
Class that obtains the human feedback from the computer's keyboard.
"""


class Feedback_spaceNav:
    def __init__(self):
        super(Feedback_spaceNav, self).__init__()
        rospy.init_node('robot_control')
        # ROS
        rospy.Subscriber('/spacenav/joy', sensor_msg.Joy, self._callback_spacenav, queue_size=10)

        # Init variables
        self.spacenav_state = np.zeros(6)
        
        # Parameters
        self.control_orientation = True
        self.h_translation = np.array([0, 0, 0])

        self.control_gripper = False  # if not control the gripper, 

        self.restart = False  # use buttons to control the restart
        self.time_count = 0
        # [To do] use buttons to control the gripper

        # [To do] you should also consider the error of the feedback signal at the beginning, when the human just touches the spacemouse

    # def _callback_spacenav(self, data):
    #     self.spacenav_state = data.axes
    #     if data.buttons[0] or data.buttons[1]:
    #         if self.time_count > 100:  # avoid reset the env serval times while only pressing once
    #             self.restart = True
    #             self.time_count = 0
    #     self.time_count = self.time_count + 1

    def _callback_spacenav(self, data):
        # spacenav_state = data.axes
        # self.spacenav_state = np.zeros(6)
        # if data.buttons[0] and data.buttons[1]:
        #     if self.time_count > 100:  # avoid reset the env serval times while only pressing once
        #         self.restart = True
        #         self.time_count = 0
        # if data.buttons[0]:  # right button for position
        #     self.spacenav_state[0:3] = np.array(spacenav_state)[:3]
        # elif data.buttons[1]:# left button for orientation
        #     self.spacenav_state[3:6] = np.array(spacenav_state)[3:]
        # else:
        #     self.spacenav_state = np.array(spacenav_state)
        # self.time_count = self.time_count + 1

        spacenav_state = data.axes
        if data.buttons[0] or data.buttons[1]:
            if self.time_count > 100:  # avoid reset the env serval times while only pressing once
                self.restart = True
                self.time_count = 0

        self.spacenav_state = np.array(spacenav_state)
        self.time_count = self.time_count + 1

    def _spacenav_command(self):
        # if np.linalg.norm(self.spacenav_state) < 0.1:
        #     return None
        
        ## ABB stuff (1) ## for now removed
        # # Get ee pose
        # ee_position, ee_orientation = self._get_ee_pose()
        # ee_position = np.array([0.0, 0.0, 0.0])
        # ee_orientation = np.identity(3)

        # Get delta x
        # print("self.spacenav_state: ", self.spacenav_state)
        ee_delta_position = np.array(self.spacenav_state)[:3]

        # AttributeError: 'numpy.ndarray' object has no attribute 'norm'
        ee_delta_position_norm = np.linalg.norm(ee_delta_position)
        if ee_delta_position_norm < 0.01:
            ee_delta_position_processed = None
        else:
            # normalize
            # ee_delta_position_processed = ee_delta_position / ee_delta_position_norm
            ee_delta_position_processed = ee_delta_position 

            # # should we normalize it? 
            # ee_delta_position_processed = ee_delta_position 
        
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
            ee_delta_orientation_zyx = np.array(self.spacenav_state)[3:]
            ee_delta_orientation_zyx_norm = np.linalg.norm(ee_delta_orientation_zyx)
            # if ee_delta_orientation_zyx_norm < 0.2:
            #     ee_delta_orientation_zyx_processed = None
            # else: 
            #     # ee_delta_orientation_zyx_processed = ee_delta_orientation_zyx / ee_delta_orientation_zyx_norm
            #     ee_delta_orientation_zyx_processed = ee_delta_orientation_zyx
            ee_delta_orientation_zyx_processed = ee_delta_orientation_zyx

        # # check which type of feedback has the largest norm (orientation or position)
        # if ee_delta_position_norm > ee_delta_orientation_zyx_norm:
        #     ee_delta_orientation_zyx_processed = np.zeros(3)
        # elif ee_delta_position_norm < ee_delta_orientation_zyx_norm:
        #     ee_delta_position_processed = np.zeros(3)
        
        # h = np.concatenate((ee_delta_position_processed, ee_delta_orientation_zyx_processed), axis=None)

        # if np.linalg.norm(h) < 0.2:
        #     h = None

        # print("ee_delta_position_processed: ", ee_delta_position_processed, " ee_delta_orientation_zyx_processed: ", ee_delta_orientation_zyx_processed)
        # return ee_delta_position_processed,  ee_delta_orientation_zyx_processed

        if ee_delta_position_processed is None:
            ee_delta_position_processed = np.zeros(3)
        if ee_delta_orientation_zyx_processed is None:
            ee_delta_orientation_zyx_processed = np.zeros(3)  
        h = np.concatenate((ee_delta_position_processed, ee_delta_orientation_zyx_processed), axis=None)
        return h
    

    def get_h(self):
        
        self.h = self._spacenav_command()
        return self.h

    def ask_for_done(self):
        done = self.restart
        self.restart = False
        return done
