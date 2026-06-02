import logging
import rospy
import sensor_msgs.msg as sensor_msg
import numpy as np
from spatialmath import SE3, SO3
from scipy.spatial.transform import Rotation
import std_msgs.msg as std_msg

from tools.feedback_spacenav import Feedback_spaceNav

logger = logging.getLogger(__name__)
"""
Class that obtains the human feedback from the computer's keyboard.
"""


import threading
from pynput.keyboard import Listener, KeyCode


class Feedback_receiver_combined:
    def __init__(self):
        super(Feedback_receiver_combined, self).__init__()
        
        self.feedback_spacemouse = Feedback_spaceNav()
        # feedback_keyboard = feedback_keyboard()  # TODO

        # Init variables
        dim_action = 7
        self.h = np.zeros(dim_action)

        self.offset_gripper = 0.0
          # Starting the listener
         # NEW: keyboard-based world-Z rotation
        self.rot_command = 0          # -1, 0, +1
        self.rot_step = 0.2          # radians per control step

        self.hand_command = 0
        listener_thread = threading.Thread(target=self.start_listener)
        listener_thread.start()
    
    def start_listener(self):
        # This function will run the listener in the background
        with Listener(on_press=self.key_press, on_release=self.key_release) as listener:
            listener.join()

        rospy.sleep(1)

    def key_press(self, k):
        if hasattr(k, 'char'):  # Check if the key has 'char' attribute
            ch = k.char.lower()
            logger.info(f'alphanumeric key {ch} pressed')
            if ch == 'o':
                self.hand_command = 1
                self.send_fixed_commands = False
                self.reduce_velocity = False
            elif ch == 'c':
                self.hand_command = -1
            # NEW: rotation commands
            elif ch == 'r':      # rotate +z in world
                self.rot_command = 1
            elif ch == 't':      # rotate -z in world
                self.rot_command = -1
            
            self.offset_gripper = self.hand_command

    def key_release(self, k):
        if hasattr(k, 'char'):
            ch = k.char.lower()
            if ch in ('o', 'c'):
                self.hand_command = 0
            # NEW: stop rotating when R/T is released
            if ch in ('r', 't'):
                self.rot_command = 0
            self.offset_gripper = self.hand_command

    def get_h(self):
        
        self.h[0:6] = self.feedback_spacemouse._spacenav_command()

        # NEW: keyboard-based rotation around WORLD z-axis
        if self.rot_command != 0:
            yaw = self.rot_command * self.rot_step  # + or - about world z
            self.h[3:6] = np.array([0.0, 0.0, yaw]).copy()
            
        

        self.h[-1] = self.offset_gripper   # gripper command, TODO
        h_output = self.h.copy()
        # self.offset_gripper = 0.0
        logger.debug('self.h:  %s', h_output)
        return h_output

    def ask_for_done(self):
        done = self.feedback_spacemouse.restart
        self.feedback_spacemouse.restart = False
        return done
