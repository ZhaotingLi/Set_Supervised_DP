import logging

from pynput.keyboard import Key, Listener
import numpy as np
import threading

logger = logging.getLogger(__name__)
"""
Class that obtains the human feedback from the computer's keyboard.

Usage: 
1. first selection an axis you want to change, like, x, y or z
2. selection the direction along the axis by the arrows on the keyboard.
"""


class Feedback_keyboard_3d_qbhand:
    def __init__(self):
        self.h = [0, 0, 0]
        self.h_null = [0, 0, 0]
        self.h_direction = None
        self.restart = False
        self.evaluation = False
        self.model_training = False
        self.hand_command = None

        # fix the robot's end effector or not
        self.fix_ee_flag = None

        # fix the robot and the hand if successfully grasp
        self.send_fixed_commands = False

        self.choose_intervention = True # use to select whether to use intervention or correction mode for CLIC method
        self.reduce_velocity = False

        # Starting the listener
        listener_thread = threading.Thread(target=self.start_listener)
        listener_thread.start()
    
    def start_listener(self):
        # This function will run the listener in the background
        with Listener(on_press=self.key_press, on_release=self.key_release) as listener:
            listener.join()

    def key_press(self, k):
        if hasattr(k, 'char'):  # Check if the key has 'char' attribute
            logger.info(f'alphanumeric key {k.char} pressed')
            if k.char == 'o':
                self.hand_command = np.array([-1])
                self.send_fixed_commands = False
                self.reduce_velocity = False
            elif k.char == 'c':
                self.hand_command = np.array([1])
                self.send_fixed_commands = False
                self.reduce_velocity = False

            if k.char == 'f':
                self.fix_ee_flag = True
                self.send_fixed_commands = False
                self.reduce_velocity = False
                self.hand_command = None
            elif k.char == 'e':
                self.fix_ee_flag = False
                self.send_fixed_commands = False
                self.reduce_velocity = False
                self.hand_command = None

            if k.char == 'q':
                self.send_fixed_commands = True
                self.reduce_velocity = False
                self.hand_command = None

            if k.char == 'd':
                self.choose_intervention = True
                self.reduce_velocity = False
            elif k.char == 'l':
                self.choose_intervention = False
                self.reduce_velocity = False

            if k.char == 's':
                self.reduce_velocity = True
            

        else:
            # Handling non-alphanumeric keys
            logger.debug('key pressed:  %s  h_direction:  %s self.h_direction is not None:  %s', k, self.h_direction, self.h_direction is not None)
            if k == Key.space:
                self.restart = True
            if (k == Key.left or k == Key.down) and self.h_direction is not None:
                self.h = [-1 * i for i in self.h_direction]
                logger.debug('self.h :  %s', self.h)
            if (k == Key.right or k == Key.up) and self.h_direction is not None:
                self.h = 1 * self.h_direction
            # if k == Key.e:
            #     self.evaluation = not self.evaluation
            #     print('EVALUATION STARTED' if self.evaluation else 'EVALUATION STOPPED')
            # if k == Key.s:
            #     self.model_training = not self.model_training
            #     print('MODEL TRAINING STARTED' if self.model_training else 'MODEL TRAINING STOPPED')

    def key_release(self, k):
        if k in (Key.left, Key.right, Key.up, Key.down):
            logger.debug("key release")
            # self.h = self.h_null

    def get_h(self):
        received_h = self.h
        self.h = self.h_null
        return received_h

    def ask_for_done(self):
        done = self.restart
        self.restart = False
        return done

    def get_hand_command(self):
        received_hand_command = self.hand_command
        self.hand_command = None
        return received_hand_command
    

    def ask_whether_fix_end_effector(self):
        flag= self.fix_ee_flag
        self.fix_ee_flag = None
        # print("fix_ee_flag: ", flag)
        return flag
    
    # send the end command (no end effector moving, hand closed)
    def get_end_command(self):
        received_end_command = self.send_fixed_commands
        self.send_fixed_commands = False
        return received_end_command
    

    # True is CLIC-D
    # False is CLIC-C
    def get_CLIC_feedback_mode(self):
        return self.choose_intervention
    

    def get_reduce_velocity_feedback(self):
        flag = self.reduce_velocity
        self.reduce_velocity = False
        return flag
