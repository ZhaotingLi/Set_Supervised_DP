import logging

from pynput.keyboard import Key, Listener
import threading

logger = logging.getLogger(__name__)
"""
Class that obtains the human feedback from the computer's keyboard.

Usage: 
1. first selection an axis you want to change, like, x, y or z
2. selection the direction along the axis by the arrows on the keyboard.
"""

# control the robot team 
# each time, the human can only control one robot
# the human can control the robot by pressing the arrow keys
# the human can switch the robot by pressing the number keys
class Feedback_keyboard_multi_2d:
    def __init__(self, robot_num):
        self.h = [0, 0]
        self.h_null = [0, 0]
        self.h_direction = None
        self.restart = False
        self.evaluation = False
        self.model_training = False

        self.robot_num = robot_num 
        self.control_robot_id = 0 # the robot id that the human is controlling

        # Starting the listener
        listener_thread = threading.Thread(target=self.start_listener)
        listener_thread.start()
    
    def start_listener(self):
        # This function will run the listener in the background
        with Listener(on_press=self.key_press, on_release=self.key_release) as listener:
            listener.join()

    def key_press(self, k):
        # check if the key is a number key
        if hasattr(k, 'char'):
            # by self.robot_num, [0, 1, ..., robot_num]
            list_of_number = [str(i+1) for i in range(self.robot_num)] 
            if k.char in list_of_number:
                self.control_robot_id = int(k.char) -  1
                logger.debug('control_robot_id:  %s', self.control_robot_id + 1)
        else:
            # check if the key is an arrow key
            if k == Key.left:
                self.h = [-1, 0]
            elif k == Key.right:
                self.h = [1, 0]
            # elif k == Key.up:
            #     self.h = [0, 1]
            # elif k== Key.down:
            #     self.h = [0, -1]
            elif k == Key.up:  # switch up and down for push T env
                self.h = [0, -1]
            elif k== Key.down:
                self.h = [0, 1]
            elif k == Key.space:
                self.restart = True
  
    def key_release(self, k):
        if k in (Key.left, Key.right, Key.up, Key.down):
            logger.debug("key release")
            # self.h = self.h_null

    def get_h(self):
        logger.debug('self.control_robot_id:  %s  self.h:  %s', self.control_robot_id, self.h)
        # transfrom the robot_id + h to the real h, whose dimension is 2 * robot_num
        received_h = [0] * (2 * self.robot_num)
        received_h[2 * self.control_robot_id] = self.h[0]
        received_h[2 * self.control_robot_id + 1] = self.h[1]
        self.h = self.h_null
        return received_h

    def ask_for_done(self):
        done = self.restart
        self.restart = False
        return done
