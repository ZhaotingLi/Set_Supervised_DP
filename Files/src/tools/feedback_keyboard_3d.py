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


class Feedback_keyboard_3d:
    def __init__(self):
        self.h = [0, 0, 0]
        self.h_null = [0, 0, 0]
        self.h_direction = None
        self.restart = False
        self.evaluation = False
        self.model_training = False

        self.pause_flag = False

        # Starting the listener
        listener_thread = threading.Thread(target=self.start_listener)
        listener_thread.start()

        self.skip_evaluation = None
    
    def start_listener(self):
        # This function will run the listener in the background
        with Listener(on_press=self.key_press, on_release=self.key_release) as listener:
            listener.join()

    def key_press(self, k):
        if hasattr(k, 'char'):  # Check if the key has 'char' attribute
            logger.info(f'alphanumeric key {k.char} pressed')
            if k.char == 'x':
                self.h_direction = [1, 0, 0]
            elif k.char == 'y':
                self.h_direction = [0, 1, 0]
            elif k.char == 'z':
                self.h_direction = [0, 0, 1]
            
            if k.char =='s':
                self.skip_evaluation = True
            elif k.char == 'q':
                self.skip_evaluation = False
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

    def ask_whether_skip_evaluation(self):
        flag= self.skip_evaluation
        self.skip_evaluation = None
        # print("fix_ee_flag: ", flag)
        return flag

    def ask_for_done(self):
        done = self.restart
        self.restart = False
        return done

    def pause(self):
        return None
