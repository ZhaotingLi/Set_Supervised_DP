import logging
from pyglet.window import key

logger = logging.getLogger(__name__)


"""
Class that obtains the human feedback from the computer's keyboard.
"""


class Feedback:
    def __init__(self, env):
        key_type = '1'
        if key_type == '1':
            env.unwrapped.viewer.window.on_key_press = self.key_press
            env.unwrapped.viewer.window.on_key_release = self.key_release
        elif key_type == '2':
            env.unwrapped.window.on_key_press = self.key_press
            env.unwrapped.window.on_key_release = self.key_release
        else:
            logger.warning('No valid feedback type selected!')
            exit()

        self.h_null = 0
        self.h_up = 0
        self.h_down = 0
        self.h_right = 1
        self.h_left = -1
        self.h = self.h_null  # Human correction
        self.restart = False
        self.evaluation = False
        self.model_training = False


    def key_press(self, k, mod):
        if k == key.A:
            self.h = self.h_right
        if k == key.D:
            self.h = self.h_left
        if k == key.SPACE:
            self.restart = True
        if k == key.LEFT:
            self.h = self.h_left
        if k == key.RIGHT:
            self.h = self.h_right
        if k == key.NUM_1:
            self.h = self.h_left
        if k == key.NUM_3:
            self.h = self.h_right
        if k == key.NUM_6:
            self.h = self.h_left
        if k == key.NUM_4:
            self.h = self.h_right
        if k == key.UP:
            self.h = self.h_up
        if k == key.DOWN:
            self.h = self.h_down
        if k == key.E:
            self.evaluation = not self.evaluation
            if self.evaluation:
                logger.info('EVALUATION STARTED')
            else:
                logger.info('EVALUATION STOPPED')
        if k == key.S:
            self.model_training = not self.model_training
            if not self.model_training:
                logger.info('MODEL TRAINING STOPPED')
            else:
                logger.info('MODEL TRAINING STARTED')

    def key_release(self, k, mod):
        if k == key.LEFT or k == key.RIGHT or k == key.UP or k == key.DOWN \
                or k == key.A or k == key.D or k == key.NUM_1 or k == key.NUM_3 or k == key.NUM_4 or k == key.NUM_6:
            self.h = self.h_null

    def get_h(self):
        return [self.h]

    def ask_for_done(self):
        done = self.restart
        self.restart = False
        return done
