import logging

logger = logging.getLogger(__name__)

import gym
import numpy as np
import pygame

class DesiredA_ToyEnv(gym.Env):
    def __init__(self, ):
        super(DesiredA_ToyEnv, self).__init__()
        self.dim_a = 2
        # self.dim_o = 10
        self.dim_o = 10
        # Define action and observation space
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.dim_a,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.dim_o,), dtype=np.float32)
        

        # Max steps before episode termination
        self.current_step = 0

        # Initial state
        self.screen_width = 600
        self.screen_height = 600
        self.screen = None
        self.state = None
        self.max_steps = 500
        self.index = 0

        '''Desired action space'''
        # defined as a circle, r = 0.4, center is (0, 0)
        self.desiredA_center = np.array([0, 0])
        self.desiredA_radius = 0.4

    def withinDesiredA(self, action):
        return np.linalg.norm(action - self.desiredA_center) < self.desiredA_radius

    def reset(self):
        # Reset the environment state
        self.current_step = 0
        # randomize the start point, and the end point, ranging from screen_width and screen_height
        
        # index = np.random.randint(0, self.dim_o)
        index = 0
        one_hot_vector = np.zeros(self.dim_o)
        one_hot_vector[index] = 0.1
        self.index = 0

        self.state = one_hot_vector
        logger.debug('%s %s', "self.state: ", self.state)
        info = {}
        info['success'] = False
        return self.state, info


    def step(self, action):
        # Apply action
        # if action is outside the desired action space, the env stops with failure
        # index = np.random.randint(0, self.dim_o)
        index = self.index
        logger.debug('%s %s', "self.index: ", self.index)
        self.index = self.index + 1
        if self.index > self.dim_o-1:
            self.index = 0

        one_hot_vector = np.zeros(self.dim_o)
        one_hot_vector[index] = 0.1

        self.state = one_hot_vector
        


        # Check if end-effector has reached the end of the line or max steps
        # done = np.linalg.norm(position - self.line_end) < 0.1 or self.current_step >= self.max_steps
        self.current_step += 1

        # Optional: Add info dictionary
        info = {}
        reward = -1 if not self.withinDesiredA(action) else 1
        info['success'] = False
        done = self.current_step >= self.max_steps

        return self.state, reward, done, False, info

    # one optimal action with Gaussian noise
    def control_policy(self, state, pre_correction_action):
        self.desiredA_radius = 0.4
        if not self.withinDesiredA(pre_correction_action):
            # vector_a_negative_to_original = pre_correction_action - self.desiredA_center
            # normlized_vector_a_negative_to_original = self.desiredA_radius * vector_a_negative_to_original / np.linalg.norm(vector_a_negative_to_original)
            # post_correction_action = 2 * normlized_vector_a_negative_to_original - pre_correction_action
            # post_correction_action = self.desiredA_center
            post_correction_action = self.desiredA_center + np.random.normal(loc=0.0, scale=0.15, size=2)
        else:
            post_correction_action = pre_correction_action

        if not self.withinDesiredA(post_correction_action):
            post_correction_action = pre_correction_action

        return post_correction_action

    # def control_policy(self, state, pre_correction_action):
    #     if not self.withinDesiredA(pre_correction_action):
    #         vector_a_negative_to_original = pre_correction_action - self.desiredA_center
    #         normlized_vector_a_negative_to_original = self.desiredA_radius * vector_a_negative_to_original / np.linalg.norm(vector_a_negative_to_original)
    #         post_correction_action = 2 * normlized_vector_a_negative_to_original - pre_correction_action
    #     else:
    #         post_correction_action = pre_correction_action

    #     if not self.withinDesiredA(post_correction_action):
    #         post_correction_action = pre_correction_action

    #     return post_correction_action

    # def control_policy(self, state, pre_correction_action):
    #     if not self.withinDesiredA(pre_correction_action):
    #         def random_point_in_circle(center, radius):
    #             r = radius * np.sqrt(np.random.uniform(0, 1))  # Random radius scaled for uniform distribution
    #             theta = np.random.uniform(0, 2 * np.pi)  # Random angle
    #             x = center[0] + r * np.cos(theta)
    #             y = center[1] + r * np.sin(theta)
    #             return np.array([x, y])

    #         # Generate three random points within the circle
    #         points = [random_point_in_circle(self.desiredA_center, self.desiredA_radius) for _ in range(3)]

    #         # Calculate the distances to the target point
    #         distances = [np.linalg.norm(point - pre_correction_action) for point in points]

    #         # Find the closest point
    #         post_correction_action = points[np.argmin(distances)]
    #     else:
    #         post_correction_action = pre_correction_action

    #     return post_correction_action


    # only one-dim
    # def control_policy(self, state, pre_correction_action):
    #     if not self.withinDesiredA(pre_correction_action):
    #         vector_a_negative_to_original = pre_correction_action - self.desiredA_center
    #         normlized_vector_a_negative_to_original = self.desiredA_radius * vector_a_negative_to_original / np.linalg.norm(vector_a_negative_to_original)
    #         post_correction_action = 2 * normlized_vector_a_negative_to_original - pre_correction_action
            
    #         difference = post_correction_action - pre_correction_action
    #         if np.linalg.norm(difference) > 0.05:
    #             if abs(difference[0]) > abs(difference[1]):
    #                 post_correction_action[1] = pre_correction_action[1]
    #             else:
    #                 post_correction_action[0] = pre_correction_action[0]
    #     else:
    #         post_correction_action = pre_correction_action

    #     return post_correction_action

    def render(self, mode='human'):
        screen_width = self.screen_width
        screen_height = self.screen_height
        
        # if self.screen is None:
        #     pygame.init()
        #     self.screen = pygame.display.set_mode((screen_width, screen_height))
        #     pygame.display.set_caption('Line Follower Env')
        
        # # Set the background to white
        # self.screen.fill((255, 255, 255))
        
        # # Convert to Pygame coordinates (y-axis is flipped)
        # def to_pygame_coords(pos):
        #     # return int(pos[0]), screen_height - int(pos[1])
        #     return int(pos[0]), int(pos[1])

        # Draw the line
        # start_pos = to_pygame_coords(self.line_start)
        # end_pos = to_pygame_coords(self.line_end)
        # pygame.draw.line(self.screen, (0, 0, 0), start_pos, end_pos, 8)
        
        # # Draw the start point as a blue box
        # #start_rect = pygame.Rect(start_pos[0] - 5, start_pos[1] - 5, 10, 10)
        # start_rect_size = 30
        # start_rect = pygame.Rect(start_pos[0] - start_rect_size / 2, start_pos[1] - start_rect_size / 2, start_rect_size, start_rect_size)
        # pygame.draw.rect(self.screen, (0, 0, 255), start_rect)
        
        # # Draw the end point as a black box
        # end_rect_size = 30
        # end_rect = pygame.Rect(end_pos[0] - end_rect_size / 2, end_pos[1] - end_rect_size / 2, end_rect_size, end_rect_size)
        # # end_rect = pygame.Rect(end_pos[0] - 5, end_pos[1] - 5, 10, 10)
        # pygame.draw.rect(self.screen, (0, 0, 0), end_rect)
        
        # # Draw the end effector as a blue circle
        # end_effctor_size = 15
        # effector_pos = to_pygame_coords((self.ee_pose[0], self.ee_pose[1]))
        # pygame.draw.circle(self.screen, (0, 0, 255), effector_pos, end_effctor_size)
        
        # pygame.display.flip()

        # # Close the window if the user clicks the window close button
        # for event in pygame.event.get():
        #     if event.type == pygame.QUIT:
        #         pygame.quit()
        #         exit()
    
    def save_image(self, save_path):
        # Save the screen to an image file
        image_filename = save_path
        pygame.image.save(self.screen, image_filename)
        return None
    
    def close(self):
        pygame.quit()

    
            


  
