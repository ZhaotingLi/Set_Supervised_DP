import logging

logger = logging.getLogger(__name__)

import gym
import numpy as np
import pygame


class LineFollowerEnv(gym.Env):
    def __init__(self, line_start=(100, 300), line_end=(480, 300), random_lines = True):
        super(LineFollowerEnv, self).__init__()
        self.dim_a = 2
        self.dim_o = 5
        # Define action and observation space
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.dim_a,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.dim_o,), dtype=np.float32)
        
        # Define line start and end points
        self.line_start = np.array(line_start, dtype=np.float32)
        self.line_end = np.array(line_end, dtype=np.float32)
        self.line_vector = self.line_end - self.line_start
        self.ee_pose = np.array([0, 0], dtype=np.float32)

        # Max steps before episode termination
        self.current_step = 0

        # Initial state
        self.screen_width = 600
        self.screen_height = 600
        self.bound_distance = 20
        self.screen = None
        self.state = None

        self.random_lines = random_lines

    def reset(self):

        # Reset the environment state
        self.current_step = 0
        # randomize the start point, and the end point, ranging from screen_width and screen_height
        if self.random_lines:   
            self.line_start = np.array([np.random.randint(self.bound_distance, self.screen_width-self.bound_distance), np.random.randint(self.bound_distance, self.screen_height-self.bound_distance)], dtype=np.float32)
            self.line_end = np.array([np.random.randint(self.bound_distance, self.screen_width-self.bound_distance), np.random.randint(self.bound_distance, self.screen_height-self.bound_distance)], dtype=np.float32)
        self.line_vector = self.line_end - self.line_start

        # self.ee_pose = np.array([np.random.randint(self.bound_distance, self.screen_width-self.bound_distance), np.random.randint(self.bound_distance, self.screen_height-self.bound_distance)], dtype=np.float32)
        self.ee_pose = self.line_start
        logger.debug('%s %s', "self.ee_pose: ", self.ee_pose)
        self.state = self.get_state()
        expanded_obs = np.expand_dims(self.state, axis=0)
        logger.debug('%s %s', "self.state: ", self.state)
        info = {}
        info['success'] = False
        return expanded_obs, info

    def get_state(self):
        # also add the distance to the line
        position = self.ee_pose
        projection = self.line_start + np.dot(position - self.line_start, self.line_vector) / np.dot(self.line_vector, self.line_vector) * self.line_vector
        distance_to_line = np.linalg.norm(projection - position)
        state = np.array(
            tuple((self.ee_pose - self.line_end)/ 100) \
            + tuple((self.line_start - self.line_end)/ 100) \
             + tuple([distance_to_line / 30]) )
        
        # state = np.array(
        #     tuple((self.ee_pose - self.line_end)/ 100) \
        #     + tuple(self.line_start - self.line_end/ 600) \
        #     + tuple(self.line_end / 600)
        #      + tuple([distance_to_line / 15]) )
        return state

    def step(self, action):
        # Apply action
        position = self.ee_pose+ action
        # check if the position is out of bound [self.bound_distance, self.screen_width-self.bound_distance]
        if position[0] < self.bound_distance:
            position[0] = self.bound_distance
        elif position[0] > self.screen_width-self.bound_distance:
            position[0] = self.screen_width-self.bound_distance
        if position[1] < self.bound_distance:
            position[1] = self.bound_distance
        elif position[1] > self.screen_height-self.bound_distance:
            position[1] = self.screen_height-self.bound_distance
        
        self.ee_pose = position
        self.state = self.get_state() 
        expanded_obs = np.expand_dims(self.state, axis=0)
        

        # Calculate reward
        reward = self._calculate_reward(position)

        # Check if end-effector has reached the end of the line or max steps
        # done = np.linalg.norm(position - self.line_end) < 0.1 or self.current_step >= self.max_steps
        self.current_step += 1

        # Optional: Add info dictionary
        info = {}
        done, info['success']= self._check_done(self.ee_pose, self.line_start, self.line_end)
        return expanded_obs, reward, done, False, info

    def render(self, mode='human'):
        screen_width = self.screen_width
        screen_height = self.screen_height
        
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((screen_width, screen_height))
            pygame.display.set_caption('Line Follower Env')
        
        # Set the background to white
        self.screen.fill((255, 255, 255))
        
        # Convert to Pygame coordinates (y-axis is flipped)
        def to_pygame_coords(pos):
            # return int(pos[0]), screen_height - int(pos[1])
            return int(pos[0]), int(pos[1])

        # Draw the line
        start_pos = to_pygame_coords(self.line_start)
        end_pos = to_pygame_coords(self.line_end)
        pygame.draw.line(self.screen, (0, 0, 0), start_pos, end_pos, 8)
        
        # Draw the start point as a blue box
        #start_rect = pygame.Rect(start_pos[0] - 5, start_pos[1] - 5, 10, 10)
        start_rect_size = 30
        start_rect = pygame.Rect(start_pos[0] - start_rect_size / 2, start_pos[1] - start_rect_size / 2, start_rect_size, start_rect_size)
        pygame.draw.rect(self.screen, (0, 0, 255), start_rect)
        
        # Draw the end point as a black box
        end_rect_size = 30
        end_rect = pygame.Rect(end_pos[0] - end_rect_size / 2, end_pos[1] - end_rect_size / 2, end_rect_size, end_rect_size)
        # end_rect = pygame.Rect(end_pos[0] - 5, end_pos[1] - 5, 10, 10)
        pygame.draw.rect(self.screen, (0, 0, 0), end_rect)
        
        # Draw the end effector as a blue circle
        end_effctor_size = 15
        effector_pos = to_pygame_coords((self.ee_pose[0], self.ee_pose[1]))
        pygame.draw.circle(self.screen, (0, 0, 255), effector_pos, end_effctor_size)
        
        pygame.display.flip()

        # Close the window if the user clicks the window close button
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
    
    def save_image(self, save_path):
        # Save the screen to an image file
        image_filename = save_path
        pygame.image.save(self.screen, image_filename)
        return None
    
    def close(self):
        pygame.quit()

    def _calculate_reward(self, position):
        # Project position onto line
        projection = self.line_start + np.dot(position - self.line_start, self.line_vector) / np.dot(self.line_vector, self.line_vector) * self.line_vector
        distance_to_line = np.linalg.norm(projection - position)

        # Penalize distance to line
        reward = -distance_to_line

        # Additional reward for moving towards the end point
        if np.allclose(position, projection, atol=0.1):  # If the position is close to the line
            reward += np.dot(position - self.line_start, self.line_vector) / np.linalg.norm(self.line_vector)

        return reward
    
    def _check_done(self, position, line_start=None, line_end=None):
        # Check if end-effector has reached the end of the line or max steps
        done = np.linalg.norm(position - line_end) < 15 
        line_vector = line_end - line_start
        if_success_ = done
        # print("if_success_: ", if_success_)
        if done:
            return done, if_success_
        else: # check if the ball going too away from the line
            projection = line_start + np.dot(position - line_start, line_vector) / np.dot(line_vector, line_vector) * line_vector
            distance_to_line = np.linalg.norm(projection - position)
            # print("distance_to_line: ", distance_to_line)
            if distance_to_line > 30:
                done = True
            return done, if_success_
            

    def intersection_point_closest_to_end(self, line_start, line_end, point, radius):
        """
        Return the point on the line segment [line_start, line_end] that lies exactly at `radius` distance from `point`,
        and is closest to `line_end`. If `line_end` is already within or on the circle, return it directly.

        :param line_start: Tuple (x, y), start of the line segment
        :param line_end: Tuple (x, y), end of the line segment
        :param point: Tuple (x, y), center of the circle
        :param radius: Float, radius of the circle
        :return: Tuple (x, y): the selected point, or None if no valid intersection on the segment
        """
        p1 = np.array(line_start, dtype=np.float64)
        p2 = np.array(line_end, dtype=np.float64)
        center = np.array(point, dtype=np.float64)

        # Check if line_end is inside or on the circle
        if np.linalg.norm(p2 - center) <= radius:
            return tuple(p2)

        # Direction vector of the line
        d = p2 - p1
        f = p1 - center

        a = np.dot(d, d)
        b = 2 * np.dot(f, d)
        c = np.dot(f, f) - radius**2

        discriminant = b**2 - 4 * a * c

        if discriminant < 0:
            # return None  # No intersection
            return self.closest_point_on_line(line_start, line_end, point)

        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2 * a)
        t2 = (-b + discriminant) / (2 * a)

        candidates = []
        for t in [t1, t2]:
            if 0 <= t <= 1:
                point_on_line = p1 + t * d
                candidates.append(point_on_line)

        if not candidates:
            return self.closest_point_on_line(line_start, line_end, point)

        # Return the candidate closest to line_end
        closest = min(candidates, key=lambda p: np.linalg.norm(p - p2))
        return tuple(closest)


    def closest_point_on_line(self, line_start, line_end, point):
        """
        Calculate the closest point on a line segment to a given point.

        :param line_start: Start point of the line segment (x, y)
        :param line_end: End point of the line segment (x, y)
        :param point: The point (x, y) to which you want to find the closest point on the line segment
        :return: The closest point on the line segment (x, y)
        """
        line_vec = np.array(line_end) - np.array(line_start)
        point_vec = np.array(point) - np.array(line_start)
        line_len = np.linalg.norm(line_vec)
        line_unitvec = line_vec / line_len
        proj_length = np.dot(point_vec, line_unitvec)
        
        if proj_length < 0:
            return line_start
        elif proj_length > line_len:
            return line_end
        else:
            return np.array(line_start) + line_unitvec * proj_length

    def control_policy_oneball(self, position, line_start, line_end):
        """
        Simple control policy to move the end effector (ball) towards or along a line segment.

        :param state: Current state (position) of the end effector (x, y)
        :param line_start: Start point of the line segment (x, y)
        :param line_end: End point of the line segment (x, y)
        :param threshold: Distance threshold to decide motion behavior
        :return: The action (dx, dy) representing the movement direction
        """
        threshold = 10
        ee_pose_ = position
        closest_point = self.closest_point_on_line(line_start, line_end, ee_pose_)
        distance_to_line = np.linalg.norm(np.array(ee_pose_) - closest_point)
        
        if distance_to_line < threshold:
            # Move along the line direction
            direction = np.array(line_end) - np.array(line_start)
            direction /= np.linalg.norm(direction)  # Normalize the direction vector
        else:
            # Move towards the closest point on the line
            direction = closest_point - np.array(ee_pose_)
            if np.linalg.norm(direction) != 0:
                direction /= np.linalg.norm(direction)  # Normalize the direction vector

        # Assuming a fixed step size for the end effector
        step_size = 1  # Adjust step size as necessary for your application
        action = direction * step_size
        return action
    
    def control_policy(self, state):
        """
        Simple control policy to move the end effector (ball) towards or along a line segment.

        :param state: Current state (position) of the end effector (x, y)
        :param line_start: Start point of the line segment (x, y)
        :param line_end: End point of the line segment (x, y)
        :param threshold: Distance threshold to decide motion behavior
        :return: The action (dx, dy) representing the movement direction
        """
        line_start = self.line_start
        line_end = self.line_end
        threshold = 10
        ee_pose_ = self.ee_pose
        # closest_point = self.closest_point_on_line(line_start, line_end, ee_pose_)
        closest_point = self.intersection_point_closest_to_end(line_start=line_start, line_end=line_end,
                                                               point=ee_pose_, radius=threshold)
        distance_to_line = np.linalg.norm(np.array(ee_pose_) - closest_point)
        
        # if distance_to_line < threshold:
        #     # Move along the line direction
        #     direction = np.array(line_end) - np.array(line_start)
        #     direction /= np.linalg.norm(direction)  # Normalize the direction vector
        # else:
        #     # Move towards the closest point on the line
        #     direction = closest_point - np.array(ee_pose_)
        #     if np.linalg.norm(direction) != 0:
        #         direction /= np.linalg.norm(direction)  # Normalize the direction vector

        direction = closest_point - np.array(ee_pose_)
        if np.linalg.norm(direction) != 0:
            direction /= np.linalg.norm(direction)  # Normalize the direction vector

        # Assuming a fixed step size for the end effector
        step_size = 0.5  # Adjust step size as necessary for your application
        action = direction * step_size
        return action

# Test the environment
# env = LineFollowerEnv()
# state = env.reset()
# done = False
# total_reward = 0

# while not done:
#     action = env.action_space.sample()  # Replace with your controller
#     state, reward, done, info = env.step(action)
#     total_reward += reward
#     env.render()

# print(f"Total Reward: {total_reward}")
