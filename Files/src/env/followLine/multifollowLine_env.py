import logging

logger = logging.getLogger(__name__)

import gym
import numpy as np
import pygame
from gym.spaces import Box
from env.followLine.followLine_env import LineFollowerEnv

class MultiLineFollowerEnv(LineFollowerEnv):
    def __init__(self, robot_num=2, line_length=200, line_distance=50, line_start=(0, 0)):
        # Call the parent class's constructor
        super().__init__()

        # Number of balls and distance between lines
        self.n_balls = robot_num
        self.line_length = line_length
        self.line_distance = line_distance

        # Define action and observation space for all balls
        self.action_space = Box(low=-1, high=1, shape=(self.n_balls * self.dim_a,), dtype=np.float32)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.n_balls * self.dim_o,), dtype=np.float32)
        
        # Define lines and end effectors' poses
        self.lines = self.initialize_lines()
        self.ee_poses = [np.zeros(2) for _ in range(self.n_balls)]

        # Initialize the state
        self.state = None
    
    def initialize_lines(self):
        # This method initializes lines within the window bounds with random start points
        lines = []
        line_angle = np.random.uniform(0, np.pi / 2)  # Random angle between 0 and 90 degrees
        valid_line = False
        while not valid_line:
            while len(lines) < self.n_balls:
                line_id = len(lines)
                # Random start point within the bounds ensuring enough space for the line length
                if line_id < 2:  # for the first two lines, sample random start points
                    buffer = self.line_length / np.sqrt(2)  # Ensures that there's room for the line
                    start_x = np.random.uniform(buffer, self.screen_width - buffer)
                    start_y = np.random.uniform(buffer, self.screen_height - buffer)
                else: # for other lines, the start point is on the line passing through line1_start and line2_start
                    line1_start = lines[0][0]
                    line2_start = lines[1][0]
                    start_id = line2_start + 1 * (-line1_start + line2_start)
                    start_x = start_id[0]
                    start_y = start_id[1]
                    logger.debug('%s %s %s %s %s %s %s %s %s %s', "line1_start: ", line1_start, "line2_start: ", line2_start, "start_id: ", start_id, "start_x: ", start_x, "start_y: ", start_y)
                    
                logger.debug('%s %s %s %s %s %s', "line_id: ", line_id, "start_x: ", start_x, "start_y: ", start_y)
                # Keeping the same direction (45-degree angle) and length for all lines
                end_x = start_x + self.line_length * np.cos(line_angle)  # 45 degrees
                end_y = start_y - self.line_length * np.sin(line_angle)  # 45 degrees

                # Create a new candidate line
                new_line = (np.array([start_x, start_y]), np.array([end_x, end_y]))

                # Check if the new line intersects with any existing line
                # if line in the window and not intersecting with any other line, add it to the list of lines
                if start_x > 0 and start_x < self.screen_width and start_y > 0 and start_y < self.screen_height and \
                    end_x > 0 and end_x < self.screen_width and end_y > 0 and end_y < self.screen_height and \
                    not any(self.line_intersection(new_line, line) for line in lines):
                    valid_line = True
                    lines.append(new_line)
                else:
                    valid_line = False
                    lines = []

        return lines

    # Return true if line segments AB and CD intersect, or too close to each other
    def line_intersection(self, line1, line2):
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        
        def intersect(A, B, C, D):
            return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
        
        A, B = line1
        C, D = line2
        return intersect(A, B, C, D) or np.linalg.norm(A - C) < self.line_distance or np.linalg.norm(A - D) < self.line_distance or np.linalg.norm(B - C) < self.line_distance or np.linalg.norm(B - D) < self.line_distance


    def reset(self):
        # Reset the environment state for all balls
        self.current_step = 0
        self.lines = self.initialize_lines()
        for i in range(self.n_balls):
            line_start_i, line_end_i = self.lines[i]
            self.ee_poses[i] = np.array(line_start_i, dtype=np.float32)
        self.state = self.get_state()
        info = {}
        info['success'] = False
        return self.state, info

    def get_state(self):  # 7 * n + n -1
        # Generate the state for all balls
        state = []
        for i in range(self.n_balls):
            line_start_i, line_end_i = self.lines[i]
            position = self.ee_poses[i]
            projection = line_start_i + np.dot(position - line_start_i, line_end_i - line_start_i) / np.dot(line_end_i - line_start_i, line_end_i - line_start_i) * (line_end_i - line_start_i)
            distance_to_line = np.linalg.norm(projection - position)
            state.extend((position - line_end_i) / 100)
            state.extend((line_start_i - position) / 100)
            state.extend((line_end_i - line_start_i) / 100)
            state.append(distance_to_line / 50)
        # we want the ball at the same position of its corresponding line, for every step
        # also include the error between the distance of the balls, and the distance of the lines
        for i in range(self.n_balls - 1):
            ball_i = self.ee_poses[i]
            ball_i_1 = self.ee_poses[i + 1]
            line_i = self.lines[i]
            line_i_1 = self.lines[i + 1]
            distance_ball = np.linalg.norm(ball_i - ball_i_1)
            distance_line = np.linalg.norm(line_i[0] - line_i_1[0])
            state.append((distance_ball - distance_line) / 100)
        return np.array(state)

    def step(self, action):
        # Apply action to all balls
        rewards = []
        dones = []
        successes = []
        for i in range(self.n_balls):
            position = self.ee_poses[i] + action[i*2:(i+1)*2]
            self.ee_poses[i] = self.apply_bounds(position)
            self.state = self.get_state()

            # Calculate reward and done for each ball
            reward_i = self._calculate_reward(self.ee_poses[i], self.lines[i][0], self.lines[i][1])
            rewards.append(reward_i)
            done_i, success_i = self._check_done(self.ee_poses[i], self.lines[i][0], self.lines[i][1])
            dones.append(done_i)
            successes.append(success_i)

        # Check if all balls are done
        done = all(dones)

        # with_in_line = True
        # for i in range(self.n_balls):
        #     line_start_i, line_end_i = self.lines[i]
        #     position = self.ee_poses[i]
        #     projection = line_start_i + np.dot(position - line_start_i, line_end_i - line_start_i) / np.dot(line_end_i - line_start_i, line_end_i - line_start_i) * (line_end_i - line_start_i)
        #     distance_to_line = np.linalg.norm(projection - position)
        #     # print("i: ", i, "distance_to_line: ", distance_to_line)
        #     if distance_to_line > 15:
        #         with_in_line = False
        #         done = True

        self.current_step += 1
        info = {'success': all(successes)}
        return self.state, sum(rewards), done, False, info

    def render(self, selected_ball=None):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption('Line Follower Env')

        # Define colors for balls, start points, and end points
        ball_colors = [
            pygame.Color('red'),
            pygame.Color('green'),
            pygame.Color('blue'),
            pygame.Color('yellow'),
            pygame.Color('purple'),
            # Add more colors if there are more than 5 balls
        ]
        # start_point_color = pygame.Color('black')  # Color for all start points
        end_point_color = pygame.Color('black')  # Color for all end points

        self.screen.fill((255, 255, 255))  # Fill the screen with a white background

        # Define a box size for the start and end points
        point_box_size = 20

        # Draw each line, start point, end point, and corresponding ball
        for i, (ball, line) in enumerate(zip(self.ee_poses, self.lines)):
            ball_color = ball_colors[i % len(ball_colors)]  # Cycle through the ball colors
            
            # Draw the line
            pygame.draw.line(self.screen, ball_color, line[0], line[1], 8)
            
            # Draw the start point as a rectangle
            start_rect = pygame.Rect(line[0][0] - point_box_size // 2, line[0][1] - point_box_size // 2, point_box_size, point_box_size)
            pygame.draw.rect(self.screen, ball_color, start_rect)
            
            # Draw the end point as a rectangle
            end_rect = pygame.Rect(line[1][0] - point_box_size // 2, line[1][1] - point_box_size // 2, point_box_size, point_box_size)
            pygame.draw.rect(self.screen, end_point_color, end_rect)
            
            # Draw the ball at its current position
            pygame.draw.circle(self.screen, ball_color, (int(ball[0]), int(ball[1])), 10)

        # Draw the selected ball with another larger circle, no color in the middle
        selected_ball_color = pygame.Color('black')
        if selected_ball is not None:
            pygame.draw.circle(self.screen, selected_ball_color, (int(self.ee_poses[selected_ball][0]), int(self.ee_poses[selected_ball][1])), 15, 3)
        # Update the full display Surface to the screen
        pygame.display.flip()



    def to_pygame_coords(self, pos):
        # Convert the position to PyGame coordinates (the origin is at the top-left corner in PyGame)
        return int(pos[0]), int(self.screen_height - pos[1])

    def apply_bounds(self, position):
        # Keep the ball within the screen bounds
        position[0] = np.clip(position[0], 0, self.screen_width)
        position[1] = np.clip(position[1], 0, self.screen_height)
        return position

    def _calculate_reward(self, position, line_start, line_end):
        # Implement the reward function based on the position of the ball and its respective line
        projection = line_start + np.dot(position - line_start, line_end - line_start) / np.dot(line_end - line_start, line_end - line_start) * (line_end - line_start)
        distance_to_line = np.linalg.norm(projection - position)
        reward = -distance_to_line  # Negative reward for being further from the line
        return reward
    

    # control policy for the balls to follow the lines, while maintaining the distance every ball traveled to be the same
    # (1) if the balls are all one its line, then the ball with large traveled distance will slow down, and the ball with small traveled distance will speed up
    # (2) if one ball is not on its line, then all the balls will move towards the start point of its line
    # (3) based on control_policy_oneball(self, position, line_start, line_end) at followLine_env.py
    def control_policy(self, state):
        # print("state: ", state)
        action = np.zeros(self.n_balls * self.dim_a)
        with_in_line = True
        for i in range(self.n_balls):
            line_start_i, line_end_i = self.lines[i]
            position = self.ee_poses[i]
            projection = line_start_i + np.dot(position - line_start_i, line_end_i - line_start_i) / np.dot(line_end_i - line_start_i, line_end_i - line_start_i) * (line_end_i - line_start_i)
            distance_to_line = np.linalg.norm(projection - position)
            # print("i: ", i, "distance_to_line: ", distance_to_line)
            # if distance_to_line < 15:
            if distance_to_line < 31:

                action_i = self.control_policy_oneball(position, line_start_i, line_end_i)
                action[i*2:(i+1)*2] = action_i
            else:
                with_in_line = False
                
        if with_in_line:
            # check the traveled distance of each ball
            distance_list = []
            for i in range(self.n_balls):
                line_start_i, line_end_i = self.lines[i]
                position = self.ee_poses[i]
                distance = np.linalg.norm(position - line_start_i)
                distance_list.append(distance)
            # check if the difference between the max and min distance is larger than 10
            if max(distance_list) - min(distance_list) > 10:
                # if so, change the speed of each ball according to (its traveled distance - average traveled distance)
                average_distance = sum(distance_list) / len(distance_list)
                for i in range(self.n_balls):
                    distance = distance_list[i]
                    speed = 1 + (-distance + average_distance) / 20
                    action[i*2:(i+1)*2] = action[i*2:(i+1)*2] * speed

            return action
        else:
            for i in range(self.n_balls):
                line_start_i, line_end_i = self.lines[i]
                position = self.ee_poses[i]
                action_i = line_start_i - position
                action_i /= np.linalg.norm(action_i) 
                action[i*2:(i+1)*2] = action_i
            return action

if __name__ == '__main__':
    # Test the environment
    env = MultiLineFollowerEnv()
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = env.action_space.sample()  # Replace with your controller
        state, reward, done, info = env.step(action)
        total_reward += reward
        env.render()

    logger.debug(f"Total Reward: {total_reward}")
