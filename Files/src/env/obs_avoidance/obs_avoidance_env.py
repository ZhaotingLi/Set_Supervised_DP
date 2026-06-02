import logging

logger = logging.getLogger(__name__)

import gym
from gym import spaces

import collections
import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
from pymunk.vec2d import Vec2d
import shapely.geometry as sg
import cv2
import skimage.transform as st
from env.pusht.pymunk_override import DrawOptions

import heapq # for a star
import math

def pymunk_to_shapely(body, shapes):
    geoms = list()
    for shape in shapes:
        if isinstance(shape, pymunk.shapes.Poly):
            verts = [body.local_to_world(v) for v in shape.get_vertices()]
            verts += [verts[0]]
            geoms.append(sg.Polygon(verts))
        elif isinstance(shape, pymunk.shapes.Circle):
            center = body.local_to_world(shape.offset)
            radius = shape.radius
            circle = sg.Point(center).buffer(radius)
            geoms.append(circle)
        else:
            raise RuntimeError(f'Unsupported shape type {type(shape)}')
    geom = sg.MultiPolygon(geoms)
    return geom


class ObstacleAvoidanceEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 10}
    reward_range = (0., 1.)

    def __init__(self,
            legacy=False, 
            block_cog=None, damping=None,
            render_action=True,
            render_size=96,
            reset_to_state=None,
            obstacle_width=10,
            obstacle_height=100
        ):
        self._seed = None
        self.seed()
        self.window_size = ws = 512  # The size of the PyGame window
        self.render_size = render_size
        self.sim_hz = 100
        # Local controller params.
        self.k_p, self.k_v = 100, 20    # PD control.z
        self.control_hz = self.metadata['video.frames_per_second']
        # legacy set_state for data compatibility
        self.legacy = legacy

        # agent_pos, obstacle_pos, goal_pos, goal_angle
        self.observation_space = spaces.Box(
            low=np.array([0,0,0,0,0], dtype=np.float64),
            high=np.array([ws,ws,ws,ws,np.pi*2], dtype=np.float64),
            shape=(5,),
            dtype=np.float64
        )

        # positional goal for agent
        self.action_space = spaces.Box(
            low=np.array([0,0], dtype=np.float64),
            high=np.array([ws,ws], dtype=np.float64),
            shape=(2,),
            dtype=np.float64
        )

        self.block_cog = block_cog
        self.damping = damping
        self.render_action = render_action

        self.obstacle_width = obstacle_width
        self.obstacle_height = obstacle_height

        self.window = None
        self.clock = None
        self.screen = None

        self.space = None
        self.teleop = None
        self.render_buffer = None
        self.latest_action = None
        self.reset_to_state = reset_to_state

        self.path = None
        self.last_path = None
    
    def reset(self):
        self.path = None
        self.last_path = None
        seed = self._seed
        self._setup()
        if self.block_cog is not None:
            self.block.center_of_gravity = self.block_cog
        if self.damping is not None:
            self.space.damping = self.damping
        
        state = self.reset_to_state
        if state is None:
            rs = np.random.RandomState(seed=seed)
            state = np.array([
                rs.randint(195, 205), rs.randint(400, 450),
                rs.randint(195, 205), rs.randint(100, 110),
                rs.randn() * 2 * np.pi - np.pi
                ])
        self._set_state(state)

        observation = self._get_obs()
        info = {}
        info['success'] = False
        # print("shape of observation: ", observation.shape)  # 40
        return observation, info

    def step(self, action_velocity):
        action = self.agent.position + action_velocity * 20
        action = np.clip(action, 0, self.window_size)
        logger.debug('%s %s', "action: ", action)
        dt = 1.0 / self.sim_hz
        self.n_contact_points = 0
        n_steps = self.sim_hz // self.control_hz
        if action is not None:
            self.latest_action = action
            for i in range(n_steps):
                # Step PD control.
                # self.agent.velocity = self.k_p * (act - self.agent.position)    # P control works too.
                acceleration = self.k_p * (action - self.agent.position) + self.k_v * (Vec2d(0, 0) - self.agent.velocity)
                self.agent.velocity += acceleration * dt

                # Step physics.
                self.space.step(dt)

        goal_body = self._get_goal_pose_body(self.goal_pose)
        goal_geom = pymunk_to_shapely(goal_body, [pymunk.shapes.Circle(goal_body, 15)])
        agent_geom = pymunk_to_shapely(self.agent, [pymunk.shapes.Circle(self.agent, 15)])

        intersection_area = goal_geom.intersection(agent_geom).area
        goal_area = goal_geom.area
        coverage = intersection_area / goal_area
        reward = np.clip(coverage / self.success_threshold, 0, 1)
        done = coverage > self.success_threshold

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, done, info

    def render(self, mode="human"):
        return self._render_frame(mode)

    def teleop_agent(self):
        TeleopAgent = collections.namedtuple('TeleopAgent', ['act'])
        def act(obs):
            act = None
            mouse_position = pymunk.pygame_util.from_pygame(Vec2d(*pygame.mouse.get_pos()), self.screen)
            if self.teleop or (mouse_position - self.agent.position).length < 30:
                self.teleop = True
                act = mouse_position
            return act
        return TeleopAgent(act)

    def _get_obs(self):
        obs = np.array(
            tuple(self.agent.position) \
            + tuple(self.obstacle.position) \
            + (self.goal_pose[0], self.goal_pose[1], self.goal_pose[2]))
        return obs

    def _get_goal_pose_body(self, pose):
        mass = 1
        inertia = pymunk.moment_for_circle(mass, 0, 15)
        body = pymunk.Body(mass, inertia)
        body.position = pose[:2]  # No need to call tolist()
        body.angle = pose[2]
        return body
    
    def _get_info(self):
        n_steps = self.sim_hz // self.control_hz
        n_contact_points_per_step = int(np.ceil(self.n_contact_points / n_steps))
        info = {
            'pos_agent': np.array(self.agent.position),
            'vel_agent': np.array(self.agent.velocity),
            'goal_pose': self.goal_pose,
            'n_contacts': n_contact_points_per_step}
        return info

    def _render_frame(self, mode):
        if self.window is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        self.screen = canvas

        draw_options = DrawOptions(canvas)

        goal_body = self._get_goal_pose_body(self.goal_pose)
        pygame.draw.circle(canvas, self.goal_color, pymunk.pygame_util.to_pygame(goal_body.position, draw_options.surface), 15)

        self.space.debug_draw(draw_options)

        # Draw the path
        if self.path is not None:
            path = self.path
            for i in range(len(path) - 1):
                pygame.draw.line(self.screen, (255, 0, 0), path[i], path[i+1], 5)  # Red path


        if mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

        img = np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        img = cv2.resize(img, (self.render_size, self.render_size))
        if self.render_action:
            if self.render_action and (self.latest_action is not None):
                action = np.array(self.latest_action)
                coord = (action / 512 * 96).astype(np.int32)
                marker_size = int(8/96*self.render_size)
                thickness = int(1/96*self.render_size)
                cv2.drawMarker(img, coord,
                    color=(255,0,0), markerType=cv2.MARKER_CROSS,
                    markerSize=marker_size, thickness=thickness)
        return img

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
    
    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0,25536)
        self._seed = seed
        self.np_random = np.random.default_rng(seed)

    def _handle_collision(self, arbiter, space, data):
        self.n_contact_points += len(arbiter.contact_point_set.points)

    def _set_state(self, state):
        if isinstance(state, np.ndarray):
            state = state.tolist()
        pos_agent = state[:2]
        pos_goal = state[2:4]
        goal_angle = state[4]
        self.agent.position = pos_agent

        self.goal_pose = [pos_goal[0], pos_goal[1], goal_angle]

        self.space.step(1.0 / self.sim_hz)

    
    def _set_state_local(self, state_local):
        agent_pos_local = state_local[:2]
        goal_pose_local = state_local[2:]
        tf_img_obj = st.AffineTransform(
            translation=self.goal_pose[:2], 
            rotation=self.goal_pose[2])
        tf_obj_new = st.AffineTransform(
            translation=goal_pose_local[:2],
            rotation=goal_pose_local[2]
        )
        tf_img_new = st.AffineTransform(
            matrix=tf_img_obj.params @ tf_obj_new.params
        )
        agent_pos_new = tf_img_new(agent_pos_local)
        new_state = np.array(
            list(agent_pos_new[0]) + list(tf_img_new.translation) \
                + [tf_img_new.rotation])
        self._set_state(new_state)
        return new_state

    def _setup(self):
        self.space = pymunk.Space()
        self.space.gravity = 0, 0
        self.space.damping = 0
        self.teleop = False
        self.render_buffer = list()
        
        walls = [
            self._add_segment((5, 506), (5, 5), 2),
            self._add_segment((5, 5), (506, 5), 2),
            self._add_segment((506, 5), (506, 506), 2),
            self._add_segment((5, 506), (506, 506), 2)
        ]
        self.space.add(*walls)

        self.agent = self.add_dynamic_circle((256, 400), 15)
        self.obstacle = self.add_obstacle((200, 256), self.obstacle_width, self.obstacle_height)
        self.goal_color = pygame.Color('LightGreen')
        self.goal_pose = np.array([256,256,np.pi/4])

        self.collision_handeler = self.space.add_collision_handler(1, 2)
        self.collision_handeler.post_solve = self._handle_collision
        self.n_contact_points = 0

        self.max_score = 50 * 100
        self.success_threshold = 0.95    # 95% coverage.

    def _add_segment(self, a, b, radius):
        shape = pymunk.Segment(self.space.static_body, a, b, radius)
        shape.color = pygame.Color('LightGray')
        return shape

    def add_circle(self, position, radius):
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = position
        body.friction = 1
        shape = pymunk.Circle(body, radius)
        shape.color = pygame.Color('RoyalBlue')
        self.space.add(body, shape)
        return body

    def add_dynamic_circle(self, position, radius):
        mass = 1
        inertia = pymunk.moment_for_circle(mass, 0, radius)
        body = pymunk.Body(mass, inertia)
        body.position = position
        body.friction = 1
        shape = pymunk.Circle(body, radius)
        shape.color = pygame.Color('RoyalBlue')
        shape.filter = pymunk.ShapeFilter(categories=1, mask=2)
        self.space.add(body, shape)
        return body

    def add_obstacle(self, position, width, height):
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        body.position = position
        shape = pymunk.Poly.create_box(body, (width, height))
        shape.color = pygame.Color('DarkGray')
        shape.filter = pymunk.ShapeFilter(categories=2, mask=1)
        self.space.add(body, shape)
        return body


    def _handle_collision(self, arbiter, space, data):
        self.n_contact_points += len(arbiter.contact_point_set.points)
        for contact in arbiter.contact_point_set.points:
            impulse = contact.normal_impulse
            self.agent.velocity = Vec2d(0, 0)


    def control_policy(self):
        start = tuple(self.agent.position)
        goal = tuple(self.goal_pose[:2])
        slack_obstacles, strict_obstacles = self.get_obstacles()
        path = self.a_star_search(start, goal, slack_obstacles, strict_obstacles)
        self.path = path
        self.last_path = self.path
        if path and len(path) > 1:
            next_point = path[1]
            action = np.array(next_point) - np.array(start)
            action_velocity = action / np.linalg.norm(action)  # Normalize to get direction
            return action_velocity
        else:
            action = (self.goal_pose[:2] - self.agent.position)
            action_velocity = action / np.linalg.norm(action) 
            return action_velocity
            # return np.array([0, 0])  # No valid path found, stay in place

    def a_star_search(self, start, goal, slack_obstacles, strict_obstacles, circle_set=None):
        # The A* search algorithm
        count_number = 0
        # Initialize both the open and closed sets
        open_set = []
        heapq.heappush(open_set, (0 + self.heuristic(start, goal), 0, start))
        came_from = {}

        # Cost from start to node
        g_score = {start: 0}

        # Main A* algorithm
        while open_set and count_number < 5000:
            count_number = count_number + 1
            _, current_cost, current = heapq.heappop(open_set)
            if self.heuristic(current, goal) < 25:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            for neighbor in self.get_neighbors(current, slack_obstacles, strict_obstacles):
                tentative_g_score = current_cost + self.heuristic(current, neighbor) + \
                                    0.25 * self.heuristic_close_to_last_path(neighbor) + \
                                    10.0 * self.heuristic_close_to_circle_sets(neighbor, circle_set)
                if self.is_within_obstacles(neighbor, slack_obstacles):
                    tentative_g_score = tentative_g_score + 3

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, tentative_g_score, neighbor))

        return None

    def heuristic(self, start, goal):
        return np.linalg.norm(np.array(start) - np.array(goal))
    
    def get_neighbors(self, node, slack_obstacles, strict_obstacles):
        # This would depend on your grid resolution
        grid_resolution = 10
        # directions = [(0, grid_resolution), (0, -grid_resolution), 
        #               (grid_resolution, 0), (-grid_resolution, 0), 
        #               (grid_resolution, grid_resolution), (-grid_resolution, -grid_resolution), 
        #               (grid_resolution, -grid_resolution), (-grid_resolution, grid_resolution)]
        directions = [(0, grid_resolution), (0, -grid_resolution), 
                  (grid_resolution, 0), (-grid_resolution, 0), 
                  (grid_resolution, grid_resolution), (-grid_resolution, -grid_resolution), 
                  (grid_resolution, -grid_resolution), (-grid_resolution, grid_resolution),
                  (2 * grid_resolution, 0), (-2 * grid_resolution, 0),
                  (0, 2 * grid_resolution), (0, -2 * grid_resolution),
                  (2 * grid_resolution, grid_resolution), (-2 * grid_resolution, -grid_resolution), 
                  (2 * grid_resolution, -grid_resolution), (-2 * grid_resolution, grid_resolution),
                  (grid_resolution, 2 * grid_resolution), (-grid_resolution, -2 * grid_resolution), 
                  (grid_resolution, -2 * grid_resolution), (-grid_resolution, 2 * grid_resolution),
                  ]
        neighbors = []
        for direction in directions:
            neighbor = (node[0] + direction[0], node[1] + direction[1])
            if not self.is_within_obstacles(neighbor, strict_obstacles):
                neighbors.append(neighbor)
        return neighbors

    def is_within_obstacles(self, point, obstacles):
        # Create a point object
        point_obj = sg.Point(point)
        # If point is within any obstacle, it's not a valid point to move to
        return any(obstacle.contains(point_obj) for obstacle in obstacles)


    def heuristic_close_to_last_path(self, node_a):
        # Check if last_path is not empty
        # if self.path is None or len(self.last_path) < 10:
        if self.last_path is None or len(self.last_path) < 20:
            # Return a high heuristic value if last_path is empty,
            # indicating that node_a is "far" from an empty path.
            # This value could be adjusted depending on the specifics of your implementation.
            return 0.0
        
        # Initialize minimum distance to a very high value
        min_distance = float('inf')
        
        # Iterate over each node in the last_path
        i = 0
        for node in self.last_path:
            i = i + 1
            # if i < 7:
            if i < 19:
                continue
            # Calculate distance from node_a to the current node in the loop
            distance = self.heuristic(node_a, node) # Assuming a distance_to method exists
            # Update min_distance if the current distance is smaller
            if distance < min_distance:
                min_distance = distance
        # Return the minimum distance found
        return min_distance
    
    def heuristic_close_to_circle_sets(self, neighbor, circle_set):
        # Define this heuristic based on your requirements
        return 0
    

    def create_inflated_obstacle(self, obstacle, inflation_distance):
        # Inflate the obstacle by a certain distance
        inflated_obstacle = obstacle.buffer(inflation_distance)
        return inflated_obstacle

    def create_obstacle(self):
        # Define the rectangular obstacle as a polygon
        obstacle_position = self.obstacle.position
        obstacle_width = self.obstacle_width
        obstacle_height = self.obstacle_height
        obstacle_polygon = sg.box(
            obstacle_position[0] - obstacle_width / 2,
            obstacle_position[1] - obstacle_height / 2,
            obstacle_position[0] + obstacle_width / 2,
            obstacle_position[1] + obstacle_height / 2
        )
        return obstacle_polygon

    def create_walls(self, wall_thickness):
        # Manually define the walls as polygons that exactly match the environment boundaries
        wall_polygons = [
            sg.Polygon([(0, -wall_thickness), (self.window_size, -wall_thickness),
                    (self.window_size, 0), (0, 0)]),  # Top wall
            sg.Polygon([(self.window_size, 0), (self.window_size + wall_thickness, 0),
                    (self.window_size + wall_thickness, self.window_size), (self.window_size, self.window_size)]),  # Right wall
            sg.Polygon([(0, self.window_size), (self.window_size, self.window_size),
                    (self.window_size, self.window_size + wall_thickness), (0, self.window_size + wall_thickness)]),  # Bottom wall
            sg.Polygon([(-wall_thickness, 0), (0, 0),
                    (0, self.window_size), (-wall_thickness, self.window_size)])  # Left wall
        ]
        return wall_polygons

    def get_obstacles(self):
        # Combine the rectangular obstacle and walls into a list of obstacles
        inflation_distance = 10.5  # Define the safe distance
        wall_thickness = 100 # Should be thicker than the grid resolution of A-star algorithm, otherwise the searched grid can be outside of the walls
        
        obstacle_polygon = self.create_obstacle()
        inflated_obstacle = self.create_inflated_obstacle(obstacle_polygon, inflation_distance)
        
        walls = self.create_walls(wall_thickness)
        slack_obstacles = [inflated_obstacle]
        strict_obstacles = [obstacle_polygon] + walls
        return slack_obstacles, strict_obstacles