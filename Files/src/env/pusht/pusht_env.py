import logging

logger = logging.getLogger(__name__)

# import gym
import gymnasium as gym
# from gym import spaces
from gymnasium import spaces

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

import random


import heapq # for a star
import math

def pymunk_to_shapely(body, shapes):
    geoms = list()
    for shape in shapes:
        if isinstance(shape, pymunk.shapes.Poly):
            verts = [body.local_to_world(v) for v in shape.get_vertices()]
            verts += [verts[0]]
            geoms.append(sg.Polygon(verts))
        else:
            raise RuntimeError(f'Unsupported shape type {type(shape)}')
    geom = sg.MultiPolygon(geoms)
    return geom

class PushTEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 10}
    reward_range = (0., 1.)

    def __init__(self,
            # legacy=False,
            legacy=True, 
            block_cog=None, damping=None,
            render_action=True,
            render_size=96,
            reset_to_state=None,
            use_abs_action = False,
            config = None
        ):
        self.abs_action = use_abs_action
        self.use_image_obs = config.use_image
        self._seed = None
        # self.seed(5)  # 1102-1
        # self.seed(4)
        self.window_size = ws = 512  # The size of the PyGame window
        self.render_size = render_size
        self.sim_hz = 100
        # Local controller params.
        self.k_p, self.k_v = 100, 20    # PD control.z
        self.control_hz = self.metadata['video.frames_per_second']
        # legacy set_state for data compatibility
        self.legacy = legacy

        # agent_pos, block_pos, block_angle
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

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        self.screen = None

        self.space = None
        self.teleop = None
        self.render_buffer = None
        self.latest_action = np.array([0, 0])
        self.reset_to_state = reset_to_state
        self.interaction_points = None
        self.circle_points = None  # define the points on the circle, which center is (0, 0.5*length*scale)
        self.path = None
        self.low_level_path_candidates = []
        self.last_path = None
        self.last_interaction_points = None
        self.selected_interaction_set = []
        self.aster_no_solution_count = 0

        self.previous_obs = None
        self.obs_dict_past = None

        # statistics used for action normalization, when abs_action = true

        self.action_max_list = [list(config.action_max)] 
        self.action_min_list = [list(config.action_min)]
        self.eef_pos_max_list = [list(config.robot0_eef_pos_max)]  # ours
        self.eef_pos_min_list = [list(config.robot0_eef_pos_min)] 
        
        self.record_action_statistics = False # used in vel mode
        # self.record_action_statistics = True # used in vel mode

    def normalize_eef_pos(self, pos, min_list, max_list):
        # arr = np.asarray(action, dtype=float)
        action = pos.copy()
        lo  = np.asarray(min_list, dtype=np.float32)
        hi  = np.asarray(max_list, dtype=np.float32)
        # centers and half-ranges
        center = (hi + lo) / 2.0
        half_range = (hi - lo) / 2.0
        # avoid division by zero
        half_range[half_range == 0] = 1.0
        action = ((action - center) / half_range)
        return action

    def normalize_abs_action(self, action_, action_min, action_max):
        # arr = np.asarray(action, dtype=float)
        action = action_.copy()
        lo  = np.asarray(action_min, dtype=np.float32)
        hi  = np.asarray(action_max, dtype=np.float32)
        # centers and half-ranges
        center = (hi + lo) / 2.0
        half_range = (hi - lo) / 2.0
        # avoid division by zero
        half_range[half_range == 0] = 1.0
        action = ((action - center) / half_range)
        return action
    
    def unnormalize_abs_action(self, norm_action_, action_min, action_max):
        norm_action = norm_action_.copy()
        arr = np.asarray(norm_action, dtype=np.float32)
        lo  = np.asarray(action_min, dtype=np.float32)
        hi  = np.asarray(action_max, dtype=np.float32)
        # compute half-range and center
        half_range = (hi - lo) / 2.0
        center     = (hi + lo) / 2.0
        # avoid division-by-zero artifacts (if hi==lo, half_range==0)
        # but here we only need half_range for multiplication, so it’s fine:
        arr = (arr * half_range + center)
        return arr
    
    def set_seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        self._seed = seed
        return None
        
    
    def reset(self):
        self.obs_dict_past = None
        self.previous_obs = None
        self.interaction_points = None
        self.path = None
        self.last_path = None
        self.last_interaction_points = None
        self.selected_interaction_set = []
        self.circle_points = None
        self.aster_no_solution_count = 0
        seed = self._seed
        self._setup()
        if self.block_cog is not None:
            self.block.center_of_gravity = self.block_cog
        if self.damping is not None:
            self.space.damping = self.damping
        
        # use legacy RandomState for compatibility
        state = self.reset_to_state
        if state is None:
            rs = np.random.RandomState(seed=seed)
            # rs = np.random.RandomState(seed = self._seed)
            state = np.array([
                rs.randint(50, 450), rs.randint(50, 450),
                rs.randint(100, 400), rs.randint(100, 400),
                rs.randn() * 2 * np.pi - np.pi
                ])
            # state = np.array([
            #         np.random.randint(50, 450), np.random.randint(50, 450),
            #         np.random.randint(100, 400), np.random.randint(100, 400),
            #         np.random.randn() * 2 * np.pi - np.pi
            #     ])
            # state = np.array([
            #     rs.randint(50, 450), rs.randint(50, 450),
            #     rs.randint(150, 350), rs.randint(150, 350),
            #     rs.randn() * 2 * np.pi - np.pi
            #     ])

        # #### [start] #####
        #     #  for testing the muli-modality of pushing-T 
        #     state = np.array([
        #         256+ 50* np.cos(np.pi/4),256-50* np.cos(np.pi/4),
        #         256- 0* np.cos(np.pi/4), 256+50* np.cos(np.pi/4),
        #         np.pi/4
        #         ])
        
        # ####  two prior path going left or right for testing the muli-modality of pushing-T 
        # path_right =  [(291.3553390593274, 227.7157287525381), (291.3553390593274, 237.7157287525381), (291.3553390593274, 247.7157287525381), (291.3553390593274, 257.7157287525381), (291.3553390593274, 267.7157287525381), (291.3553390593274, 277.7157287525381), (291.3553390593274, 287.7157287525381), (291.3553390593274, 297.7157287525381), (291.3553390593274, 307.7157287525381), (291.3553390593274, 317.7157287525381), (291.3553390593274, 327.7157287525381), (291.3553390593274, 337.7157287525381), (281.3553390593274, 347.7157287525381), (271.3553390593274, 357.7157287525381), (261.3553390593274, 367.7157287525381), (251.35533905932738, 377.7157287525381), (241.35533905932738, 377.7157287525381), (231.35533905932738, 377.7157287525381), (221.35533905932738, 377.7157287525381), (211.35533905932738, 377.7157287525381), (201.35533905932738, 377.7157287525381), (191.35533905932738, 377.7157287525381), (181.35533905932738, 387.7157287525381), (171.35533905932738, 397.7157287525381), (161.35533905932738, 407.7157287525381), (151.35533905932738, 407.7157287525381), (141.35533905932738, 407.7157287525381), (131.35533905932738, 407.7157287525381)]
        # path_left = [(274.5154105735076, 230.54926443019346), (264.5154105735076, 230.54926443019346), (254.5154105735076, 230.54926443019346), (244.5154105735076, 230.54926443019346), (234.5154105735076, 230.54926443019346), (224.5154105735076, 230.54926443019346), (214.5154105735076, 230.54926443019346), (204.5154105735076, 230.54926443019346), (194.5154105735076, 230.54926443019346), (184.5154105735076, 220.54926443019346), (174.5154105735076, 220.54926443019346), (164.5154105735076, 230.54926443019346), (154.5154105735076, 240.54926443019346), (144.5154105735076, 250.54926443019346), (134.5154105735076, 260.5492644301935), (124.51541057350761, 270.5492644301935), (114.51541057350761, 280.5492644301935), (114.51541057350761, 290.5492644301935), (114.51541057350761, 300.5492644301935), (114.51541057350761, 310.5492644301935), (114.51541057350761, 320.5492644301935), (114.51541057350761, 330.5492644301935), (104.51541057350761, 340.5492644301935), (104.51541057350761, 350.5492644301935), (104.51541057350761, 360.5492644301935), (104.51541057350761, 370.5492644301935), (104.51541057350761, 380.5492644301935)]
        # rand_number = np.random.random(1)[0]
        # if rand_number > 0.5:
        #     self.last_path = path_right
        # else:
        #     self.last_path = path_left
        # ##### [end]#####

        self._set_state(state)
        for i in range(5):
            if not self.abs_action:
                self.step(np.zeros(2))
            else:
                self.step(self.normalize_abs_action(np.array(self.agent.position),  action_max=self.action_max_list[0], action_min=self.action_min_list[0]))
        observation = self._get_obs()
        info = {}
        info['success'] = False
        # print("shape of observation: ", observation.shape)  # 40

        # # save the initial state and the corresponding seed (self._seed)
        # img = self._render_frame(mode='rgb_array')
        # import matplotlib.pyplot as plt
        # plt.imsave('initial_frame_'+str(self._seed)+'.png', img)

        return observation, info

    def step(self, action):
        # print("last_path: ", self.last_path)
        # action_velocity = self.control_policy()
        # action in range [-1, 1]
        # change action from position to velocity
        action_ = action.copy()
        logger.debug('%s %s', "action: ", action_)

        if not self.abs_action:
            action = self.agent.position + action_ * 10
            action = np.clip(action, 0, self.window_size)
        else:
            # check if action_ is insdie [-1, 1]
            if not (np.all((action_ >= -1) & (action_ <= 1))):
                logger.warning(
                    "Action is out of bounds; continuing without clipping. min=%s max=%s",
                    np.min(action_),
                    np.max(action_),
                )
            action_ = self.unnormalize_abs_action(action_, action_max=self.action_max_list[0], action_min=self.action_min_list[0])

            ##TODO add limits for the action, as the robosuite osc controller doesn't have any torque constraints
            if self.obs_dict_past is not None:
                past_eef_pos = self.unnormalize_abs_action(self.obs_dict_past['agent_pos'], action_max=self.eef_pos_max_list[0], action_min=self.eef_pos_min_list[0])
                # print("past_eef_pos: ", past_eef_pos)
                action_eef_pos = action_
                norm_max = 15 # few of the robomimic demo execceds this limits, but the trained policy can still succeed
                direction = action_eef_pos - past_eef_pos
                dist = np.linalg.norm(direction)
                if dist > norm_max:
                    direction = direction / dist * norm_max
                    logger.debug("execced norm_max")
                saturated_eef_pos = past_eef_pos + direction
                action_ = saturated_eef_pos
            action = action_
        

        if self.record_action_statistics:
            if not hasattr(self, 'action_max'):
                # first-ever action: initialize max & min
                self.action_max = list(action)
                self.action_min = list(action)
            else:
                # update per-dimension
                for i, v in enumerate(action):
                    if v > self.action_max[i]:
                        self.action_max[i] = v
                    if v < self.action_min[i]:
                        self.action_min[i] = v
            logger.debug('%s %s %s %s', "self.action_max: ", self.action_max, " self.action_min: ", self.action_min)

        # # action is position, reshape for [-1, 1] to [0, 512]
        # action = (action + 1) / 2 * 512

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

        # compute reward
        goal_body = self._get_goal_pose_body(self.goal_pose)
        goal_geom = pymunk_to_shapely(goal_body, self.block.shapes)
        block_geom = pymunk_to_shapely(self.block, self.block.shapes)

        intersection_area = goal_geom.intersection(block_geom).area
        goal_area = goal_geom.area
        coverage = intersection_area / goal_area
        reward = np.clip(coverage / self.success_threshold, 0, 1)
        done = coverage > self.success_threshold

        observation = self._get_obs()
        info = self._get_info()
        # done_success = coverage > 0.8
        logger.debug('%s %s %s %s', 'coverage: ', coverage, ' reward: ', reward)
        done_success = coverage > self.success_threshold
        info['success'] = done_success

        # return observation, reward, done_success, False,  info
        return observation, coverage, done_success, False,  info

    def render(self):
        mode = "human"
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

    # def _get_obs(self):
    #     # also add goal_pose to obs
    #     block_pose = np.array(list(self.block.position) + [self.block.angle% (2 * np.pi)])
    #     block_pose_local = block_pose - self.goal_pose

    #     orientation_error = (self.goal_pose[2] - self.block.angle) % (2 * np.pi)
    #     orientation_error = (orientation_error + np.pi) % (2 * np.pi) - np.pi  # Normalize to the range [-pi, pi]

    #     direction_vector = np.array(self.goal_pose[:2]) - np.array(self.block.position)

    #     # obs = np.array(
    #     #     tuple(self.agent.position / 512) \
    #     #     + tuple(block_pose_local / 512) \
    #     #     + tuple(self.block.position / 512) \
    #     #     + (self.block.angle % (2 * np.pi),))

    #     obs = np.array(
    #          tuple(self.agent.position / 512) \
    #         + tuple(direction_vector / 512) \
    #         + (orientation_error, ) \
    #         + tuple(self.block.position / 512) \
    #         + tuple((self.block.position - self.agent.position)/512) \
    #         + (self.block.angle % (2 * np.pi) - np.pi,))
    #     return obs

    def _get_obs(self):
        # Calculate the pose of the block in global coordinates and normalize the angle
        block_pose = np.array(list(self.block.position) + [self.block.angle % (2 * np.pi)])
        block_pose_local = block_pose - self.goal_pose

        # Calculate the orientation error and normalize it
        orientation_error = (self.goal_pose[2] - self.block.angle) % (2 * np.pi)
        orientation_error = (orientation_error + np.pi) % (2 * np.pi) - np.pi  # Normalize to the range [-pi, pi]

        # Use sine and cosine for the orientation error
        orientation_error_sin = np.sin(orientation_error)
        orientation_error_cos = np.cos(orientation_error)

        # Direction vector from the agent to the goal
        direction_vector = np.array(self.goal_pose[:2]) - np.array(self.block.position)

        # Calculate the sine and cosine of the block angle relative to the agent
        block_angle_relative = (self.block.angle % (2 * np.pi) - np.pi)
        block_angle_sin = np.sin(block_angle_relative)
        block_angle_cos = np.cos(block_angle_relative)

        # Construct the observation array
        current_obs = np.array(
            tuple(self.agent.position / 512) \
            + tuple(direction_vector / 512) \
            + (orientation_error_sin, orientation_error_cos) \
            + tuple(self.block.position / 512) \
            + tuple((self.block.position - self.agent.position) / 512) \
            + (block_angle_sin, block_angle_cos))

        expanded_obs = np.expand_dims(current_obs, axis=0)

        # Concatenate with previous observation if it exists
        if self.previous_obs is None:
            self.previous_obs = expanded_obs

        concatenated_obs = np.concatenate((self.previous_obs, expanded_obs))

        # Update the previous observation
        self.previous_obs = expanded_obs

        return concatenated_obs

    def _get_goal_pose_body(self, pose):
        mass = 1
        inertia = pymunk.moment_for_box(mass, (50, 100))
        body = pymunk.Body(mass, inertia)
        # preserving the legacy assignment order for compatibility
        # the order here doesn't matter somehow, maybe because CoM is aligned with body origin
        body.position = pose[:2].tolist()
        body.angle = pose[2]
        return body
    
    def _get_info(self):
        n_steps = self.sim_hz // self.control_hz
        n_contact_points_per_step = int(np.ceil(self.n_contact_points / n_steps))
        info = {
            'pos_agent': np.array(self.agent.position),
            'vel_agent': np.array(self.agent.velocity),
            'block_pose': np.array(list(self.block.position) + [self.block.angle]),
            'goal_pose': self.goal_pose,
            'n_contacts': n_contact_points_per_step,
            'success':  False}
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

        # Draw goal pose.
        goal_body = self._get_goal_pose_body(self.goal_pose)
        for shape in self.block.shapes:
            goal_points = [pymunk.pygame_util.to_pygame(goal_body.local_to_world(v), draw_options.surface) for v in shape.get_vertices()]
            goal_points += [goal_points[0]]
            pygame.draw.polygon(canvas, self.goal_color, goal_points)

        # Draw agent and block.
        self.space.debug_draw(draw_options)

        # self.draw_interaction_points()

        if mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # the clock is already ticked during in step for "human"


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
        pos_block = state[2:4]
        rot_block = state[4]
        self.agent.position = pos_agent
        # setting angle rotates with respect to center of mass
        # therefore will modify the geometric position
        # if not the same as CoM
        # therefore should be modified first.
        if self.legacy:
            # for compatibility with legacy data
            self.block.position = pos_block
            self.block.angle = rot_block
        else:
            self.block.angle = rot_block
            self.block.position = pos_block

        # Run physics to take effect
        self.space.step(1.0 / self.sim_hz)
    
    def _set_state_local(self, state_local):
        agent_pos_local = state_local[:2]
        block_pose_local = state_local[2:]
        tf_img_obj = st.AffineTransform(
            translation=self.goal_pose[:2], 
            rotation=self.goal_pose[2])
        tf_obj_new = st.AffineTransform(
            translation=block_pose_local[:2],
            rotation=block_pose_local[2]
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
        
        # Add walls.
        walls = [
            self._add_segment((5, 506), (5, 5), 2),
            self._add_segment((5, 5), (506, 5), 2),
            self._add_segment((506, 5), (506, 506), 2),
            self._add_segment((5, 506), (506, 506), 2)
        ]
        self.space.add(*walls)

        # Add agent, block, and goal zone.
        self.agent = self.add_circle((256, 400), 15)
        self.block = self.add_tee((256, 300), 0)
        self.goal_color = pygame.Color('LightGreen')
        self.goal_pose = np.array([256,256,np.pi/4])  # x, y, theta (in radians)

        # Add collision handling
        self.collision_handeler = self.space.add_collision_handler(0, 0)
        self.collision_handeler.post_solve = self._handle_collision
        self.n_contact_points = 0

        self.max_score = 50 * 100
        # self.success_threshold = 0.95    # 95% coverage.
        self.success_threshold = 0.94    # 94% coverage.

    def _add_segment(self, a, b, radius):
        shape = pymunk.Segment(self.space.static_body, a, b, radius)
        shape.color = pygame.Color('LightGray')    # https://htmlcolorcodes.com/color-names
        return shape

    def add_circle(self, position, radius):
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = position
        body.friction = 1
        shape = pymunk.Circle(body, radius)
        shape.color = pygame.Color('RoyalBlue')
        self.space.add(body, shape)
        return body

    def add_box(self, position, height, width):
        mass = 1
        inertia = pymunk.moment_for_box(mass, (height, width))
        body = pymunk.Body(mass, inertia)
        body.position = position
        shape = pymunk.Poly.create_box(body, (height, width))
        shape.color = pygame.Color('LightSlateGray')
        self.space.add(body, shape)
        return body

    def add_tee(self, position, angle, scale=30, color='LightSlateGray', mask=pymunk.ShapeFilter.ALL_MASKS()):
        mass = 1
        length = 4
        vertices1 = [(-length*scale/2, scale),
                                 ( length*scale/2, scale),
                                 ( length*scale/2, 0),
                                 (-length*scale/2, 0)]
        inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)
        vertices2 = [(-scale/2, scale),
                                 (-scale/2, length*scale),
                                 ( scale/2, length*scale),
                                 ( scale/2, scale)]
        inertia2 = pymunk.moment_for_poly(mass, vertices=vertices1)
        body = pymunk.Body(mass, inertia1 + inertia2)
        shape1 = pymunk.Poly(body, vertices1)
        shape2 = pymunk.Poly(body, vertices2)
        shape1.color = pygame.Color(color)
        shape2.color = pygame.Color(color)
        shape1.filter = pymunk.ShapeFilter(mask=mask)
        shape2.filter = pymunk.ShapeFilter(mask=mask)
        body.center_of_gravity = (shape1.center_of_gravity + shape2.center_of_gravity) / 2
        body.position = position
        body.angle = angle
        body.friction = 1
        self.space.add(body, shape1, shape2)
        return body

    def control_policy(self, robot_action):
        # by default, the robot_action is velocity 
        if self.abs_action:
            # action = self.agent.position + action_ * 10 # in step function
            robot_action = 0.1 * (self.agent.position - robot_action)

        # based on the current state, output the desired action_velocity
        
        # step 1: predefine several interaction points of the T-shape (i.e. 4 points for rotating, 4 points for pushing)

        # step 2: check whether the current state of T-shape has errors in both position and orientation (angle)
        # Use the simple strategy: If the 'T' object is not aligned with the goal orientation, rotate it. Once the 'T' object is roughly aligned, push it towards the goal position. 
        
        # step 3: find the interaction point that the we want the robot to move to
        # step 4: find the collision-free path that drive the robot to the desired interaction point, once we have the path, we caculate the "action_velocity"

        # self.define_interaction_points()
        # # self.select_interaction_point()
        # i_point_key= self.select_interaction_point()

        # if i_point_key is not None:
        #     i_point = self.interaction_points[i_point_key]
        # else:
        #     i_point = None
        # self.last_interaction_points = i_point
        # print("i_point: ", i_point)
        # if i_point is not None:
        #     # [To do] when choose the desired interaction point, also check whether it conflicts with the robot_action (robot could have a different goal)
        #     closest_velocity = self.control_policy_given_one_subGoal(i_point, robot_action)
        #     return closest_velocity
        # else: 
        #     return np.array([0, 0])
        

        self.define_interaction_points()
        # self.select_interaction_point()
        i_point_key= self.select_interaction_point()

        if i_point_key is not None:
            if isinstance(i_point_key, str):
                i_point_key = [i_point_key]
            i_point_list = [self.interaction_points[i] for i in i_point_key]
        else:
            i_point_list = None
        # print("i_point: ", i_point)
        if i_point_list is not None:
            # [To do] when choose the desired interaction point, also check whether it conflicts with the robot_action (robot could have a different goal)
            
            selected_i_point_id = 0
            if len(i_point_list) > 1:
                path_length_min = 100
                min_path_id = None
                # for i in range(0, len(i_point_list)):
                #     path_length = self.control_policy_given_one_subGoal_minimum_path(i_point_list[i])
                #     if path_length < path_length_min:
                #         min_path_id = i
                #         path_length_min = path_length
                
                path_length_0 = self.control_policy_given_one_subGoal_minimum_path(i_point_list[0])
                path_length_1 = self.control_policy_given_one_subGoal_minimum_path(i_point_list[1])
                if path_length_0 > path_length_1 + 5:
                    min_path_id = 1
                elif path_length_1 > path_length_0 + 5:
                    min_path_id = 0
                else:
                    min_path_id = 0
                if min_path_id is not None:
                    selected_i_point_id = min_path_id


            closest_velocity = self.control_policy_given_one_subGoal(i_point_list[selected_i_point_id], robot_action)
            self.last_interaction_points = i_point_list[selected_i_point_id]
            
            if self.abs_action:
                # by default, the output of the teacher policy is velocity, so transfer to position if abs_action=true
                teacher_abs_action = self.agent.position + closest_velocity * 10
                teacher_abs_action = np.clip(teacher_abs_action, 0, self.window_size)
                teacher_abs_action = self.normalize_abs_action(teacher_abs_action, action_max=self.action_max_list[0], action_min=self.action_min_list[0])
                # print("normliazlie teacher action, ", teacher_abs_action)
                return teacher_abs_action
            else:
                if np.isnan(closest_velocity[0]):
                    logger.warning("Teacher velocity contains NaN; returning zero velocity. closest_velocity=%s", closest_velocity)
                    return np.array([0, 0])
                action_max = np.max(abs(closest_velocity))
                if action_max > 1:
                    closest_velocity =  closest_velocity / action_max
                logger.debug('%s %s', "closest_velocity: ", closest_velocity)
                return closest_velocity

            # distance_to_robot_action_min = 100
            # closest_velocity_min = None
            # for i in range(0, len(i_point_list)):
            #     closest_velocity = self.control_policy_given_one_subGoal(i_point_list[i], robot_action)
            #     if closest_velocity is None:
            #         distance_to_robot_action = 100
            #     else:
            #         distance_to_robot_action = np.linalg.norm(closest_velocity - robot_action)
            #     if distance_to_robot_action_min > distance_to_robot_action:
            #         distance_to_robot_action_min = distance_to_robot_action
            #         self.last_interaction_points = i_point_list[i]
            #         closest_velocity_min = closest_velocity
            # return closest_velocity_min
        else: 
            if self.abs_action:
                teacher_abs_action = np.array(self.agent.position) 
                teacher_abs_action = self.normalize_abs_action(teacher_abs_action, action_max=self.action_max_list[0], action_min=self.action_min_list[0])
                # print("normliazlie teacher action")
                return teacher_abs_action
            else:
                return np.array([0, 0])
        
    def control_policy_given_one_subGoal(self, i_point, robot_action):
        desired_interaction_position = i_point['global_point']
        slack_obstacles, strict_obstacles = self.get_obstacles()
        adjusted_goal_point = self.adjust_goal_point(i_point,strict_obstacles, 5 )
        # print("adjusted_goal_point: ", adjusted_goal_point, " desired_interaction_position: ", desired_interaction_position)
        start_pose = self.agent.position
        # print("start_pose: ", start_pose)
        if self.last_path is None:
            self.define_circle_sample_points()
            circle_set1, circle_set2 = self.find_two_closest_points_and_divide_sorted_simple(self.agent.position, desired_interaction_position)
            self.low_level_path_candidates = []
            possible_path_1 = self.a_star_search(start_pose, adjusted_goal_point, slack_obstacles, strict_obstacles, circle_set1)
            possible_path_2 = self.a_star_search(start_pose, adjusted_goal_point, slack_obstacles, strict_obstacles, circle_set2)
            # for each path, find which circle_set they are close to
            
            def path_circle_set_distance(path, circle_set):
                distance = 0
                for i in range(0, len(path)):
                    distance = distance + self.heuristic_close_to_circle_sets(path[i], circle_set)
                return distance
            
            self.low_level_path_candidates.append(possible_path_1)
            self.low_level_path_candidates.append(possible_path_2)
            
            solution_flag1, velocity_candidate1 = self.generate_velocity_command(possible_path_1,start_pose, 1, i_point)
            solution_flag2, velocity_candidate2 = self.generate_velocity_command(possible_path_2,start_pose, 1, i_point)
            # find the closest teacher action to the robot action
            closest_velocity = self.find_closest_action(robot_action, velocity_candidate1, velocity_candidate2)
            if possible_path_1 is not None and possible_path_2 is not None:
                path_1_wrong = path_circle_set_distance(possible_path_1, circle_set2) < path_circle_set_distance(possible_path_1, circle_set1)
                path_2_wrong = path_circle_set_distance(possible_path_2, circle_set1) < path_circle_set_distance(possible_path_2, circle_set2)
                # if len(possible_path_1) > len(possible_path_2) + 5 or path_1_wrong:
                if len(possible_path_1) > len(possible_path_2) + 1 or path_1_wrong:
                    self.path = possible_path_2
                    closest_velocity = velocity_candidate2
                    self.last_path = self.path
                elif len(possible_path_1)  + 1 < len(possible_path_2) or path_2_wrong:
                    self.path = possible_path_1
                    closest_velocity = velocity_candidate1
                    self.last_path = self.path
            if solution_flag1 is False or solution_flag2 is False:
                self.aster_no_solution_count = self.aster_no_solution_count + 1
                # if self.abs_action:
                #     # closest_velocity = None
                #     closest_velocity = np.zeros(2)
                # else:
                #     closest_velocity = None
                closest_velocity = np.zeros(2)
            if self.aster_no_solution_count > 10:
                self._seed = self._seed + 1 # otherwise the reset env is still the same
                self.reset()
                    
        else:
            self.path = self.a_star_search(start_pose, adjusted_goal_point, slack_obstacles, strict_obstacles, None)
            solution_flag, closest_velocity = self.generate_velocity_command(self.path,start_pose, 1, i_point)
            self.last_path = self.path
            if solution_flag is False:
                self.aster_no_solution_count = self.aster_no_solution_count + 1
                # if self.abs_action:
                #     # closest_velocity = None
                #     closest_velocity = np.zeros(2)
                # else:
                #     closest_velocity = None
                closest_velocity = np.zeros(2)
            if self.aster_no_solution_count > 10:
                self._seed = self._seed + 1 # otherwise the reset env is still the same
                self.reset()
                        
        # self.last_path = self.path
            
        # print("path: ", self.path)
        # return self.generate_velocity_command(self.path,start_pose, 1, i_point)
        logger.debug('%s %s', "closest_velocity: ", closest_velocity)
        return closest_velocity

    def control_policy_given_one_subGoal_minimum_path(self, i_point):
        desired_interaction_position = i_point['global_point']
        slack_obstacles, strict_obstacles = self.get_obstacles()
        adjusted_goal_point = self.adjust_goal_point(i_point,strict_obstacles, 5 )
        # print("adjusted_goal_point: ", adjusted_goal_point, " desired_interaction_position: ", desired_interaction_position)
        start_pose = self.agent.position
        # print("start_pose: ", start_pose)
        
        self.last_path = None
        possible_path_1 = self.a_star_search(start_pose, adjusted_goal_point, slack_obstacles, strict_obstacles)
        
        solution_flag1, velocity_candidate1 = self.generate_velocity_command(possible_path_1,start_pose, 1, i_point)

        # find the closest teacher action to the robot action
        
        if possible_path_1 is not None:
            if solution_flag1 is False:
                return 1000
            else:
                return len(possible_path_1)
        else:
            return 1000


    
    def find_closest_action(self, teacher_action, velocity_candidate1, velocity_candidate2):
        # Assume teacher_action, velocity_candidate1, and velocity_candidate2 are 2D vectors (x, y)
        # Calculate Euclidean distances from the teacher_action to each velocity candidate
        if velocity_candidate1 is not None:
            distance1 = self.calculate_euclidean_distance(teacher_action, velocity_candidate1)
        else:
            distance1 = float('inf')  # Assign infinity if velocity_candidate1 is not defined

        if velocity_candidate2 is not None:
            distance2 = self.calculate_euclidean_distance(teacher_action, velocity_candidate2)
        else:
            distance2 = float('inf')  # Assign infinity if velocity_candidate2 is not defined

        # Choose the closest velocity candidate
        if distance1 < distance2:
            return velocity_candidate1
        else:
            return velocity_candidate2

    def calculate_euclidean_distance(self, point1, point2):
        return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

    def generate_velocity_command(self, path, current_position, desired_speed, desired_interaction_point):
        close_to_interaction_point = False
        if not path or len(path) < 2:
            # Path is empty or too short to generate a command
            # print("Path is empty or too short to generate a command")
            # print(desired_interaction_point['global_point'])
            next_waypoint = desired_interaction_point['global_point']

            # check whether this is feasible by checking the distance between current_position and next_waypoint
            if np.linalg.norm(current_position - next_waypoint) > 40:
                logger.debug(" infeasible solution")
                return False, np.array([0, 0])
            else:
                close_to_interaction_point = True
            # return desired_interaction_point['global_point']
            # return np.array([0, 0])
        
        # Assuming the current position is the first point in the path
        # Find the next waypoint
        else:
            next_waypoint = path[1]  # path[-1] is the current position, path[-2] is the next waypoint
        
        # Calculate direction vector from current position to next waypoint
        direction_vector = np.array(next_waypoint) - np.array(current_position).copy()
        
        # Normalize the direction vector
        norm = np.linalg.norm(direction_vector)
        # if norm == 0:
        if norm < 1e-6:
            # Avoid division by zero
            return True, np.array([0, 0])
        normalized_vector = direction_vector / norm
        
        # Scale the normalized vector by the desired speed to get the velocity command

        if path is None or len(path) < 5:
            desired_speed = 1 * desired_speed
        else:
            desired_speed = 1.5 * desired_speed
        velocity_command = normalized_vector * desired_speed

        # if close_to_interaction_point:
        #     velocity_command = 0.5 * velocity_command

        # print("velocity_command: ", velocity_command)
        # check if this is nan, if yes, change to zeros
        if np.isnan(velocity_command[0]):
            logger.warning("Velocity command contains NaN; returning zero velocity.")
            return False, np.array([0, 0])

        return True, velocity_command

    def define_interaction_points(self):
        # Interaction points and their corresponding actions are defined with respect to the T-shape's geometry
        # Define the interaction points on the T-shape relative to its center of mass
        length = 4
        scale = 30
        inflation_goal_parameter = 10
        # print("center of gravity: ", self.block.center_of_gravity)  # (0, 45)
        self.interaction_points = {
            'top_center': {'point': (0, 0), 'action': 'push_y_up', 'movement': (0, -inflation_goal_parameter), 'selected': False},
            'bottom_center': {'point': (0, length * scale), 'action': 'push_y_down', 'movement': (0, length * scale + inflation_goal_parameter), 'selected': False},
            # 'right_center': {'point': (0.5 * length * scale, 0.5 * scale), 'action': 'push_left', 'movement': 'translate_left'},
            # 'left_center': {'point': (-0.5 * length * scale,  0.5 * scale), 'action': 'push_right', 'movement': 'translate_right'},
            'right_center': {'point': (0.7 * scale, 1.6 * scale), 'action': 'push_x_left', 'movement': (0.6 * scale + inflation_goal_parameter, 1.6 * scale), 'selected': False},
            'left_center': {'point': (-0.7 * scale,  1.6 * scale), 'action': 'push_x_right', 'movement': (-0.6 * scale - inflation_goal_parameter, 1.6 * scale), 'selected': False},
            # Add points and actions for rotation if needed
            'top_right': {'point': (0.5 * scale, 0.9* length * scale), 'action': 'rotate_clockwise', 'movement': (0.5 * scale + 1.5*inflation_goal_parameter, 0.9* length * scale), 'selected': False},
            'top_left': {'point': (-0.5 * scale, 0.9*length * scale), 'action': 'rotate_counterclockwise', 'movement': (-0.5 * scale - 1.5*inflation_goal_parameter, 0.9*length * scale), 'selected': False},
            # ...
        }
        # Transform the interaction points based on the current position and orientation of the block
        for key in self.interaction_points:
            interaction = self.interaction_points[key]
            interaction['global_point'] = self.block.local_to_world(interaction['point'])

    def define_circle_sample_points(self):
        # Assuming length and scale are already defined
        n = 21
        length = 4
        scale = 30
        radius = 0.75* length * scale  # Example radius, adjust as necessary
        center_x = 0
        center_y = 0.5 * length * scale
        points = {}

        for i in range(n):
            angle_rad = 2 * np.pi * i / n  # Angle in radians for each point
            x = center_x + radius * np.cos(angle_rad)  # X coordinate
            y = center_y + radius * np.sin(angle_rad)  # Y coordinate

            # Define the point and any associated action, here just illustrative
            points[f'point_{i}'] = {'point': (x, y), 'action': f'action_{i}', 'selected': False}

        self.circle_points = points

        # Optionally transform the interaction points based on the current position and orientation of the circle
        for key in self.circle_points:
            interaction = self.circle_points[key]
            # Assuming circle object has a method `local_to_world` similar to block
            interaction['global_point'] = self.block.local_to_world(interaction['point'])


    def draw_interaction_points(self, ):
        # Define the color and size for the interaction point markers
        marker_color = (255, 0, 0)  # Red color
        marker_color2 = (0, 255, 0)
        marker_radius = 5  # Size of the marker
        
        
        # Make sure to call define_interaction_points before drawing
        # self.define_interaction_points()
        # self.select_interaction_point()
        # self.define_circle_sample_points()
        # # self.find_closest_point_and_divide(self.agent.position)
        # i_point= self.select_interaction_point()
        # desired_interaction_position = i_point['global_point']
        # self.find_two_closest_points_and_divide(self.agent.position, desired_interaction_position)
        # obstacles = self.get_obstacles()
        # adjusted_goal_point = self.adjust_goal_point(desired_interaction_position,obstacles, 10 )
        # print("adjusted_goal_point: ", adjusted_goal_point, " desired_interaction_position: ", desired_interaction_position)

        # path = self.a_star_search(self.agent.position, adjusted_goal_point, obstacles)
        # print("path: ", path)
        if self.interaction_points is not None:
            # Draw markers for each interaction point
            for interaction_name, interaction_info in self.interaction_points.items():
                global_point = interaction_info['global_point']
                # Convert the point to Pygame coordinates
                pygame_point = pymunk.pygame_util.to_pygame(Vec2d(*global_point), self.screen)
                # Draw a circle at the interaction point
                if interaction_info['selected']:
                    pygame.draw.circle(self.screen, marker_color, pygame_point, marker_radius+3)
                else: 
                    pygame.draw.circle(self.screen, marker_color, pygame_point, marker_radius)

        if self.circle_points is not None:
            # Draw markers for each interaction point
            for interaction_name, interaction_info in self.circle_points.items():
                global_point = interaction_info['global_point']
                # Convert the point to Pygame coordinates
                pygame_point = pymunk.pygame_util.to_pygame(Vec2d(*global_point), self.screen)
                # Draw a circle at the interaction point
                if interaction_info['selected']:
                    pygame.draw.circle(self.screen, marker_color2, pygame_point, marker_radius)
                else:
                    pygame.draw.circle(self.screen, marker_color, pygame_point, marker_radius)

        # draw the path cadidate 
        if self.low_level_path_candidates is not None:
            for p in range(0, len(self.low_level_path_candidates)):
                path = self.low_level_path_candidates[p]
                if path is not None:
                    for i in range(len(path) - 1):
                        pygame.draw.line(self.screen, (0, 0, 255), path[i], path[i+1], 5)  # Red path

        # Draw the path
        if self.path is not None:
            path = self.path
            for i in range(len(path) - 1):
                pygame.draw.line(self.screen, (255, 0, 0), path[i], path[i+1], 5)  # Red path


 
    def find_two_closest_points_and_divide_sorted_simple(self, global_point1, global_point2):
        # Find the closest points to the given global points
        closest_point1 = self.find_closest_point(global_point1)
        closest_point2 = self.find_closest_point(global_point2)

        if not closest_point1 or not closest_point2:
            return None, [], []  # If closest points were not found

        # Calculate the coefficients for the line equation defined by closest_point1 and closest_point2
        A = closest_point2[1] - closest_point1[1]
        B = closest_point1[0] - closest_point2[0]
        C = (closest_point2[0] * closest_point1[1]) - (closest_point1[0] * closest_point2[1])

        points_with_side_and_distance = []

        for interaction_name, value in self.circle_points.items():
            point = value['global_point']
            side = A * point[0] + B * point[1] + C
            distance_to_global_point1 = math.sqrt((point[0] - global_point1[0]) ** 2 + (point[1] - global_point1[1]) ** 2)
            points_with_side_and_distance.append((point, side, distance_to_global_point1))
            if side < 0:
                value['selected'] = True
            else:
                value['selected'] = False

        # Sort the points by distance to global_point1 within their respective sides
        points_with_side_and_distance.sort(key=lambda x: x[2])

        set1 = []
        set2 = []

        # Append points to set1 or set2 based on their side, only including the point itself
        for point, side, _ in points_with_side_and_distance:
            if side < 0:
                set1.append(point)
            else:
                set2.append(point)

        return set1, set2

    def find_closest_point(self, global_point):
        min_distance = float('inf')
        closest_point_key = None
        closest_global_point = None

        for key, value in self.circle_points.items():
            point = value['global_point']
            distance = math.sqrt((point[0] - global_point[0]) ** 2 + (point[1] - global_point[1]) ** 2)
            if distance < min_distance:
                min_distance = distance
                closest_point_key = key
                closest_global_point = point

        return closest_global_point

    def select_interaction_point(self):
        # Assume thresholds have been defined to decide when to translate vs. rotate
        POSITION_THRESHOLD = 0.15
        ORIENTATION_THRESHOLD = 0.05

        # self.select_interaction_point_list()

        position_error = np.linalg.norm(np.array(self.block.position) - np.array(self.goal_pose[:2]))

        # Calculate the orientation error taking into account the angular periodicity
        orientation_error = (self.goal_pose[2] - self.block.angle) % (2 * np.pi)
        orientation_error = (orientation_error + np.pi) % (2 * np.pi) - np.pi  # Normalize to the range [-pi, pi]

        direction_vector = np.array(self.goal_pose[:2]) - np.array(self.block.position)
        direction_vector /= np.linalg.norm(direction_vector)

        # print("direction_vector: ", direction_vector)
        goal_local = self.block.world_to_local((self.goal_pose[0], self.goal_pose[1])) - self.block.world_to_local(self.block.position)
        # goal_local /= np.linalg.norm(goal_local)
        # print("goal local: ", goal_local)
        horizontal_push = abs(goal_local[0]) > abs(goal_local[1])
        vertical_push = not horizontal_push

        # make the simulated teacher consistent to one sub-goal (and remove the error w.r.t this interaction point)
        if self.last_interaction_points is not None:
            # check whether the error is reduced
            if self.last_interaction_points['action'].startswith('rotate') and abs(orientation_error) > ORIENTATION_THRESHOLD:
                # Continue with the last selected rotation action
                if orientation_error > 0:
                    self.interaction_points['top_right']['selected'] = True
                    # return self.interaction_points['top_right']  # Need to rotate clockwise
                    return 'top_right'
                else:
                    self.interaction_points['top_left']['selected'] = True
                    return 'top_left'
            elif self.last_interaction_points['action'].startswith('push_x_left') and goal_local[0] < -10 * POSITION_THRESHOLD:
                # Continue with the last selected push action
                
                self.interaction_points['right_center']['selected'] = True
                # return self.interaction_points['right_center']  # Need to push leftwards
                return 'right_center'
            
            elif self.last_interaction_points['action'].startswith('push_x_right') and goal_local[0] > 10 * POSITION_THRESHOLD:

                self.interaction_points['left_center']['selected'] = True
                # return self.interaction_points['left_center']   # Need to push rightwards
                return 'left_center'
            
            elif self.last_interaction_points['action'].startswith('push_y_down') and goal_local[1] < -10 * POSITION_THRESHOLD:
                # Continue with the last selected push action
                
                self.interaction_points['bottom_center']['selected'] = True
                # return self.interaction_points['bottom_center'] # Need to push upwards
                return 'bottom_center'
            elif self.last_interaction_points['action'].startswith('push_y_up') and goal_local[1] > 10 * POSITION_THRESHOLD:
                self.interaction_points['top_center']['selected'] = True
                # return self.interaction_points['top_center']    # Need to push downwards
                return 'top_center'
            

        if abs(orientation_error) > ORIENTATION_THRESHOLD:
            # If orientation error is above the threshold, decide which way to rotate
            if orientation_error > 0:
                self.interaction_points['top_right']['selected'] = True
                self.last_path = None
                # return self.interaction_points['top_right']  # Need to rotate clockwise
                return 'top_right'
            else:
                self.interaction_points['top_left']['selected'] = True
                self.last_path = None
                # return self.interaction_points['top_left']   # Need to rotate counterclockwise
                return 'top_left'
        elif position_error > POSITION_THRESHOLD:
            # If position error is above the threshold, decide which way to translate
            # Calculate the direction vector from the T-shape's center to the goal's center
            # direction_vector = np.array(self.goal_pose[:2]) - np.array(self.block.position)
            # # Normalize the vector
            # direction_vector /= np.linalg.norm(direction_vector)
            # Check the primary direction of the required movement

            # if horizontal_push:
            #     # Movement is primarily in the horizontal direction
            #     if goal_local[0] < 0:
            #         self.interaction_points['right_center']['selected'] = True
            #         self.last_path = None
            #         # return self.interaction_points['right_center']  # Need to push leftwards
            #         return 'right_center'
            #     else:
            #         self.interaction_points['left_center']['selected'] = True
            #         self.last_path = None
            #         # return self.interaction_points['left_center']   # Need to push rightwards
            #         return 'left_center'
            # else:
            #     # Movement is primarily in the vertical direction
            #     if goal_local[1] < 0:
            #         self.interaction_points['bottom_center']['selected'] = True
            #         self.last_path = None
            #         # return self.interaction_points['bottom_center'] # Need to push upwards
            #         return 'bottom_center'
            #     else:
            #         self.interaction_points['top_center']['selected'] = True
            #         self.last_path = None
            #         # return self.interaction_points['top_center']    # Need to push downwards
            #         return 'top_center'

            # interaction_list_ = []  # check which one the robot's action is more closer
            # if horizontal_push:
            #     # Movement is primarily in the horizontal direction
            #     if goal_local[0] < 0:
            #         self.interaction_points['right_center']['selected'] = True
            #         self.last_path = None
            #         interaction_list_.append('right_center')
            #         # return self.interaction_points['right_center']  # Need to push leftwards
            #         # return 'right_center'
            #     else:
            #         self.interaction_points['left_center']['selected'] = True
            #         self.last_path = None
            #         interaction_list_.append('left_center')
            #         # return self.interaction_points['left_center']   # Need to push rightwards
            #         # return 'left_center'
            # else:
            #     # Movement is primarily in the vertical direction
            #     if goal_local[1] < 0:
            #         self.interaction_points['bottom_center']['selected'] = True
            #         self.last_path = None
            #         interaction_list_.append('bottom_center')
            #         # return self.interaction_points['bottom_center'] # Need to push upwards
            #         # return 'bottom_center'
            #     else:
            #         self.interaction_points['top_center']['selected'] = True
            #         self.last_path = None
            #         interaction_list_.append('top_center')
            #         # return self.interaction_points['top_center']    # Need to push downwards
            #         # return 'top_center'
            interaction_list_ = []  # check which one the robot's action is more closer
            # if abs(goal_local[0]) > 0.2 *  POSITION_THRESHOLD:  # too strict requirement, the
            if abs(goal_local[0]) >  5 * POSITION_THRESHOLD and abs(goal_local[1]) >  5 * POSITION_THRESHOLD:
                # Movement is primarily in the horizontal direction
                if goal_local[0] < 0:
                    self.interaction_points['right_center']['selected'] = True
                    self.last_path = None
                    interaction_list_.append('right_center')
                    # return self.interaction_points['right_center']  # Need to push leftwards
                    # return 'right_center'
                else:
                    self.interaction_points['left_center']['selected'] = True
                    self.last_path = None
                    interaction_list_.append('left_center')
                    # return self.interaction_points['left_center']   # Need to push rightwards
                    # return 'left_center'

                # Movement is primarily in the vertical direction
                if goal_local[1] < 0:
                    self.interaction_points['bottom_center']['selected'] = True
                    self.last_path = None
                    interaction_list_.append('bottom_center')
                    # return self.interaction_points['bottom_center'] # Need to push upwards
                    # return 'bottom_center'
                else:
                    self.interaction_points['top_center']['selected'] = True
                    self.last_path = None
                    interaction_list_.append('top_center')
                    # return self.interaction_points['top_center']    # Need to push downwards
                    # return 'top_center'

                if vertical_push:  # make sure the 0-th item of the interaction_list_ has the largest error
                    interaction_list_ = interaction_list_[::-1]
            else:
            # if abs(goal_local[0]) <  5 * POSITION_THRESHOLD and abs(goal_local[1]) <  5 * POSITION_THRESHOLD:
                if horizontal_push:
                    # Movement is primarily in the horizontal direction
                    if goal_local[0] < 0:
                        self.interaction_points['right_center']['selected'] = True
                        self.last_path = None
                        interaction_list_.append('right_center')
                        # return self.interaction_points['right_center']  # Need to push leftwards
                        # return 'right_center'
                    else:
                        self.interaction_points['left_center']['selected'] = True
                        self.last_path = None
                        interaction_list_.append('left_center')
                        # return self.interaction_points['left_center']   # Need to push rightwards
                        # return 'left_center'
                else:
                    # Movement is primarily in the vertical direction
                    if goal_local[1] < 0:
                        self.interaction_points['bottom_center']['selected'] = True
                        self.last_path = None
                        interaction_list_.append('bottom_center')
                        # return self.interaction_points['bottom_center'] # Need to push upwards
                        # return 'bottom_center'
                    else:
                        self.interaction_points['top_center']['selected'] = True
                        self.last_path = None
                        interaction_list_.append('top_center')
                        # return self.interaction_points['top_center']    # Need to push downwards
                        # return 'top_center'
            return interaction_list_
        else:
            # If both errors are below their thresholds, no movement is needed
            return None  # No interaction point necessary
        
    
    def select_interaction_point_list(self):
        # Assume thresholds have been defined to decide when to translate vs. rotate
        POSITION_THRESHOLD = 0.15
        ORIENTATION_THRESHOLD = 0.05

        position_error = np.linalg.norm(np.array(self.block.position) - np.array(self.goal_pose[:2]))

        # Calculate the orientation error taking into account the angular periodicity
        orientation_error = (self.goal_pose[2] - self.block.angle) % (2 * np.pi)
        orientation_error = (orientation_error + np.pi) % (2 * np.pi) - np.pi  # Normalize to the range [-pi, pi]

        direction_vector = np.array(self.goal_pose[:2]) - np.array(self.block.position)
        direction_vector /= np.linalg.norm(direction_vector)

        # print("direction_vector: ", direction_vector)
        goal_local = self.block.world_to_local((self.goal_pose[0], self.goal_pose[1])) - self.block.world_to_local(self.block.position)

        selected_list = []
        if abs(orientation_error) > ORIENTATION_THRESHOLD:
            # If orientation error is above the threshold, decide which way to rotate
            if orientation_error > 0:
                selected_list.append('top_right')
            else:
                selected_list.append('top_left')
        if position_error > POSITION_THRESHOLD:
            
            if goal_local[0] < 0:
                selected_list.append('right_center')
            else:
                selected_list.append('left_center')
            
            # Movement is primarily in the vertical direction
            if goal_local[1] < 0:
                selected_list.append('bottom_center')
            else:
                selected_list.append('top_center')
        
        # print('selected_list: ', selected_list)
        self.selected_interaction_set = []


    def create_inflated_t_shape(self, inflation_distance):
        # Inflate the T-shape obstacle by a certain distance
        # Assuming self.block is the T-shape object and it has vertices attribute
        t_shape = pymunk_to_shapely(self.block, self.block.shapes)
        inflated_t_shape = t_shape.buffer(inflation_distance)
        return inflated_t_shape

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
        # Combine the T-shape and walls into a list of obstacles
        inflation_distance = 20.5  # Define the safe distance
        wall_thickness = 100 # should be thinker than the grid_resolution of A-star algorithm, otherwise the searched grid can be outside of the walls 
        t_shape = self.create_inflated_t_shape(20.5)
        inflated_t_shape = self.create_inflated_t_shape(inflation_distance)
        walls = self.create_walls(wall_thickness)
        slack_obstacles = [inflated_t_shape]
        strict_obstacles = [t_shape] + walls
        return slack_obstacles, strict_obstacles
    
    def heuristic(self, a, b):
        # Use Euclidean distance as the heuristic
        return np.linalg.norm(np.array(a) - np.array(b))
    
    def heuristic_close_to_last_path(self, node_a):
        # Check if last_path is not empty
        # if self.path is None or len(self.last_path) < 10:
        if self.last_path is None or len(self.last_path) < 10:
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
            if i < 9:
                continue
            # Calculate distance from node_a to the current node in the loop
            distance = self.heuristic(node_a, node) # Assuming a distance_to method exists
            # Update min_distance if the current distance is smaller
            if distance < min_distance:
                min_distance = distance
        # Return the minimum distance found
        return min_distance
    
    def heuristic_close_to_circle_sets(self, node_a, set_circle):
        if self.last_path is not None:
            return 0.0
        # Check if last_path is not empty
        # if self.path is None or len(self.last_path) < 10:
        if set_circle is None or len(set_circle) < 4:
            # Return a high heuristic value if last_path is empty,
            # indicating that node_a is "far" from an empty path.
            # This value could be adjusted depending on the specifics of your implementation.
            return 0.0
        
        # Initialize minimum distance to a very high value
        min_distance = float('inf')
        
        # Iterate over each node in the last_path
        i = 0
        for node in set_circle:
            i = i + 1
            if i < 4:
                continue
            # Calculate distance from node_a to the current node in the loop
            distance = self.heuristic(node_a, node) # Assuming a distance_to method exists
            # Update min_distance if the current distance is smaller
            if distance < min_distance:
                min_distance = distance
        # Return the minimum distance found
        return min_distance
    
    def a_star_search(self, start, goal, slack_obstacles, strict_obstacles, circle_set = None):
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
            # print("current: ", current, " goal: ", goal)
            # if current == goal:
            if self.heuristic(current, goal) < 25:
                # print("self.heuristic(current, goal): ", self.heuristic(current, goal))
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            for neighbor in self.get_neighbors(current, slack_obstacles, strict_obstacles):
                tentative_g_score = current_cost + self.heuristic(current, neighbor) + \
                                     0.25 * self.heuristic_close_to_last_path(neighbor) +\
                                     10.0 * self.heuristic_close_to_circle_sets(neighbor, circle_set)
                if self.is_within_obstacles(neighbor, slack_obstacles):  # another heursitcs for obstacles
                    tentative_g_score = tentative_g_score + 3

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, tentative_g_score, neighbor))

        return None

    def get_neighbors(self, node, slack_obstacles, strict_obstacles):
        # This would depend on your grid resolution
        grid_resolution = 10
        directions = [(0, grid_resolution), (0, -grid_resolution), 
                      (grid_resolution, 0), (-grid_resolution, 0), 
                      (grid_resolution, grid_resolution), (-grid_resolution, -grid_resolution), 
                      (grid_resolution, -grid_resolution), (-grid_resolution, grid_resolution)]
        
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

    # def adjust_goal_point(self, goal_point, obstacles, inflation_distance):
    #     # Create an inflated goal point
    #     gt= sg.Point(goal_point)
    #     inflated_goal = sg.Point(goal_point).buffer(inflation_distance)
    #     for obstacle in obstacles:
    #         print(obstacle.contains(inflated_goal) )
    #     # If the inflated goal is not fully inside any obstacle, 
    #     # find the closest point on its boundary to the original goal that is not inside an obstacle
    #     if any(obstacle.contains(inflated_goal) for obstacle in obstacles):
    #         # print("adjust adjust_goal_point")
    #         if any(obstacle.intersects(inflated_goal) for obstacle in obstacles):
    #             # The inflated goal intersects with an obstacle but is not fully contained within it
    #             # We need to find the closest point to the original goal that is outside the obstacles
    #             boundary = inflated_goal.boundary
    #             min_dist = float('inf')
    #             closest_point = None
    #             for point in boundary.coords:
    #                 pt = sg.Point(point)
    #                 dist = gt.distance(pt)
    #                 # print("dist: ", dist)
    #                 if dist < min_dist and any(obstacle.contains(pt) for obstacle in obstacles):
    #                     closest_point = pt
    #                     min_dist = dist

    #             # Return the adjusted goal point
    #             return closest_point.x, closest_point.y
        
    #     # If the original goal point is outside all obstacles, return it
    #     return goal_point

    def adjust_goal_point(self, selected_interaction, obstacles, inflation_distance):
        # Extract the goal point and the predefined movement direction
        goal_point = selected_interaction['point']
        move_direction = selected_interaction['movement']
        move_direction_global = self.block.local_to_world(move_direction) #- goal_point
        # Invert the movement direction to move away from the T-shape
        adjusted_direction = (move_direction_global[0], move_direction_global[1])

        # goal_point = goal_point + adjusted_direction
        # Create an inflated goal point
        # inflation_distance = 3
        # inflated_goal = sg.Point(goal_point).buffer(inflation_distance)

        # # Check if the inflated goal intersects with the obstacles
        # # for obstacle in obstacles:
        # #     if inflated_goal.intersects(obstacle):
        # #         # If so, adjust the goal point in the opposite direction of the movement
        # #         while inflated_goal.intersects(obstacle):
        # #             goal_point = (goal_point[0] + adjusted_direction[0] * inflation_distance,
        # #                           goal_point[1] + adjusted_direction[1] * inflation_distance)
        # #             inflated_goal = sg.Point(goal_point).buffer(inflation_distance)
        # #         return goal_point
        
        # while any(obstacle.contains(inflated_goal) for obstacle in obstacles):
        #     goal_point = (goal_point[0] + adjusted_direction[0] * inflation_distance,
        #                           goal_point[1] + adjusted_direction[1] * inflation_distance)
        #     inflated_goal = sg.Point(goal_point).buffer(inflation_distance)

        # If the original goal point does not intersect with obstacles, return it
        return move_direction_global
    
    def save_image(self, save_path):
        # Save the screen to an image file
        image_filename = save_path
        pygame.image.save(self.screen, image_filename)
        return None
