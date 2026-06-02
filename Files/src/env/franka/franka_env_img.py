import logging

logger = logging.getLogger(__name__)

import math
import numpy as np
import rospy

from geometry_msgs.msg import PoseStamped


# from env.panda.panda_interface import Panda
from env.franka.panda import Panda  # if your Panda class is in the same folder

from env.realsense_Image_receiver import ImageReceiver
from env.robotsuite.env_robosuite import combine_obs_dicts, rot6d_to_axisangle
from env.franka.pose_transform_functions import array_quat_2_pose
from env.franka.pose_transform_functions import get_quaternion_from_euler, axis_angle_to_euler
import quaternion  # numpy-quaternion


class PANDAenv_pushT_img:
    """
    Franka Panda env with the same interface & logic style as KUKAenv_pushT_img.

    Obs:
        - dict with RGB images (image1, image2, ...) and low-dim entries like 'robot0_eef_pos_vel'
          where robot0_eef_pos_vel = [x, y, vx, vy] (in world frame, normalized if abs_action is used)

    Actions:
        - 2D action (dx, dy) if use_abs_action == False
        - 2D absolute xy position (normalized) if use_abs_action == True
    """

    def __init__(self, config):
        # ========= Config / shape_meta parsing =========
        shape_meta = config.shape_meta
        self.abs_action = config.use_abs_action

        rgb_keys = []
        lowdim_keys = []
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            t = attr.get("type", "low_dim")
            if t == "rgb":
                rgb_keys.append(key)
            elif t == "low_dim":
                lowdim_keys.append(key)
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys

        # ========= Robot interface =========
        self.robot = Panda()  # your existing low-level interface

        # ========= Cameras =========
        # TODO: adjust topics if needed
        self.receiver_img1 = ImageReceiver("/camera1/color/image_raw/compressed")
        self.receiver_img2 = ImageReceiver(
            "/camera2/color/image_raw/compressed", enable_crop=False
        )

        # ========= Internal state =========
        # We'll read directly from self.robot.curr_pos, self.robot.curr_ori
        self.ee_velocity = np.zeros(3)
        self.prev_ee_pos = None
        self.prev_time = None

        self.ee_goal_position = None
        self.use_simulation = False  # keep for compatibility with your other envs
        self.obs_dict_past = None
        self.count = 0

        # ========= Control parameters =========
        # Same semantics as KUKA env
        self.controller = "cartesian"  # we control Cartesian equilibrium pose directly
        self.control_orientation = False if shape_meta['action']['shape'][0] <= 3 else True
        self.control_position_z = False if shape_meta['action']['shape'][0] == 2 else True
        self.scale = 0.05  # similar to KUKA's scale
        self.gripper_scale = 0.01

        # Position limits for Panda end-effector [x_low, x_high, y_low, y_high, z_low, z_high]
        # TODO: tune for your table / workspace
        self.ee_pos_limit_xyz = [0.3, 0.8, -0.4, 0.4, 0.05, 0.5]

        # For abs-action normalization (only x, y)
        self.action_min = [self.ee_pos_limit_xyz[0], self.ee_pos_limit_xyz[2], self.ee_pos_limit_xyz[4]]
        self.action_max = [self.ee_pos_limit_xyz[1], self.ee_pos_limit_xyz[3], self.ee_pos_limit_xyz[5]]

        # Desired constant z height when not controlling z explicitly
        self.position_goal_z = 0.45

        # Orientation: keep a fixed downward orientation (similar to your Panda.home())
        # home quat = np.quaternion(0, 1, 0, 0)
        self.orientation_goal_quat = np.quaternion(0, 1, 0, 0)

        # If you want to gradually move joints in reset, you can add something like KUKA's q_goal_reset here.
        # For now we'll just use Panda.home()
        self.rest_controller_gain = 0.35

    # ==================== Observation helpers ====================

    def _get_obs_dict(self):
        obs_dict = {}

        # ---- RGB observations ----
        for key in self.rgb_keys:
            if key in ["image1"]:
                image = self.receiver_img1.image
            elif key in ["image2"]:
                image = self.receiver_img2.image
            else:
                # Unknown key, skip
                continue

            # (H, W, C) -> (C, H, W), normalize to [-1, 1]
            img = np.moveaxis(image, -1, 0).astype(np.float32) / 255.0
            img = (2.0 * img - 1.0).astype(np.float32)
            obs_dict[key] = img

        # ---- Low-dim observations ----
        for key in self.lowdim_keys:
            if key in ["robot0_eef_pos_vel"]:
                obs = self._get_eef_pose()
                if self.abs_action:
                    # Normalize x, y (first two entries) into [-1, 1] using action_min/action_max
                    obs[:3] = self.normalize_abs_action(
                        obs[:3], min_list=self.action_min, max_list=self.action_max
                    )
                obs_dict[key] = obs

        if self.obs_dict_past is None:
            self.obs_dict_past = obs_dict

        # Combine current and past obs as in KUKA env
        obs_dict_combined = combine_obs_dicts(self.obs_dict_past, obs_dict)
        self.obs_dict_past = obs_dict

        return obs_dict_combined

    def _get_eef_pose(self):
        """
        Returns [x, y, vx, vy].
        Velocity is estimated by finite differences on the Panda's ee pose.
        """
        if self.robot.curr_pos is None:
            logger.debug("No Panda ee pose yet, returning zeros.")
            return np.zeros(6, dtype=np.float32)

        pos = np.array(self.robot.curr_pos).copy()  # [x, y, z]
        ori = self.robot.curr_ori  #TODO: whether to change this to euler angle?

        return np.concatenate([pos, ori, np.array([self.robot.gripper_width])], axis=0)

    # ==================== Normalization helpers ====================

    def normalize_abs_action(self, action_, min_list, max_list):
        # arr = np.asarray(action, dtype=float)
        action = action_.copy()
        lo  = np.asarray(min_list, dtype=np.float32)
        hi  = np.asarray(max_list, dtype=np.float32)
        # centers and half-ranges
        center = (hi + lo) / 2.0
        half_range = (hi - lo) / 2.0
        # avoid division by zero
        half_range[half_range == 0] = 1.0
        action[0:3] = ((action[0:3] - center) / half_range)
        return action

    def unnormalize_abs_action(self, norm_action_, min_list, max_list):
        norm_action = norm_action_.copy()
        arr = np.asarray(norm_action, dtype=np.float32)
        lo  = np.asarray(min_list, dtype=np.float32)
        hi  = np.asarray(max_list, dtype=np.float32)
        # compute half-range and center
        half_range = (hi - lo) / 2.0
        center     = (hi + lo) / 2.0
        # avoid division-by-zero artifacts (if hi==lo, half_range==0)
        # but here we only need half_range for multiplication, so it’s fine:
        arr[0:3] = (arr[0:3]  * half_range + center)
        return arr

    # ==================== Core env API ====================

    def render(self):
        # For real robot, no special rendering
        return None

    def step(self, action):
        """
        action: np.ndarray, shape (2,)
            - if abs_action == False: delta in x,y (roughly in [-1, 1]) scaled by self.scale
            - if abs_action == True: normalized absolute x,y ∈ [-1, 1]
        """
        action_r = np.array(action, dtype=np.float32).copy()

        if self.abs_action:
            # Unnormalize from [-1, 1] to actual xy workspace
            action_r = self.unnormalize_abs_action(
                action_r, min_list=self.action_min, max_list=self.action_max
            )


        # Run control
        action_r_pos = action_r[:3]
        action_ori = action_r[3:-1]
        hand_command = action_r[-1]
        self.run(action_r_pos, action_ori, hand_command)

        reward = 0.0
        self.count += 1
        done = False  # you use human to terminate episodes
        terminated = False
        info = {"success": False}

        if done or terminated or info["success"]:
            self.hold_on_mode()

        obs_dict = self._get_obs_dict()
        return [obs_dict, reward, done, terminated, info]

    def hold_on_mode(self):
        """
        Keep Panda at current pose by re-publishing the same equilibrium pose.
        """
        if self.robot.curr_pos is None or self.robot.curr_ori is None:
            return

        rate_hz = 200
        rate = rospy.Rate(rate_hz)

        pos_array = np.array(self.robot.curr_pos)
        quat = np.quaternion(
            self.robot.curr_ori[0],
            self.robot.curr_ori[1],
            self.robot.curr_ori[2],
            self.robot.curr_ori[3],
        )
        for _ in range(500):
            goal = array_quat_2_pose(pos_array, quat)
            goal.header.seq = 1
            goal.header.stamp = rospy.Time.now()
            self.robot.goal_pub.publish(goal)
            rate.sleep()

    def reset(self):
        """
        Reset the Panda to some home configuration and return initial obs_dict, info.
        """
        self.obs_dict_past = None
        self.count = 0

        # Use your existing Panda.home() routine
        logger.debug("Reset: moving Panda to home pose...")
        self.robot.home()

        # Let everything settle
        rospy.sleep(1.0)

        # Reset velocity estimation
        self.prev_ee_pos = None
        self.prev_time = None
        self.ee_goal_position = None
        self.goal_gripper_command = self.robot.curr_grip_width

        obs_dict = self._get_obs_dict()
        info = {}
        return [obs_dict, info]

    # ==================== Control logic ====================

    def run(self, ee_action_xy, ee_action_ori, hand_command):
        """
        Compute a new Cartesian goal for Panda and send it as an equilibrium pose.
        ee_action_xy: np.ndarray shape (2,)  -> desired x,y or delta x,y depending on abs_action.
        """
        if self.robot.curr_pos is None:
            logger.debug("Panda ee pose not ready, skipping control step.")
            return False

        ee_position = np.array(self.robot.curr_pos).copy()  # current [x, y, z]

        
        if not self.abs_action:
            # delta mode
            ee_delta = np.zeros(3, dtype=np.float32)
            ee_delta[0:2] = ee_action_xy[0:2]  # assumed ~[-1, 1]

            if not self.control_position_z:
                # keep z around self.position_goal_z
                ee_position_z = ee_position[2]
                z_goal = self.position_goal_z
                ee_delta[2] = (z_goal - ee_position_z) / self.scale
            else: 
                ee_delta[2] = ee_action_xy[2]

            if self.ee_goal_position is None:
                self.ee_goal_position = ee_position.copy()

            if np.linalg.norm(ee_delta[:3]) > 0.01:
                self.ee_goal_position = ee_position + ee_delta * self.scale

            self.goal_gripper_command = self.goal_gripper_command  + hand_command * self.gripper_scale

            quat_goal=np.quaternion(self.robot.curr_ori[0],self.robot.curr_ori[1],self.robot.curr_ori[2],self.robot.curr_ori[3])
            # ==== Orientation ====
            if self.control_orientation:
                logger.debug('%s %s', "ee_action_ori: ", ee_action_ori)
                q_delta_array=get_quaternion_from_euler(ee_action_ori[0], ee_action_ori[1], ee_action_ori[2])
                q_delta=np.quaternion(q_delta_array[0],q_delta_array[1],q_delta_array[2],q_delta_array[3]) 
                quat_goal=q_delta*quat_goal

                # # NEW: keyboard-based rotation around WORLD z-axis
                # if self.rot_command != 0:
                #     yaw = self.rot_command * self.rot_step  # + or - about world z
                #     qz_array = get_quaternion_from_euler(0.0, 0.0, yaw)
                #     qz_delta = np.quaternion(qz_array[0], qz_array[1], qz_array[2], qz_array[3])
                #     quat_goal = qz_delta * quat_goal        # left-multiply = world-frame z-rot


            else:
                quat_goal = self.orientation_goal_quat

        else:
            # absolute mode: ee_action_xy is desired x,y (in world coords)
            # self.ee_goal_position = ee_position.copy()
            self.ee_goal_position = ee_action_xy
            # self.ee_goal_position = ee_position

            ee_action_axis_angle = rot6d_to_axisangle(ee_action_ori)
            ee_action_ori = axis_angle_to_euler(ee_action_axis_angle)
            quat_goal = get_quaternion_from_euler(ee_action_ori[0], ee_action_ori[1], ee_action_ori[2])
            quat_goal = np.quaternion(quat_goal[0], quat_goal[1], quat_goal[2], quat_goal[3])

            # Limit step size relative to last obs (just like KUKA env)
            if self.obs_dict_past is not None:
                key = "robot0_eef_pos_vel"
                past_norm_xy = self.obs_dict_past[key][:3]
                past_xy = self.unnormalize_abs_action(
                    past_norm_xy, min_list=self.action_min, max_list=self.action_max
                )
                # print("past_xy: ", past_xy)
                desired_xy = self.ee_goal_position[:3]
                delta = desired_xy - past_xy
                dist = np.linalg.norm(delta)
                max_step = 0.04  # 2 cm
                if dist > max_step:
                    delta = delta / dist * max_step
                    logger.debug("Panda exceeded max_step in abs_action, clipping.")
                self.ee_goal_position[:3] = past_xy + delta

            self.goal_gripper_command = hand_command
        # ==== Apply workspace limits ====
        # x bounds
        self.ee_goal_position[0] = max(
            self.ee_pos_limit_xyz[0],
            min(self.ee_goal_position[0], self.ee_pos_limit_xyz[1]),
        )
        # y bounds
        self.ee_goal_position[1] = max(
            self.ee_pos_limit_xyz[2],
            min(self.ee_goal_position[1], self.ee_pos_limit_xyz[3]),
        )
        # z bounds
        self.ee_goal_position[2] = max(
            self.ee_pos_limit_xyz[4],
            min(self.ee_goal_position[2], self.ee_pos_limit_xyz[5]),
        )

        


        # ==== Send command as equilibrium pose ====
        goal = array_quat_2_pose(self.ee_goal_position, quat_goal)
        goal.header.seq = 1
        goal.header.stamp = rospy.Time.now()
        self.robot.goal_pub.publish(goal)

        # goal_grip = goal_grip + self.offset_gripper
        # print('gripper: ', self.goal_gripper_command, " width: ", self.robot.curr_grip_width)
        self.goal_gripper_command = np.maximum(np.minimum(self.goal_gripper_command, 0.09), -0.02)
        # self.offset_gripper = 0.0
        self.robot.move_gripper(self.goal_gripper_command)

        return True
