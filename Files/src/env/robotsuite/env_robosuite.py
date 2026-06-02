"""
This file contains the robosuite environment wrapper that is used
to provide a standardized environment API for training policies and interacting
with metadata present in datasets.
"""
import logging

logger = logging.getLogger(__name__)

import json
import numpy as np
from copy import deepcopy

import cv2 # for visualize the camera observation

# import mujoco_py
import robosuite
from robosuite import load_controller_config
# from robosuite.controllers import load_composite_controller_config  # only works in version >= 1.5.0

import env.robotsuite.obs_utils as ObsUtils
import env.robotsuite.env_base as EB
import robosuite.utils.transform_utils as T
from robosuite.utils.sim_utils import check_contact


# def rot6d_to_axisangle(rot6d):
#     r6 = np.asarray(rot6d)
#     flat6 = r6.reshape(3,2)

#     # split into two “basis” vectors
#     a1 = flat6[:, 0]
#     a2 = flat6[:, 1]

#     # Gram–Schmidt to get orthonormal b1, b2
#     eps = 1e-8
#     b1 = a1 / (np.linalg.norm(a1) if np.linalg.norm(a1) > eps else np.linalg.norm(a1) + eps)
#     proj = np.sum(b1 * a2) * b1
#     b2 = a2 - proj
#     b2 = b2 / (np.linalg.norm(b2) if np.linalg.norm(b2) > eps else np.linalg.norm(b2) + eps)

#     # third column via cross product
#     b3 = np.cross(b1, b2)

#     # build rotation matrix of shape (N,3,3)
#     mats = np.stack([b1, b2, b3], axis=-1)

#     # mat -> quaternion -> axis-angle
#     quat = T.mat2quat(mats)                # (N,4)
#     axisangle = T.quat2axisangle(quat)   # (N,3)

#     return axisangle

# def axisangle_to_rot6d(axisangle):
#     '''rotation_matrix[:, :2]
#     [ a1, b1,
#       a2, b2,
#       a3, b3]
#       ==> rotation_matrix[:, :2].reshape(-1): [a1, b1, a2, b2, a3, b3]
#     '''
#     rotation_matrix =  T.quat2mat(T.axisangle2quat(axisangle))
#     rot6d = rotation_matrix[:, :2].reshape(-1)
#     # import pdb 
#     # pdb.set_trace()
#     return rot6d

# from agents.DP_model.common.rotation_transformer import RotationTransformer

# rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')

# def rot6d_to_axisangle(rot6d):
#     r6 = np.asarray(rot6d)
#     axisangle = rotation_transformer.inverse(r6)

#     return axisangle


# def axisangle_to_rot6d(axisangle):
#     return rotation_transformer.forward(axisangle)

from scipy.spatial.transform import Rotation as R
def rot6d_to_axisangle(rot6d):
    r6 = np.asarray(rot6d)
    flat6 = r6.reshape(3,2)

    # split into two “basis” vectors
    # a1 = flat6[:, 0]
    # a2 = flat6[:, 1]
    a1 = r6[:3]
    a2 = r6[3:]

    # Gram–Schmidt to get orthonormal b1, b2
    eps = 1e-8
    b1 = a1 / (np.linalg.norm(a1) if np.linalg.norm(a1) > eps else np.linalg.norm(a1) + eps)
    proj = np.sum(b1 * a2) * b1
    b2 = a2 - proj
    b2 = b2 / (np.linalg.norm(b2) if np.linalg.norm(b2) > eps else np.linalg.norm(b2) + eps)

    # third column via cross product
    b3 = np.cross(b1, b2)

    # build rotation matrix of shape (N,3,3)
    mats = np.stack([b1, b2, b3], axis=-1)

    # mat -> quaternion -> axis-angle
    # quat = T.mat2quat(mats)                # (N,4)
    # axisangle = T.quat2axisangle(quat)   # (N,3)

    r = R.from_matrix(mats)
    axisangle =r.as_rotvec()

    return axisangle

def axisangle_to_rot6d(axisangle):
    '''rotation_matrix[:, :2]
    [ a1, b1,
      a2, b2,
      a3, b3]
      ==> rotation_matrix[:, :2].reshape(-1): [a1, b1, a2, b2, a3, b3]
    '''
    # rotation_matrix =  T.quat2mat(T.axisangle2quat(axisangle))
    r = R.from_rotvec(axisangle)
    rotation_matrix = r.as_matrix()

    rot6d = rotation_matrix[:, :2].reshape(-1)
    rot6d_a = rotation_matrix[:, 0]
    rot6d_b = rotation_matrix[:, 1]
    rot6d   = np.concatenate([rot6d_a, rot6d_b], axis=0)  # shape (6,)
    # import pdb 
    # pdb.set_trace()
    return rot6d

def visualize_two_cameras(obs_dict, obs_keys):
    """
    Displays two image observations side by side (or in separate windows).
    obs_dict[key] should be a float32 array in C×H×W, values in [0,1].
    """

    key1 = obs_keys[0]
    key2 = obs_keys[1]
    img1 = obs_dict[key1]
    img2 = obs_dict[key2]

    # Option A: show in separate windows
    cv2.imshow(f"Obs_{key1}", img1)
    cv2.imshow(f"Obs_{key2}", img2)

    # Option B: side-by-side in one window
    # combined = np.hstack((img1, img2))
    # cv2.imshow("Obs comparison", combined)

    # waitKey(1) for non‐blocking display in a loop, or 0 to wait until a keypress
    cv2.waitKey(1)

def quaternion_multiply(quat1, quat2):
    """Return multiplication of two quaternions."""
    x1, y1, z1, w1 = quat1
    x2, y2, z2, w2 = quat2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return np.array([x, y, z, w])

def quat_to_yaw(q):  # used to determine the z-angle of objects on the table
    """
    Given a quaternion q = [qx, qy, qz, qw], returns yaw (rotation about Z) in radians.
    """
    q = q / np.linalg.norm(q)
    qx, qy, qz, qw = q

    # Standard formula for ZYX (roll-pitch-yaw) convention:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy*qy + qz*qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return yaw

def check_whether_invalid_initial_state(obs_extracted):  # can be used in square task to filter out invalid configurations. Not used for now. 
    invalid_config = False
    nut_ori = T.mat2quat(obs_extracted[15:24].reshape(3, 3))
    nut_pos = obs_extracted[3:6]
    nut_z_angle = quat_to_yaw(nut_ori) * 180.0 / np.pi
    # if (nut_z_angle >= -10 and nut_z_angle < 90): # and nut_pos:
    if (nut_z_angle >= -20 and nut_z_angle < 95): # and nut_pos:
        invalid_config = True
    logger.debug('%s %s %s %s %s %s', "nut_pos: ", nut_pos, "nut_z_angle: ", nut_z_angle, " invalid_config: ", invalid_config)
    return invalid_config

def quaternion_conjugate(quat):
    """Return the conjugate of a quaternion."""
    x, y, z, w = quat
    return np.array([-x, -y, -z, w])

# used to combine the current and previous observation
def combine_obs_dicts(obs_dict_past, obs_dict_current):
    obs_dict_combined = {}
    for key in obs_dict_current:
        obs_dict_combined[key] = np.stack([obs_dict_past[key], obs_dict_current[key]], axis=0)
    return obs_dict_combined

def extract_latest_obs_dict(obs_dict_combined):
    """
    Given a dict where each value is a np.ndarray of shape (2, …)
    (0 = past, 1 = current), return a dict of just the "current" obs.
    """
    return { key: arr[1] for key, arr in obs_dict_combined.items() }

class EnvRobosuite(EB.EnvBase):
    """Wrapper class for robosuite environments (https://github.com/ARISE-Initiative/robosuite)"""
    def __init__(
        self, 
        # env_name='NutAssembly',
        env_name='NutAssemblySquare', 
        render=True, 
        # render_offscreen=True, 
        render_offscreen=False,
        # use_image_obs=False, 
        use_image_obs=True, 
        use_abs_action = False, 
        visualize_image_obs = False, 
        # visualize_image_obs = True, 
        obs_horizon = 1,
        postprocess_visual_obs=True, 
        config = None,
        **kwargs,
    ):
        """
        Args:
            env_name (str): name of environment. Only needs to be provided if making a different
                environment from the one in @env_meta.

            render (bool): if True, environment supports on-screen rendering

            render_offscreen (bool): if True, environment supports off-screen rendering. This
                is forced to be True if @env_meta["use_images"] is True.

            use_image_obs (bool): if True, environment is expected to render rgb image observations
                on every env.step call. Set this to False for efficiency reasons, if image
                observations are not required.

            postprocess_visual_obs (bool): if True, postprocess image observations
                to prepare for learning. This should only be False when extracting observations
                for saving to a dataset (to save space on RGB images for example).
        """
        self.postprocess_visual_obs = postprocess_visual_obs

        # robosuite version check
        self._is_v1 = (robosuite.__version__.split(".")[0] == "1")
        if self._is_v1:
            assert (int(robosuite.__version__.split(".")[1]) >= 2), "only support robosuite v0.3 and v1.2+"

        kwargs = deepcopy(kwargs)

        robot_config = 'Panda'
        self.num_robot = 1
        if env_name == 'TwoArmLift':
            robot_config = ["Panda", "Panda"]
            self.num_robot = 2
        # update kwargs based on passed arguments
            
        shape_meta = config.shape_meta
        self.limit_eef_vel = config.limit_eef_vel   # used when doing absolute control
        self.scale_vel_control = config.get("scale_vel_control", True)  # used when doing velocity control together with the oracle policy

        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys

        self.use_image_obs = use_image_obs
        self.visualize_image_obs = visualize_image_obs
        self.obs_extracted = None
        self.obs_horizon = obs_horizon
        self.obs_dict_past = None  # the size should be obs_horizon - 1, and if it is none, set to current obs_dict
        controller_config = load_controller_config(default_controller='OSC_POSE')
        # position_lim = 0.3 # doesn't work, it just clip the position and ori goal by these limits...
        # controller_config['position_limits'] = np.array([[-position_lim]*3, [ position_lim]*3])
        # controller_config['orientation_limits'] = np.array([[-position_lim]*3, [ position_lim]*3])

        # controller_config = load_controller_config(default_controller='IK_POSE')  # needs pybullet....
        # controller_config = load_composite_controller_config(robot="Panda")
        
        self.abs_action = use_abs_action
        # self.abs_action = False

        controller_config['control_delta'] = not self.abs_action # control the absolute pose
        # controller_config['kp'] = 100
        # controller_config['damping'] = 20
        # if self.abs_action:
        #     controller_config['uncouple_pos_ori'] =False
        # controller_config['input_ref_frame'] = 'world' # not working
        # controller_config["body_parts"]["right"]['input_ref_frame'] = "world"
        logger.debug('%s %s', "controller_config: ", controller_config)
        self.camera_name  = []
        if use_image_obs:
            self.camera_name = list(config.camera_names)
            update_kwargs = dict(
                horizon=2000, 
                control_freq=20,
                has_renderer=render,
                has_offscreen_renderer=(render_offscreen or use_image_obs),
                ignore_done=False,
                hard_reset=True,
                use_object_obs=True,
                use_camera_obs=use_image_obs,
                camera_depths=False,
                camera_names=self.camera_name,
                # camera_names = ["robot0_eye_in_hand", "robot1_eye_in_hand", "robot0_robotview", "robot1_robotview"],
                camera_heights=84,                      # image height
                camera_widths=84,                       # image width
                #robots = 'Panda',
                # robots = ["Panda", "Panda"],
                robots = robot_config, 
                renderer = 'mujoco',
                controller_configs = controller_config
            )
        else:
            update_kwargs = dict(
                horizon=2000, 
                control_freq=20,
                has_renderer=render,
                has_offscreen_renderer=(render_offscreen or use_image_obs),
                ignore_done=False,
                hard_reset=True,
                use_object_obs=True,
                use_camera_obs=False,
                camera_depths=False,
                #robots = 'Panda',
                # robots = ["Panda", "Panda"],
                robots = robot_config, 
                renderer = 'mujoco',
                controller_configs = controller_config
            )
        kwargs.update(update_kwargs)

        if self._is_v1:
            if kwargs["has_offscreen_renderer"]:
                # ensure that we select the correct GPU device for rendering by testing for EGL rendering
                # NOTE: this package should be installed from this link (https://github.com/StanfordVL/egl_probe)
                # import egl_probe
                # valid_gpu_devices = egl_probe.get_available_devices()
                # if len(valid_gpu_devices) > 0:
                #     kwargs["render_gpu_device_id"] = valid_gpu_devices[0]
                import os
                cuda_id = os.environ.get('CUDA_VISIBLE_DEVICES')
                if cuda_id is None or len(cuda_id) == 0:
                    pass
                else:
                    kwargs["render_gpu_device_id"] = int(cuda_id)
        else:
            # make sure gripper visualization is turned off (we almost always want this for learning)
            kwargs["gripper_visualization"] = False
            del kwargs["camera_depths"]
            kwargs["camera_depth"] = False # rename kwarg

        self._env_name = env_name
        self._init_kwargs = deepcopy(kwargs)
        self.env = robosuite.make(self._env_name, **kwargs)
        self.env.hard_reset = False # Robosuite's hard reset causes excessive memory consumption: https://github.com/ARISE-Initiative/robosuite/blob/v1.4.0/robosuite/environments/base.py#L247

        if self._is_v1:
            # Make sure joint position observations and eef vel observations are active
            for ob_name in self.env.observation_names:
                if ("joint_pos" in ob_name) or ("eef_vel" in ob_name):
                    self.env.modify_observable(observable_name=ob_name, attribute="active", modifier=True)

        # statistics used for action normalization, when abs_action = true
        # TODO this is task-dependent, improve this, you can save the data in locally files
        if env_name != 'TwoArmLift':
            self.action_max_list = [list(config.action_max)] 
            self.action_min_list = [list(config.action_min)] 

            self.eef_pos_max_list = [list(config.robot0_eef_pos_max)]  # ours
            self.eef_pos_min_list = [list(config.robot0_eef_pos_min)]
        elif env_name == 'TwoArmLift':
            self.action_max_list = [list(config.action_max_r0), list(config.action_max_r1)]  
            self.action_min_list = [list(config.action_min_r0), list(config.action_min_r1)]

            self.eef_pos_max_list = [list(config.robot0_eef_pos_max), list(config.robot1_eef_pos_max)]  # ours
            self.eef_pos_min_list = [list(config.robot0_eef_pos_min), list(config.robot1_eef_pos_min)]
        else:
            raise NotImplementedError("Env robosuite, action limits, not implemented")

        self.gripper_qpose_max = list(config.gripper_qpose_max)
        self.gripper_qpose_min = list(config.gripper_qpose_min)

        


    def normalize_gripper_qpose(self, qpose_):
        qpose = qpose_.copy()
        lo  = np.asarray(self.gripper_qpose_min, dtype=np.float32)
        hi  = np.asarray(self.gripper_qpose_max, dtype=np.float32)
        # centers and half-ranges
        center = (hi + lo) / 2.0
        half_range = (hi - lo) / 2.0
        # avoid division by zero
        half_range[half_range == 0] = 1.0
        qpose_normalized = ((qpose- center) / half_range)
        return qpose_normalized
    
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
        action[:3] = ((action[:3] - center) / half_range)
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
        action[0:3] = ((action[0:3] - center) / half_range)
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
        arr[0:3] = (arr[0:3]  * half_range + center)
        return arr

    def step(self, action):
        """
        Step in the environment with an action.

        Args:
            action (np.array): action to take

        Returns:
            observation (dict): new observation dictionary
            reward (float): reward for this step
            done (bool): whether the task is done
            info (dict): extra information
        """
        # very important! action cannot be modified as python carry the values!
        # action[0:-1] = action[0:-1] * 0.2 # renormalize the action
        action_ = action.copy()

        # cid = self.env.sim.model.camera_name2id('agentview')
        # # camera extrinsics (world → camera)
        # cam_pos = self.env.sim.model.cam_pos[cid]
        # cam_quat = self.env.sim.model.cam_quat[cid]
        # # cam_mat = mujoco_quat2mat(cam_quat)
        # # camera intrinsics – focal length from FOV
        # fovy = self.env.sim.model.cam_fovy[cid]
        # # f = img_h / (2 * np.tan(np.deg2rad(fovy)/2))
        

        # mj_state = self.env.sim.get_state() # mj_state.flatten()
        # import pdb; pdb.set_trace

   
        # if self.abs_action:
        #     action_ = self.unnormalize_abs_action(action_)

        #     #TODO add limits for the action, as the robosuite osc controller doesn't have any torque constraints
        #     if self.obs_dict_past is not None:
        #         past_eef_pos = self.unnormalize_abs_action(self.obs_dict_past['robot0_eef_pos'])
        #         action_eef_pos = action_[:3]
        #         norm_max = 0.05  # few of the robomimic demo execceds this limits, but the trained policy can still succeed
        #         direction = action_eef_pos - past_eef_pos
        #         dist = np.linalg.norm(direction)
        #         if dist > norm_max:
        #             direction = direction / dist * norm_max
        #             print("execced norm_max")
        #         saturated_eef_pos = past_eef_pos + direction
        #         action_[:3] = saturated_eef_pos


        #     rot = rot6d_to_axisangle(action_[3:-1])

        #     raw_actions = np.concatenate([
        #         action_[:3], rot, action_[-1:]
        #     ], axis=-1).astype(np.float32)
        #     action_ = raw_actions

        if self.abs_action:
            total_dim = action_.shape[-1]
            single_raw_dim = int(total_dim / self.num_robot)

            # import pdb; pdb.set_trace()
            processed = []
            for i in range(self.num_robot):
                # slice out robot i’s chunk
                start = i * single_raw_dim
                end   = start + single_raw_dim
                a_raw = action_[start:end]

                # unnormalize
                a = self.unnormalize_abs_action(a_raw, action_max=self.action_max_list[i], action_min=self.action_min_list[i])

                # limit EEF displacement if we have a past observation
                if self.obs_dict_past is not None and self.limit_eef_vel is True:  # observation include images
                    if self.use_image_obs:
                        key = f"robot{i}_eef_pos"
                        past_pos = self.unnormalize_abs_action(self.obs_dict_past[key], action_max=self.eef_pos_max_list[i], action_min=self.eef_pos_min_list[i])
                    else:
                        past_pos =  self.obs_dict_past['robot0_eef_pos'] 
                    desired = a[:3]
                    delta   = desired - past_pos
                    dist    = np.linalg.norm(delta)
                    max_step = 0.05
                    if dist > max_step:
                        delta = delta / dist * max_step
                        logger.debug(f"robot {i} exceeded max_step")
                    a[:3] = past_pos + delta
                

                # convert 6D → axis‐angle (3)
                rot = rot6d_to_axisangle(a[3:-1])

                # rebuild final per‐arm action (3 + 3 + 1 = 7 dims)
                out = np.concatenate([a[:3], rot, a[-1:]], axis=-1).astype(np.float32)
                processed.append(out)

            # if one robot, this is just its 7-D action;
            # if two, it's [r0(7), r1(7)] → 14 dims
            action_ = np.concatenate(processed, axis=-1)
        else:
            total_dim = action_.shape[-1]
            single_raw_dim = int(total_dim / self.num_robot)

            if self.scale_vel_control:
                processed = []
                for i in range(self.num_robot):
                    # slice out robot i’s chunk
                    start = i * single_raw_dim
                    end   = start + single_raw_dim
                    out = action_[start:end]
                    out[0:-1] = out[0:-1] * 0.2
                    processed.append(out)
                action_ = np.concatenate(processed, axis=-1)
            # print("action_ processed: ", action_)

        # if not self.abs_action:
        #     if self._env_name == 'TwoArmLift':
        #         action_[0:6] = action_[0:6] * 0.2 # renormalize the action
        #         action_[7:-1] = action_[7:-1] * 0.2 # renormalize the action
        #     else:
        #         action_[0:-1] = action_[0:-1] * 0.2 # renormalize the action
        obs, r, done, info = self.env.step(action_)
        obs = self.get_observation_original(obs)
        # return obs, r, self.is_done(), info
        info['success'] = self.is_success()['task']
        done = info['success']

        self.obs_extracted = self.get_observation()  # used for teacher policy

        # # # check obs_ori and eef_xmat whether has a constant relationship
        # # current_ori_mat = self.obs_extracted[6:15].reshape(3, 3)
        # current_ori_mat = R.from_quat(obs['robot0_eef_quat']).as_matrix()
        # eef_xmat = self.env.sim.data.site_xmat[self.env.sim.model.site_name2id("gripper0_grip_site")].reshape(3, 3)
        # Rot_A_to_B = eef_xmat.dot(np.linalg.inv(current_ori_mat))
        # print("Rot_A_to_B: ", Rot_A_to_B)
        # # print("current pos: ", self.obs_extracted[0:3])

        # import pdb
        # pdb.set_trace()

        if self.use_image_obs:
            if self.visualize_image_obs:
                visualize_two_cameras(obs, self.rgb_keys)  # used for visualization
            obs_dict = dict()
            for key in self.rgb_keys:
                obs_dict[key] = np.moveaxis(obs[key],-1,0).astype(np.float32) / 255.   # (C, H, W)
                obs_dict[key] = (2.0 * obs_dict[key] - 1.0).astype(np.float32)
            for key in self.lowdim_keys:
                obs_dict[key] = obs[key].astype(np.float32)  # (dim_L_obs)
                # if key == 'robot0_eef_quat':
                #     obs_dict[key] = T.quat2mat(obs_dict[key]).reshape(-1).astype(np.float32)
                #     # TODO check whether the orientation matters
                #     # obs_dict[key] = self.env._eef_xmat.reshape(-1).astype(np.float32)
                #     # obtain rotation matrix
                if key.endswith('_pos') and self.abs_action:
                    # print('pos : ', obs_dict[key] )
                    if key == 'robot0_eef_pos':
                        obs_dict[key] = self.normalize_eef_pos(obs_dict[key], max_list=self.eef_pos_max_list[0], min_list=self.eef_pos_min_list[0])
                    if key == 'robot1_eef_pos':
                        obs_dict[key] = self.normalize_eef_pos(obs_dict[key], max_list=self.eef_pos_max_list[1], min_list=self.eef_pos_min_list[1])
                    # print('normalized obs_dict[key] : ', obs_dict[key] )
                if key.endswith('qpos'):
                    obs_dict[key] = self.normalize_gripper_qpose(obs_dict[key])


            # combine past the current obs_dict
            if self.obs_dict_past is None:
                self.obs_dict_past = obs_dict
            
            obs_dict_combined = combine_obs_dicts(self.obs_dict_past, obs_dict) 
            # obs_dict_combined[self.rgb_keys] -> (2, C, H, W)
            # obs_dict_combined[self.lowdim_keys] -> (2, lowdim_keys)

            self.obs_dict_past = obs_dict
            
            return obs_dict_combined, r, done, None, info
        else:
            self.obs_dict_past = self.env._get_observations()
        
        obs = self.obs_extracted
        
        expanded_obs = np.expand_dims(obs, axis=0)

        return expanded_obs, r, done, None, info

    def reset(self):
        """
        Reset environment.

        Returns:
            observation (dict): initial observation dictionary.
        """
        raw_obs = self.env.reset()
        self.obs_dict_past = None
        obs = self.get_observation_original(raw_obs)
        self.obs_extracted = self.get_observation()  # used for teacher policy

        if self.use_image_obs:
            obs_dict = dict()
            for key in self.rgb_keys:
                obs_dict[key] = np.moveaxis(obs[key],-1,0).astype(np.float32) / 255.   # (C, H, W)
                obs_dict[key] = (2.0 * obs_dict[key] - 1.0).astype(np.float32)
            for key in self.lowdim_keys:
                obs_dict[key] = obs[key].astype(np.float32)  # (dim_L_obs)
                # if key == 'robot0_eef_quat':
                #     obs_dict[key] = T.quat2mat(obs_dict[key]).reshape(-1).astype(np.float32)
                #     # TODO check whether the orientation matters
                #     # obs_dict[key] = self.env._eef_xmat.reshape(-1).astype(np.float32)
                if key.endswith('_pos') and self.abs_action:
                    # print("pos: ", obs_dict[key])
                    if key == 'robot0_eef_pos':
                        obs_dict[key] = self.normalize_eef_pos(obs_dict[key], max_list=self.eef_pos_max_list[0], min_list=self.eef_pos_min_list[0])
                    if key == 'robot1_eef_pos':
                        obs_dict[key] = self.normalize_eef_pos(obs_dict[key], max_list=self.eef_pos_max_list[1], min_list=self.eef_pos_min_list[1])
                if key.endswith('qpos'):
                    obs_dict[key] = self.normalize_gripper_qpose(obs_dict[key])

            # combine past the current obs_dict
            if self.obs_dict_past is None:
                self.obs_dict_past = obs_dict
            
            obs_dict_combined = combine_obs_dicts(self.obs_dict_past, obs_dict) 
            # obs_dict_combined[self.rgb_keys] -> (2, C, H, W)
            # obs_dict_combined[self.lowdim_keys] -> (2, lowdim_keys)

            self.obs_dict_past = obs_dict
            
            return obs_dict_combined, []
        obs = self.get_observation()
        expanded_obs = np.expand_dims(obs, axis=0)
        return expanded_obs, []

    def reset_to(self, state):
        """
        Reset to a specific simulator state.

        Args:
            state (dict): current simulator state that contains one or more of:
                - states (np.ndarray): initial state of the mujoco environment
                - model (str): mujoco scene xml
        
        Returns:
            observation (dict): observation dictionary after setting the simulator state (only
                if "states" is in @state)
        """
        logger.debug("start reset to")
        should_ret = False
        if "model" in state:
            self.reset()
            robosuite_version_id = int(robosuite.__version__.split(".")[1])
            if robosuite_version_id <= 3:
                from robosuite.utils.mjcf_utils import postprocess_model_xml
                xml = postprocess_model_xml(state["model"])
            else:
                # v1.4 and above use the class-based edit_model_xml function
                xml = self.env.edit_model_xml(state["model"])
            self.env.reset_from_xml_string(xml)
            self.env.sim.reset()
            if not self._is_v1:
                # hide teleop visualization after restoring from model
                self.env.sim.model.site_rgba[self.env.eef_site_id] = np.array([0., 0., 0., 0.])
                self.env.sim.model.site_rgba[self.env.eef_cylinder_id] = np.array([0., 0., 0., 0.])
        if "states" in state:
            self.env.sim.set_state_from_flattened(state["states"])
            self.env.sim.forward()
            should_ret = True

        self.obs_extracted = self.get_observation()  # used for teacher policy

        logger.debug("end reset to")
        if "goal" in state:
            self.set_goal(**state["goal"])
        if should_ret:
            # only return obs if we've done a forward call - otherwise the observations will be garbage
            obs = self.get_observation_original()
            if self.use_image_obs:
                obs_dict = dict()
                for key in self.rgb_keys:
                    obs_dict[key] = np.moveaxis(obs[key],-1,0).astype(np.float32) / 255.   # (C, H, W)
                    obs_dict[key] = (2.0 * obs_dict[key] - 1.0).astype(np.float32)
                for key in self.lowdim_keys:
                    obs_dict[key] = obs[key].astype(np.float32)  # (dim_L_obs)
                    # if key == 'robot0_eef_quat':
                    #     obs_dict[key] = T.quat2mat(obs_dict[key]).reshape(-1).astype(np.float32)
                    #     # TODO check whether the orientation matters
                    #     # obs_dict[key] = self.env._eef_xmat.reshape(-1).astype(np.float32)
                    if key.endswith('_pos') and self.abs_action:
                        # print("pos: ", obs_dict[key])
                        if key == 'robot0_eef_pos':
                            obs_dict[key] = self.normalize_eef_pos(obs_dict[key], max_list=self.eef_pos_max_list[0], min_list=self.eef_pos_min_list[0])
                        if key == 'robot1_eef_pos':
                            obs_dict[key] = self.normalize_eef_pos(obs_dict[key], max_list=self.eef_pos_max_list[1], min_list=self.eef_pos_min_list[1])
                    if key.endswith('qpos'):
                        obs_dict[key] = self.normalize_gripper_qpose(obs_dict[key])

                # combine past the current obs_dict
                if self.obs_dict_past is None:
                    self.obs_dict_past = obs_dict
                
                obs_dict_combined = combine_obs_dicts(self.obs_dict_past, obs_dict) 

                self.obs_dict_past = obs_dict
                
                return obs_dict_combined, []
            return self.get_observation_original()
        return None

    def render(self, mode="human", height=None, width=None, camera_name="agentview"):
        """
        Render from simulation to either an on-screen window or off-screen to RGB array.

        Args:
            mode (str): pass "human" for on-screen rendering or "rgb_array" for off-screen rendering
            height (int): height of image to render - only used if mode is "rgb_array"
            width (int): width of image to render - only used if mode is "rgb_array"
            camera_name (str): camera name to use for rendering
        """
        if mode == "human":
            cam_id = self.env.sim.model.camera_name2id(camera_name)
            # self.env.viewer.set_camera(cam_id)
            return self.env.render()
        elif mode == "rgb_array":
            return self.env.sim.render(height=height, width=width, camera_name=camera_name)[::-1]
        else:
            raise NotImplementedError("mode={} is not implemented".format(mode))

    # def get_observation(self, di=None):
    #     """
    #     Get current environment observation dictionary.

    #     Args:
    #         di (dict): current raw observation dictionary from robosuite to wrap and provide 
    #             as a dictionary. If not provided, will be queried from robosuite.
    #     """
    #     # if di is None:
    #     #     di = self.env._get_observations(force_update=True) if self._is_v1 else self.env._get_observation()
    #     # ret = {}
    #     # for k in di:
    #     #     if (k in ObsUtils.OBS_KEYS_TO_MODALITIES) and ObsUtils.key_is_obs_modality(key=k, obs_modality="rgb"):
    #     #         ret[k] = di[k][::-1]
    #     #         if self.postprocess_visual_obs:
    #     #             ret[k] = ObsUtils.process_obs(obs=ret[k], obs_key=k)

    #     # # "object" key contains object information
    #     # ret["object"] = np.array(di["object-state"])
    #     # if "addl-object-state" in di:
    #     #     ret["addl-object"] = np.array(di["addl-object-state"])

    #     # if self._is_v1:
    #     #     for robot in self.env.robots:
    #     #         # add all robot-arm-specific observations. Note the (k not in ret) check
    #     #         # ensures that we don't accidentally add robot wrist images a second time
    #     #         pf = robot.robot_model.naming_prefix
    #     #         for k in di:
    #     #             if k.startswith(pf) and (k not in ret) and (not k.endswith("proprio-state")):
    #     #                 ret[k] = np.array(di[k])
    #     # else:
    #     #     # minimal proprioception for older versions of robosuite
    #     #     ret["proprio"] = np.array(di["robot-state"])
    #     #     ret["eef_pos"] = np.array(di["eef_pos"])
    #     #     ret["eef_quat"] = np.array(di["eef_quat"])
    #     #     ret["gripper_qpos"] = np.array(di["gripper_qpos"])
    #     # return ret
    #     return self.env._get_observations()
    

    #only for nut-assembly task
    def get_observation_nutAssembly(self, di=None):
        """
        Get current environment observation dictionary.

        Args:
            di (dict): current raw observation dictionary from robosuite to wrap and provide 
                as a dictionary. If not provided, will be queried from robosuite.
        """

        # only for nut-assembly task
        obs_raw = self.env._get_observations()
        current_ee_pos = obs_raw['robot0_eef_pos'] # 3
        nut_pos = obs_raw['SquareNut_pos']   # 3
        current_ori = T.quat2mat(obs_raw['robot0_eef_quat']).reshape(-1) # 9
        nut_ori_quat = obs_raw['SquareNut_quat']
        nut_ori = T.quat2mat(obs_raw['SquareNut_quat']).reshape(-1)   # 9
        # relative_rotation = quaternion_multiply(target_ori, quaternion_conjugate(current_ori))
        peg_up_pos = np.array([0.227, 0.101, 1.0])  # 3

        peg_up_pos_lower = np.array([0.227, 0.101, 0.9])  # 3

        gripper_qpos = obs_raw['robot0_gripper_qpos'] # 2
        gripper_qv = obs_raw['robot0_gripper_qvel'] # 2

        target_orientatoin_1 = np.array([-0.000, 0.000, 0.706, 0.709])  # rot_Z_90
        target_orientatoin_2 = np.array([0, 0, 1, 0])  # rot_Z_180
        target_orientatoin_3 = np.array([ 0, 0, -0.7071068, 0.7071068 ])  # rot_Z_negative90

        mat_handle_to_nut = T.quat2mat(np.array([ 1, 0, 0, 0 ]))
        mat_handle_w = nut_ori.reshape(3, 3) .dot(mat_handle_to_nut)
        quat_handle_w = T.mat2quat(mat_handle_w)
        pos_handle_to_nutcenter = np.array([0.065, 0, 0])
        pos_handle_to_world = nut_pos + mat_handle_w.dot(pos_handle_to_nutcenter)

        quat_rotz180 = np.array([0, 0, 1, 0])
        mat_rotz180 = T.quat2mat(quat_rotz180)
        mat_handle_w2 = mat_rotz180.dot(mat_handle_w)
        quat_handle_w2 = T.mat2quat(mat_handle_w2)

        relative_rotation1 = quaternion_multiply(quat_handle_w, quaternion_conjugate(obs_raw['robot0_eef_quat']))
        relative_rotation2 = quaternion_multiply(quat_handle_w2, quaternion_conjugate(obs_raw['robot0_eef_quat']))

        active_nuts = []
        for i, nut in enumerate(self.env.nuts):
            if self.env.objects_on_pegs[i]:
                continue
            active_nuts.append(nut)

        grasped = self.env._check_grasp(
                    gripper=self.env.robots[0].gripper,
                    object_geoms=[g for active_nut in active_nuts for g in active_nut.contact_geoms],
                )
        # print("grasped: ", grasped)
        
        obs_processed = np.hstack((
            current_ee_pos,
            nut_pos,
            current_ori,
            nut_ori,
            [gripper_qpos[0]],
            [gripper_qv[0]],
            nut_ori_quat,
            relative_rotation1,
            relative_rotation2,
            # target_orientatoin_1,
            # target_orientatoin_2,
            # target_orientatoin_3,
            current_ee_pos - pos_handle_to_world,
            peg_up_pos - nut_pos,
            peg_up_pos_lower - nut_pos,
            [grasped]
        ))
        # print("obs_processed: ", obs_processed)
        
        return obs_processed
    
    #only for nut-assembly task
    def get_observation_pickCan(self, di=None):
        """
        Get current environment observation dictionary.

        Args:
            di (dict): current raw observation dictionary from robosuite to wrap and provide 
                as a dictionary. If not provided, will be queried from robosuite.
        """

        # only for nut-assembly task
        obs_raw = self.env._get_observations()
        current_ee_pos = obs_raw['robot0_eef_pos'] # 3
        can_pos = obs_raw['Can_pos']   # 3
        current_ori = T.quat2mat(obs_raw['robot0_eef_quat']).reshape(-1) # 9
        nut_ori_quat = obs_raw['Can_quat']
        nut_ori = T.quat2mat(obs_raw['Can_quat']).reshape(-1)   # 9
        # relative_rotation = quaternion_multiply(target_ori, quaternion_conjugate(current_ori))

        gripper_qpos = obs_raw['robot0_gripper_qpos'] # 2
        gripper_qv = obs_raw['robot0_gripper_qvel'] # 2

        can_pos_target = np.array([0.145, 0.38, 0.95])

        can_obj = None
        for i, obj in enumerate(self.env.objects):
            if self.env.objects_in_bins[i]:
                continue
            can_obj = obj
        grasped = self.env._check_grasp(gripper=self.env.robots[0].gripper, object_geoms=can_obj.contact_geoms)
        # print("grasped: ", grasped)
        
        obs_processed = np.hstack((
            current_ee_pos,
            can_pos,
            current_ori,
            nut_ori,
            [gripper_qpos[0]],
            [gripper_qv[0]],
            nut_ori_quat,
            current_ee_pos - can_pos,
            can_pos_target - can_pos,
            can_pos_target,
            [grasped]
        ))  # 40
        # print("obs_processed: ", obs_processed.shape)
        
        return obs_processed
    
    def get_observation_TwoArmList(self, di=None):
        """
        Get current environment observation dictionary.

        Args:
            di (dict): current raw observation dictionary from robosuite to wrap and provide 
                as a dictionary. If not provided, will be queried from robosuite.
        """

        # only for nut-assembly task
        obs_raw = self.env._get_observations()
        current_pos_r0 =obs_raw['robot0_eef_pos']
        current_pos_r1 =obs_raw['robot1_eef_pos']

        current_ori_r0 = T.quat2mat(obs_raw['robot0_eef_quat']).reshape(-1) # 9
        current_ori_r1 = T.quat2mat(obs_raw['robot1_eef_quat']).reshape(-1) # 9
        
        # relative_rotation = quaternion_multiply(target_ori, quaternion_conjugate(current_ori))
        

        handle0_pose = obs_raw['handle0_xpos']
        handle1_pose = obs_raw['handle1_xpos']
        vec_object = handle0_pose - handle1_pose
        vec_object = vec_object / np.linalg.norm(vec_object)
        x = vec_object
        z = np.array([0, 0, -1])
        # Calculate x as the cross product of y and z
        y = np.cross(z, x)


        gripper_qpos_r0 = obs_raw['robot0_gripper_qpos'] # 2
        gripper_qv_r0 = obs_raw['robot0_gripper_qvel'] # 2
        gripper_qpos_r1 = obs_raw['robot1_gripper_qpos'] # 2
        gripper_qv_r1 = obs_raw['robot1_gripper_qvel'] # 2

        (g0, g1) = (
            (self.env.robots[0].gripper["right"], self.env.robots[0].gripper["left"])
            if self.env.env_configuration == "bimanual"
            else (self.env.robots[0].gripper, self.env.robots[1].gripper)
        )

        # Grasping reward
        grasped_r0= self.env._check_grasp(gripper=g0, object_geoms=self.env.pot.handle0_geoms)
           
        grasped_r1= self.env._check_grasp(gripper=g1, object_geoms=self.env.pot.handle1_geoms)

        
        obs_processed = np.hstack((
            current_pos_r0,
            current_ori_r0,
            current_pos_r1,
            current_ori_r1,
            handle0_pose,
            handle1_pose,
            vec_object,
            y,
            [gripper_qpos_r0[0]],
            [gripper_qv_r0[0]],
            [gripper_qpos_r1[0]],
            [gripper_qv_r1[0]],
            current_pos_r0 - handle0_pose,
            current_pos_r1 - handle1_pose,
            [grasped_r0],
            [grasped_r1]
        ))  # 40
        # print("obs_processed: ", obs_processed.shape)
        
        return obs_processed
        
    def get_observation_toolhang(self, di=None):
        """
        Get current environment observation dictionary.

        Args:
            di (dict): current raw observation dictionary from robosuite to wrap and provide 
                as a dictionary. If not provided, will be queried from robosuite.
        """
        if di is None:
            di = self.env._get_observations(force_update=True) if self._is_v1 else self.env._get_observation()
        # print("raw observation di agentview_image: ", di['agentview_image'].shape)
        obs_raw = di
        current_pos = obs_raw['robot0_eef_pos']
        current_ori = T.quat2mat(obs_raw['robot0_eef_quat']).reshape(-1)
        gripper_qpos = obs_raw['robot0_gripper_qpos'] # 2
        gripper_qv = obs_raw['robot0_gripper_qvel'] # 2
        grasped_tool = self.env._check_grasp(gripper=self.env.robots[0].gripper, object_geoms=self.env.tool.contact_geoms)
        grasped_frame = self.env._check_grasp(gripper=self.env.robots[0].gripper, object_geoms=self.env.frame.contact_geoms)
        tool_hole1_center =self.env.sim.data.site_xpos[self.env.obj_site_id["tool_hole1_center"]]
        frame_hook_endpoint_pos =self.env.sim.data.site_xpos[self.env.obj_site_id["frame_hang_site"]]
        hook_endpoint =self.env.sim.data.site_xpos[self.env.obj_site_id["frame_mount_site"]]
        frame_hook_vec =frame_hook_endpoint_pos - hook_endpoint
        frame_intersection01_pos = self.env.sim.data.site_xpos[self.env.obj_site_id["frame_intersection_site"]]
        frame_intersection_pos =self.env.sim.data.site_xpos[self.env.obj_site_id["frame_intersection_site"]] + frame_hook_vec * 0.025
        frame_tip_pos =self.env.sim.data.site_xpos[self.env.obj_site_id["frame_tip_site"]] 
        
        frame_hang_vec =frame_hook_endpoint_pos - self.env.sim.data.site_xpos[self.env.obj_site_id["frame_intersection_site"]]
        frame_hang_length = np.linalg.norm(frame_hang_vec)
        frame_hang_vec = -frame_hang_vec / frame_hang_length

        tool_handle =self.env.sim.data.body_xpos[self.env.obj_body_id['tool']] 
        
        tool_vec = tool_hole1_center - tool_handle
        tool_vec[2] = 0
        tool_vec_length = np.linalg.norm(tool_vec)
        tool_vec = -tool_vec / tool_vec_length

        base_walls_geom_positions = [
           self.env.sim.data.geom_xpos[self.env.obj_geom_id["stand_wall_{}".format(i)]]  for i in range(4)
        ]
        base_walls_geom_positions_average = np.mean(np.array(base_walls_geom_positions), axis=0)+ np.array([0,0, 0.005])
        delta_frame_tip_base = frame_tip_pos - base_walls_geom_positions_average

        contacts_frame_tool = 1 * check_contact(self.env.sim, self.env.frame.contact_geoms, self.env.stand.contact_geoms)



        frame_hook2_vec = frame_intersection01_pos - hook_endpoint
        frame_hook2_length = np.linalg.norm(frame_hook2_vec)
        frame_hook2_vec = frame_hook2_vec / frame_hook2_length

        # geom wall position vectors relative to base position
        geom_positions = [
            self.env.sim.data.geom_xpos[self.env.obj_geom_id["stand_wall_{}".format(i)]] - hook_endpoint for i in range(4)
        ]

        # take cross product of each point against the line, and then dot the result to see if
        # the sign is positive or negative. If it is positive, then they are on the same side
        # (visualize with right-hand-rule to see this)
        rod_is_between_stand_walls = np.array(
            [
                np.dot(np.cross(geom_positions[0], frame_hook2_vec), np.cross(geom_positions[2], frame_hook2_vec)) < 0,
                np.dot(np.cross(geom_positions[1], frame_hook2_vec), np.cross(geom_positions[3], frame_hook2_vec)) < 0,
            ])
        
        # print("rod_is_between_stand_walls: ", 1 * rod_is_between_stand_walls)

        obs_processed = np.hstack((
            current_pos,
            current_ori,
            [gripper_qpos[0]],
            [gripper_qv[0]],
            [grasped_tool],
            [grasped_frame],
            tool_hole1_center,
            frame_hook_endpoint_pos,
            hook_endpoint,
            frame_hook_vec,
            frame_hang_vec,
            [frame_intersection01_pos[2] - 1.1],
            frame_intersection_pos,
            frame_intersection01_pos,
            frame_tip_pos,
            tool_handle,
            tool_vec,
            base_walls_geom_positions_average,
            base_walls_geom_positions[0] - hook_endpoint, 
            base_walls_geom_positions[1] - hook_endpoint,
            base_walls_geom_positions[2] - hook_endpoint,
            base_walls_geom_positions[3] - hook_endpoint,
            1 * rod_is_between_stand_walls,
            [1 * all(rod_is_between_stand_walls)],
            current_pos - tool_handle,
            current_pos - frame_intersection_pos- np.array([0, 0, 0.05]),
            delta_frame_tip_base,
            [np.linalg.norm(delta_frame_tip_base[0:2])],
            tool_hole1_center - frame_hook_endpoint_pos,
            [contacts_frame_tool]
        ))  # 76
        # print("shape of obs: ", obs_processed.shape)
        return obs_processed

    def get_observation(self):
        if self._env_name == 'NutAssemblySquare':
            obs = self.get_observation_nutAssembly()
        elif self._env_name == 'ToolHang':
            obs = self.get_observation_toolhang()
        elif self._env_name == 'PickPlaceCan':
            obs = self.get_observation_pickCan()
        elif self._env_name == "TwoArmLift":
            obs = self.get_observation_TwoArmList()
        else:
            obs = self.get_observation_original()
        return obs.copy()

    def get_observation_original(self, di=None):
        """
        Get current environment observation dictionary.

        Args:
            di (dict): current raw observation dictionary from robosuite to wrap and provide 
                as a dictionary. If not provided, will be queried from robosuite.
        """
        if di is None:
            di = self.env._get_observations(force_update=True) if self._is_v1 else self.env._get_observation()

        ret = {}
        # import pdb
        # pdb.set_trace()
        for k in di:
            # if (k in ObsUtils.OBS_KEYS_TO_MODALITIES) and ObsUtils.key_is_obs_modality(key=k, obs_modality="rgb"):
            # if k in ['robot0_eye_in_hand_image', 'agentview_image']:
            if k in self.rgb_keys:
                ret[k] = di[k][::-1]
                # if self.postprocess_visual_obs:
                #     ret[k] = ObsUtils.process_obs(obs=ret[k], obs_key=k)

        # "object" key contains object information
        # print("di: ", di)
        ret["object"] = np.array(di["object-state"])
        if "addl-object-state" in di:
            ret["addl-object"] = np.array(di["addl-object-state"])

        if self._is_v1:
            for robot in self.env.robots:
                # add all robot-arm-specific observations. Note the (k not in ret) check
                # ensures that we don't accidentally add robot wrist images a second time
                pf = robot.robot_model.naming_prefix
                for k in di:
                    if k.startswith(pf) and (k not in ret) and (not k.endswith("proprio-state")):
                        ret[k] = np.array(di[k])
        else:
            # minimal proprioception for older versions of robosuite
            ret["proprio"] = np.array(di["robot-state"])
            ret["eef_pos"] = np.array(di["eef_pos"])
            ret["eef_quat"] = np.array(di["eef_quat"])
            ret["gripper_qpos"] = np.array(di["gripper_qpos"])
        return ret
    
    def get_state(self):
        """
        Get current environment simulator state as a dictionary. Should be compatible with @reset_to.
        """
        xml = self.env.sim.model.get_xml() # model xml file
        state = np.array(self.env.sim.get_state().flatten()) # simulator state
        return dict(model=xml, states=state)

    def get_reward(self):
        """
        Get current reward.
        """
        return self.env.reward()

    def get_goal(self):
        """
        Get goal observation. Not all environments support this.
        """
        return self.get_observation_original(self.env._get_goal())

    def set_goal(self, **kwargs):
        """
        Set goal observation with external specification. Not all environments support this.
        """
        return self.env.set_goal(**kwargs)

    def is_done(self):
        """
        Check if the task is done (not necessarily successful).
        """

        # Robosuite envs always rollout to fixed horizon.
        return False

    def is_success(self):
        """
        Check if the task condition(s) is reached. Should return a dictionary
        { str: bool } with at least a "task" key for the overall task success,
        and additional optional keys corresponding to other task criteria.
        """
        succ = self.env._check_success()
        if isinstance(succ, dict):
            assert "task" in succ
            return succ
        return { "task" : succ }

    @property
    def action_dimension(self):
        """
        Returns dimension of actions (int).
        """
        return self.env.action_spec[0].shape[0]

    @property
    def name(self):
        """
        Returns name of environment name (str).
        """
        return self._env_name

    @property
    def type(self):
        """
        Returns environment type (int) for this kind of environment.
        This helps identify this env class.
        """
        return EB.EnvType.ROBOSUITE_TYPE

    def serialize(self):
        """
        Save all information needed to re-instantiate this environment in a dictionary.
        This is the same as @env_meta - environment metadata stored in hdf5 datasets,
        and used in utils/env_utils.py.
        """
        return dict(env_name=self.name, type=self.type, env_kwargs=deepcopy(self._init_kwargs))

    @classmethod
    def create_for_data_processing(
        cls, 
        env_name, 
        camera_names, 
        camera_height, 
        camera_width, 
        reward_shaping, 
        **kwargs,
    ):
        """
        Create environment for processing datasets, which includes extracting
        observations, labeling dense / sparse rewards, and annotating dones in
        transitions. 

        Args:
            env_name (str): name of environment
            camera_names (list of str): list of camera names that correspond to image observations
            camera_height (int): camera height for all cameras
            camera_width (int): camera width for all cameras
            reward_shaping (bool): if True, use shaped environment rewards, else use sparse task completion rewards
        """
        is_v1 = (robosuite.__version__.split(".")[0] == "1")
        has_camera = (len(camera_names) > 0)

        new_kwargs = {
            "reward_shaping": reward_shaping,
        }

        if has_camera:
            if is_v1:
                new_kwargs["camera_names"] = list(camera_names)
                new_kwargs["camera_heights"] = camera_height
                new_kwargs["camera_widths"] = camera_width
            else:
                # assert len(camera_names) == 1
                if has_camera:
                    new_kwargs["camera_name"] = camera_names[0]
                    new_kwargs["camera_height"] = camera_height
                    new_kwargs["camera_width"] = camera_width

        kwargs.update(new_kwargs)

        # also initialize obs utils so it knows which modalities are image modalities
        image_modalities = list(camera_names)
        if is_v1:
            image_modalities = ["{}_image".format(cn) for cn in camera_names]
        elif has_camera:
            # # v0.3 only had support for one image, and it was named "image"
            # assert len(image_modalities) == 1
            # image_modalities = ["image"]

            # more hacking....
            input_image_modalities = image_modalities.copy()
            image_modalities = []
            for im_mod in input_image_modalities:
                if im_mod == "agentview":
                    image_modalities.append("image")
                elif im_mod == "robot0_eye_in_hand":
                    image_modalities.append("image_wrist")
                else:
                    raise ValueError
        obs_modality_specs = {
            "obs": {
                "low_dim": [], # technically unused, so we don't have to specify all of them
                "rgb": image_modalities,
            }
        }
        ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs)

        # note that @postprocess_visual_obs is False since this env's images will be written to a dataset
        return cls(
            env_name=env_name,
            render=False, 
            render_offscreen=has_camera, 
            use_image_obs=has_camera, 
            postprocess_visual_obs=False,
            **kwargs,
        )

    @property
    def rollout_exceptions(self):
        """
        Return tuple of exceptions to except when doing rollouts. This is useful to ensure
        that the entire training run doesn't crash because of a bad policy that causes unstable
        simulation computations.
        """
        # return (mujoco_py.builder.MujocoException)
        return None

    def __repr__(self):
        """
        Pretty-print env description.
        """
        return self.name + "\n" + json.dumps(self._init_kwargs, sort_keys=True, indent=4)
