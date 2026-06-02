# used to convert robomimic image/low_dim dataset into a data buffer
# things needs to be changed: output, which defines the path of the hdf5 file

import argparse
import h5py
import pickle
import numpy as np
import sys, os
import cv2 # for visualize the camera observation

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
)
from tools.buffer import Buffer_uniform_sampling, HDF5Buffer
import robosuite.utils.transform_utils as T
from env.robotsuite.env_robosuite import EnvRobosuite
# Make sure your Buffer_uniform_sampling class is defined or imported
# from tools.buffer import Buffer_uniform_sampling


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a robomimic HDF5 dataset into an HDF5 replay buffer."
    )
    parser.add_argument("input_path", help="Path to the robomimic HDF5 dataset.")
    parser.add_argument(
        "--use-abs-action",
        action="store_true",
        help="Normalize absolute actions before conversion.",
    )
    return parser.parse_args()


args = parse_args()


from hydra import initialize, compose
from omegaconf import OmegaConf

# If your configs live in ./conf and the main file is next to it:
with initialize(version_base=None, config_path="../config/exp_offline/"):
    # name of the YAML to load, e.g., conf/config_env.yaml
    cfg = compose(config_name="train_Diffusion_image_Ta8",
                  overrides=[
                    'task=pickcan_image_abs'
                      # you can pass other overrides here as needed
                      # f"use_abs_action={use_abs_action}",  # if you have a Python var
                  ]
                  )
    config_env = cfg.task
    

# used to combine the current and previous observation
def combine_obs_dicts(obs_dict_past, obs_dict_current):
    obs_dict_combined = {}
    for key in obs_dict_current:
        obs_dict_combined[key] = np.stack([obs_dict_past[key], obs_dict_current[key]], axis=0)
    return obs_dict_combined

def visualize_two_cameras(obs_dict, obs_keys, t):
    """
    Displays two image observations side by side (or in separate windows).
    obs_dict[key] should be a float32 array in C×H×W, values in [0,1].
    """

    key1 = obs_keys[0]
    key2 = obs_keys[1]
    img1 = obs_dict[key1][t]
    img2 = obs_dict[key2][t]

    # Option A: show in separate windows
    cv2.imshow(f"Obs_{key1}", img1)
    cv2.imshow(f"Obs_{key2}", img2)

    # Option B: side-by-side in one window
    # combined = np.hstack((img1, img2))
    # cv2.imshow("Obs comparison", combined)

    # waitKey(1) for non‐blocking display in a loop, or 0 to wait until a keypress
    cv2.waitKey(1)

shape_meta = {  # should be defined outside this class, as this depends on the env
    "obs": {
        "agentview_image": {"shape": [3, 84, 84], "type": "rgb"},
        "robot0_eye_in_hand_image": {"shape": [3, 84, 84], "type": "rgb"},
        "robot0_eef_pos": {"shape": [3]},
        # "robot0_eef_quat": {"shape": [4]},
        "robot0_eef_quat": {"shape": [9]},  # use rotation matrix instead of quat
        "robot0_gripper_qpos": {"shape": [2]}
    },
    "action": {"shape": [7]}
}
rgb_keys = list()
lowdim_keys = list()
obs_shape_meta = shape_meta['obs']
for key, attr in obs_shape_meta.items():
    type = attr.get('type', 'low_dim')
    if type == 'rgb':
        rgb_keys.append(key)
    elif type == 'low_dim':
        lowdim_keys.append(key)
obs_rgb_keys = rgb_keys
obs_lowdim_keys = lowdim_keys


# Configure your buffer
min_size = 1000        # minimum buffer size before sampling is allowed
max_size = 1_000_000   # maximum capacity of the buffer
use_hdf5_buffer = True
use_abs_action = args.use_abs_action

action_dim = 7 if not use_abs_action else 10
# use_abs_action = False

if use_hdf5_buffer:
    # buffer = HDF5Buffer(min_size=min_size, max_size=max_size)
    n_obs_steps = 2
    field_shapes = {
        'agentview_image':           (n_obs_steps, 3, 84, 84),
        'robot0_eye_in_hand_image':  (n_obs_steps,3, 84, 84),
        'robot0_eef_pos':            (n_obs_steps,3),
        'robot0_eef_quat':           (n_obs_steps,4),
        'robot0_gripper_qpos':       (n_obs_steps,2),
        # 'teacher_action':                    (10,),
        # 'robot_action':                    (10,),
        'teacher_action':                    (16, action_dim),
        'robot_action':                    (16, action_dim),
    }
    ## To check: should remove existing buffer.h5
    buffer = HDF5Buffer(filename ='outputs/buffer.h5', field_shapes=field_shapes, min_size=32,
                                            max_size=max_size, dtype_map={})
else:
    buffer = Buffer_uniform_sampling(min_size=min_size, max_size=max_size)



# load config_env using hydra
# env = EnvRobosuite(env_name='NutAssemblySquare', use_image_obs=True, 
#                     use_abs_action= use_abs_action, config= config_env)

env = EnvRobosuite(env_name='PickPlaceCan', use_image_obs=True, 
                    use_abs_action= use_abs_action, config= config_env)

output = os.path.expanduser(os.path.expandvars(args.input_path))

action_horizon = 16
with h5py.File(output, 'r+') as out_file:
    data_group = out_file['data']
    
    # Iterate through each demonstration
    for i, demo_key in enumerate(data_group.keys()):
        ## step saving if we have enough data
        print("i episode: ", i)
        if i > 300: 
            break
        demo = data_group[demo_key]
        actions = demo['actions'][:]  # shape (T, action_dim)
        obs = demo['obs']   # <KeysViewHDF5 ['agentview_image', 'object', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_eef_vel_ang', 'robot0_eef_vel_lin', 'robot0_eye_in_hand_image', 'robot0_gripper_qpos', 'robot0_gripper_qvel', 'robot0_joint_pos', 'robot0_joint_pos_cos', 'robot0_joint_pos_sin', 'robot0_joint_vel']>
        
        if use_abs_action:
            from env.robotsuite.env_robosuite import axisangle_to_rot6d
            all_a = []
            for t in range(actions.shape[0]):
                rot6d = axisangle_to_rot6d(actions[t, 3:6])
                action_ = np.concatenate([actions[t,:3], rot6d, actions[t,6:]], 0)
                action_ = env.normalize_abs_action(action_,  action_max=env.action_max_list[0], action_min=env.action_min_list[0])
                all_a.append(action_)
            actions = np.stack(all_a, 0)
        # import pdb
        # pdb.set_trace()
        
        # Add each (state, action) pair into the buffer
        obs_dict_past = None
        for t in range(0, actions.shape[0]):
            visualize_two_cameras(obs, obs_rgb_keys, t)
            obs_dict = dict()
            for key in obs_rgb_keys:
                obs_dict[key] = np.moveaxis(obs[key][t],-1,0).astype(np.float32) / 255.   # (C, H, W)
                obs_dict[key] = (2.0 * obs_dict[key] - 1.0).astype(np.float32)
            for key in obs_lowdim_keys:
                obs_dict[key] = obs[key][t].astype(np.float32)  # (dim_L_obs)
                # if key == 'robot0_eef_quat':
                #     obs_dict[key] = T.quat2mat(obs_dict[key]).reshape(-1).astype(np.float32)

                if key.endswith('_pos') and use_abs_action:
                    # print('obs_dict[key] : ', obs_dict[key] )
                    obs_dict[key] = env.normalize_abs_action(obs_dict[key],  action_max=env.action_max_list[0], action_min=env.action_min_list[0])
                    # print('normalized obs_dict[key] : ', obs_dict[key] )
                    arr = np.asarray(obs_dict[key])
                    if not (np.all(arr >= -1.0) and np.all(arr <= 1.0)):
                        import pdb 
                        pdb.set_trace()
                if key.endswith('qpos'):
                    # print('qpos obs_dict[key] : ', obs_dict[key] )
                    obs_dict[key] = env.normalize_gripper_qpose(obs_dict[key])
                    # print('normalized qpos obs_dict[key] : ', obs_dict[key] )

            if obs_dict_past is None:
                obs_dict_past = obs_dict
                
            obs_dict_combined = combine_obs_dicts(obs_dict_past, obs_dict) 
            
            if t == 0:
                action_t0 = np.zeros_like(actions[t:t+action_horizon])
                action_t0[1:] = actions[t:t-1+action_horizon]
                action_t0[0] = action_t0[1]
                buffer.add([obs_dict_combined, action_t0,action_t0])
            elif t-1+action_horizon < actions.shape[0]:
                buffer.add([obs_dict_combined, actions[t-1:t-1+action_horizon], actions[t-1:t-1+action_horizon]])
            else:
                # padding data as the same as diffusion policy
                action_tlast =  np.zeros_like(action_t0)
                action_tlast[: actions.shape[0] - (t) ] = actions[t-1: -1]
                action_tlast[actions.shape[0] - (t)  :] = actions[-1]
                buffer.add([obs_dict_combined, action_tlast, action_tlast])
                # print(action_tlast)
            # import pdb 
            # pdb.set_trace()
            
            obs_dict_past = obs_dict

            
if use_hdf5_buffer:
    buffer.save_to_file('outputs/state_action_buffer.h5')
else:
    buffer.save_to_file('state_action_buffer.pkl')
print("buffer length: ", buffer.length())
# (Optional) Save the populated buffer for later use
# with open('state_action_buffer.pkl', 'wb') as f:
#     pickle.dump(buffer, f)

