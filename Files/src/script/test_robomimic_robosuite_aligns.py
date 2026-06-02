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
from tools.buffer import Buffer_uniform_sampling
import robosuite.utils.transform_utils as T
from env.robotsuite.env_robosuite import EnvRobosuite


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare robomimic observations/actions against robosuite playback."
    )
    parser.add_argument("input_path", help="Path to the robomimic HDF5 dataset.")
    parser.add_argument(
        "--use-abs-action",
        action="store_true",
        help="Use the absolute-action dataset/action transform.",
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
                    'task=square_image_vel'
                      # you can pass other overrides here as needed
                      # f"use_abs_action={use_abs_action}",  # if you have a Python var
                  ]
                  )
    config_env = cfg.task
    


# Make sure your Buffer_uniform_sampling class is defined or imported
# from tools.buffer import Buffer_uniform_sampling

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
    cv2.imshow(f"Obs_1{key1}", img1)
    cv2.imshow(f"Obs_1{key2}", img2)

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
buffer = Buffer_uniform_sampling(min_size=min_size, max_size=max_size)

# Path to your robomimic HDF5 file
# output = 'path/to/robomimic_data.hdf5'
# output = '${HOME}/Github_Opensource_Projects/diffusion_policy/data/robomimic/datasets/square/ph/image.hdf5'

use_abs_action = args.use_abs_action
output = os.path.expanduser(os.path.expandvars(args.input_path))


env = EnvRobosuite(env_name='NutAssemblySquare', use_image_obs=True, 
                    use_abs_action= use_abs_action, config=config_env)

# list of all demonstration episodes (sorted in increasing number order)
f = h5py.File(output, "r")
demos = list(f["data"].keys())
inds = np.argsort([int(elem[5:]) for elem in demos])
demos = [demos[i] for i in inds]

# # maybe reduce the number of demonstrations to playback
# if args.n is not None:
#     demos = demos[:args.n]

if not use_abs_action:
    # action_max =  np.ones(7)
    # action_min =  -1 * np.ones(7)
    action_max =  -1 * np.ones(7)
    action_min =  1 * np.ones(7)
else:
    action_max = [0.29555445, 0.31293629, 1.10113877]  
    action_min =[-0.33556048831637464, -0.04873956,  0.77917393]

total_samples = 0
for ind in range(len(demos)):
    ep = demos[ind]

    # prepare initial state to reload from
    states = f["data/{}/states".format(ep)][()]
    initial_state = dict(states=states[0])

    initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]
    # print("initial_state: ", initial_state)
    # extract obs, rewards, dones
    actions = f["data/{}/actions".format(ep)][()]
    obs_dataset = f["data/{}/obs".format(ep)]
    obs_, info = env.reset()
    obs = env.reset_to(initial_state)

    traj = dict(
        obs=[], 
        next_obs=[], 
        rewards=[], 
        dones=[], 
        actions=np.array(actions), 
        states=np.array(states), 
        initial_state_dict=initial_state,
    )
    traj_len = states.shape[0]

    # iteration variable @t is over "next obs" indices
    accumulated_mse = {key: [] for key in obs_.keys()}
    for t in range(0, traj_len ):

        # # # get next observation
        # if t == traj_len:
        #     # play final action to get next observation for last timestep
        #     next_obs = env.reset_to({"states" : states[t]})
        #     next_obs, _, _, _, _ = env.step(actions[t - 1])
        # else:
        #     # reset to simulator state to get observation
        #     next_obs = env.reset_to({"states" : states[t]})

        if use_abs_action:
            from env.robotsuite.env_robosuite import axisangle_to_rot6d
            rot6d = axisangle_to_rot6d(actions[t, 3:6])
            print("rot6d: ", rot6d)
            action_ = np.concatenate([actions[t,:3], rot6d, actions[t,6:]], 0)
            action_ = env.normalize_abs_action(action_)
        else:
            action_ = actions[t]
        print("action_: ", action_)
        next_obs, _, _, _, _ = env.step(action_)
        env.render()


        # for i, v in enumerate(actions[t][:3]):
        for i, v in enumerate(actions[t]):
            if v > action_max[i]:
                action_max[i] = v
                # import pdb; pdb.set_trace()
            if v < action_min[i]:
                action_min[i] = v
                # import pdb; pdb.set_trace()
        print("self.action_max: ", action_max, " self.action_min: ", action_min)

        # import pdb
        # pdb.set_trace()

        # eef_quat_recorded = T.quat2mat(obs_dataset['robot0_eef_quat'][t+1]).reshape(-1).astype(np.float32)
        # eef_quat_observed = next_obs['robot0_eef_quat']
        # # next_obs.keys(): dict_keys(['agentview_image', 'robot0_eye_in_hand_image', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos'])

        mse = {}

        obs_dict = dict()
        obs_dict_past = None
        visualize_two_cameras(obs_dataset, obs_rgb_keys, t)
        for key in obs_rgb_keys:
            obs_dict[key] = np.moveaxis(obs_dataset[key][t],-1,0).astype(np.float32) / 255.   # (C, H, W)
            obs_dict[key] = (2.0 * obs_dict[key] - 1.0).astype(np.float32)
        for key in obs_lowdim_keys:
            obs_dict[key] = obs_dataset[key][t].astype(np.float32)  # (dim_L_obs)
            # if key == 'robot0_eef_quat':
            #     obs_dict[key] = T.quat2mat(obs_dict[key]).reshape(-1).astype(np.float32)
            if key.endswith('_pos') and use_abs_action:
                obs_dict[key] = env.normalize_abs_action(obs_dict[key])
            if key.endswith('qpos'):
                obs_dict[key] = env.normalize_gripper_qpose(obs_dict[key])
                # TODO qpos need normalization
               
        if obs_dict_past is None:
            obs_dict_past = obs_dict
            
        obs_dict_combined = combine_obs_dicts(obs_dict_past, obs_dict) 

        for key, obs_val in next_obs.items():
            # grab the “recorded” value at t+1
            rec_val = obs_dict_combined[key]
            accumulated_mse[key].append( np.mean((obs_val - rec_val)**2))
            mse[key] = np.mean((obs_val - rec_val)**2)
        print("mse: ", mse)



average_mse = {key: np.mean(values) for key, values in accumulated_mse.items()}
print("\nAverage MSE per key over all iterations:")
for key, avg in average_mse.items():
    print(f"  {key}: {avg:.6f}")

# with h5py.File(output, 'r+') as out_file:
#     data_group = out_file['data']
    
#     # Iterate through each demonstration
#     for demo_key in data_group.keys():
#         demo = data_group[demo_key]
#         actions = demo['actions'][:]  # shape (T, action_dim)
#         obs = demo['obs']   # <KeysViewHDF5 ['agentview_image', 'object', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_eef_vel_ang', 'robot0_eef_vel_lin', 'robot0_eye_in_hand_image', 'robot0_gripper_qpos', 'robot0_gripper_qvel', 'robot0_joint_pos', 'robot0_joint_pos_cos', 'robot0_joint_pos_sin', 'robot0_joint_vel']>


#         # import pdb
#         # pdb.set_trace()

        
#         # Add each (state, action) pair into the buffer
#         obs_dict_past = None
#         for t in range(0, actions.shape[0]):
#             if t == 0:
#                 env.reset_to()
                

#             visualize_two_cameras(obs, obs_rgb_keys, t)
#             obs_dict = dict()
#             for key in obs_rgb_keys:
#                 obs_dict[key] = np.moveaxis(obs[key][t],-1,0).astype(np.float32) / 255.   # (C, H, W)
#             for key in obs_lowdim_keys:
#                 obs_dict[key] = obs[key][t].astype(np.float32)  # (dim_L_obs)
#                 if key == 'robot0_eef_quat':
#                     obs_dict[key] = T.quat2mat(obs_dict[key]).reshape(-1).astype(np.float32)
#             if obs_dict_past is None:
#                 obs_dict_past = obs_dict
                
#             obs_dict_combined = combine_obs_dicts(obs_dict_past, obs_dict) 
#             buffer.add([obs_dict_combined, actions[t], actions[t]])
#             print('actions[t]: ', actions[t])
            
#             obs_dict_past = obs_dict

            
            

