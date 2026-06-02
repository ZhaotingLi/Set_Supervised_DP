import argparse
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Compare two robomimic HDF5 datasets.")
parser.add_argument("--v1-path", required=True, help="Path to the first dataset.")
parser.add_argument("--v2-path", required=True, help="Path to the second dataset.")
args = parser.parse_args()
v1_path = os.path.expanduser(os.path.expandvars(args.v1_path))
v2_path = os.path.expanduser(os.path.expandvars(args.v2_path))

# Open datasets
f1 = h5py.File(v1_path, 'r')
f2 = h5py.File(v2_path, 'r')

# Observation shape metadata
shape_meta = {
    "obs": {
        "agentview_image": {"shape": [3, 84, 84], "type": "rgb"},
        "robot0_eye_in_hand_image": {"shape": [3, 84, 84], "type": "rgb"},
        "robot0_eef_pos": {"shape": [3]},
        "robot0_eef_quat": {"shape": [9]},  # rotation matrix
        "robot0_gripper_qpos": {"shape": [2]}
    },
    "action": {"shape": [7]}
}

# Separate keys
rgb_keys = []
lowdim_keys = []
for key, attr in shape_meta['obs'].items():
    if attr.get("type", "low_dim") == "rgb":
        rgb_keys.append(key)
    else:
        lowdim_keys.append(key)

# Get episode lists (intersection)
demos1 = sorted(f1['data'].keys(), key=lambda x: int(x[5:]))
demos2 = sorted(f2['data'].keys(), key=lambda x: int(x[5:]))
shared_demos = list(set(demos1).intersection(demos2))
shared_demos.sort(key=lambda x: int(x[5:]))

print(f"Comparing {len(shared_demos)} shared episodes")

# Initialize diff stats
action_diffs = []
obs_diffs = {key: [] for key in rgb_keys + lowdim_keys}

# Compare each episode
for ep in shared_demos:
    actions1 = f1[f"data/{ep}/actions"][()]
    actions2 = f2[f"data/{ep}/actions"][()]
    min_len = min(len(actions1), len(actions2))

    # Action diff
    action_diff = np.abs(actions1[:min_len] - actions2[:min_len])
    action_diffs.append(np.mean(action_diff))

    # Obs diff for each key
    for key in rgb_keys + lowdim_keys:
        obs1 = f1[f"data/{ep}/obs/{key}"][()]
        obs2 = f2[f"data/{ep}/obs/{key}"][()]
        obs_diff = np.abs(obs1[:min_len] - obs2[:min_len])
        obs_diffs[key].append(np.mean(obs_diff))

# Summary
print("\n=== Mean Absolute Differences ===")
print(f"Actions: {np.mean(action_diffs):.6f}")
for key in obs_diffs:
    mean_diff = np.mean(obs_diffs[key])
    print(f"Obs[{key}]: {mean_diff:.6f}")



# Function to visualize image differences
def show_image_comparison(ep, t, key):
    def prepare_image_for_display(img):
        if img.shape[0] == 3:
            return np.transpose(img, (1, 2, 0)).astype(np.uint8)
        elif img.shape[-1] == 3:
            return img.astype(np.uint8)
        elif img.shape[1] == 3:
            return np.transpose(img, (0, 2, 1)).astype(np.uint8)
        else:
            raise ValueError(f"Unrecognized image shape: {img.shape}")

    img1 = prepare_image_for_display(f1[f"data/{ep}/obs/{key}"][t])
    img2 = prepare_image_for_display(f2[f"data/{ep}/obs/{key}"][t])
    diff = np.abs(img1.astype(np.int32) - img2.astype(np.int32)).astype(np.uint8)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(img1)
    axs[0].set_title(f"{key} - v3")
    axs[1].imshow(img2)
    axs[1].set_title(f"{key} - v2")
    axs[2].imshow(diff)
    axs[2].set_title("abs diff")
    for ax in axs:
        ax.axis('off')
    plt.suptitle(f"{ep} - timestep {t} - {key}")
    plt.tight_layout()
    plt.show()


# Visualize a few examples
example_episodes = shared_demos[:100]  # First 3 episodes
timesteps = [0, 10, 20, 30, 50, 70, 90, 100]              # Some example timesteps

for ep in example_episodes:
    for t in timesteps:
        for key in rgb_keys:
            try:
                show_image_comparison(ep, t, key)
            except Exception as e:
                print(f"Failed to show {ep} t={t} key={key}: {e}")
