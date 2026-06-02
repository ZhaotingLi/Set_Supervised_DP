import logging
import torch
import numpy as np
import matplotlib.pyplot as plt

import pdb

logger = logging.getLogger(__name__)

def visualize_diffusion_vectors(
    model, state, robot_action, teacher_action, saved_path, saved_data_path = None, sampled_action = None,
    resolution=30, steps=[0, 5, 10, 20, 40, 60, 80, 100], device='cuda'
):
    """
    Visualize diffusion noise vectors at specified denoising steps using PyTorch with vector fields.

    Parameters:
    - model: PyTorch model implementing diffusion denoising.
    - state: Input state tensor.
    - robot_action: The robot action (negative example).
    - teacher_action: The teacher action (positive example).
    - saved_path: Path prefix to save plots.
    - resolution: Number of points per dimension.
    - steps: List of denoising steps to visualize.
    - device: PyTorch device ('cpu' or 'cuda').
    - saved_data_path: Path to optionally save the computed noise data.
    """

    # Define action space bounds
    (x_min, x_max), (y_min, y_max) = (-1, 1), (-1, 1)
    # (x_min, x_max), (y_min, y_max) = (-0, 0.6), (-0, 0.6)

    # Generate a 2D grid for the action space
    x = torch.linspace(x_min, x_max, resolution)
    y = torch.linspace(y_min, y_max, resolution)
    action_x, action_y = torch.meshgrid(x, y, indexing='ij')
    action_space_flat = torch.stack([action_x.flatten(), action_y.flatten()], dim=1).to(device)
    action_space_flat_unsqueeze = action_space_flat.unsqueeze(1)
    # Repeat state across all actions
    if hasattr(state, 'numpy'):
        state = state.numpy()
    state = np.reshape(state, [1,  -1]) 
    state = torch.from_numpy(state).float().to(device)
    logger.debug('state shape:  %s', state.shape)
    # state_repeated = state.unsqueeze(0).repeat(action_space_flat.size(0), 1).to(device)
    state_repeated = state.repeat(action_space_flat.size(0), 1).to(device)
    # state_repeated = state_repeated.unsqueeze(1)
    # import pdb; pdb.set_trace()
    # Ensure the model is in eval mode
    model.eval()

    # Visualize for each specified step
    for step in steps:
        with torch.no_grad():
            # pdb.set_trace()
            # Compute diffusion noise vectors at the given denoising step
            noise_pred = model(action_space_flat_unsqueeze, step * torch.ones_like(action_space_flat[:,0]), 
            local_cond=None, global_cond=state_repeated)

            noise_pred = noise_pred.squeeze(1).cpu().numpy()
            # Plot using quiver (vector field)
            plt.figure(figsize=(7, 7))
            cmap = plt.get_cmap('coolwarm')
            magnitudes = np.linalg.norm(noise_pred, axis=1)
            logger.debug('noise_pred : %s', noise_pred.shape)
            plt.quiver(
                action_space_flat[:, 0].cpu().numpy(),
                action_space_flat[:, 1].cpu().numpy(),
                noise_pred[:, 0],
                noise_pred[:, 1],
                magnitudes,
                cmap=cmap,
                scale=10,  # 2
                scale_units='xy',
                angles='xy',
            )
            plt.colorbar(label=f'Noise Magnitude at Step {step}')
            plt.scatter(robot_action[0], robot_action[1], color='yellow', s=100, edgecolor='black', label='Robot Action')
            plt.scatter(teacher_action[0], teacher_action[1], color='green', s=100, edgecolor='black', label='Teacher Action')
            plt.xlabel('Action Dimension 1')
            plt.ylabel('Action Dimension 2')
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.title(f'Diffusion Noise Vector Field at Step {step}')
            plt.legend()

            if sampled_action is not None:
                plt.scatter(
                    sampled_action[:, 0],  # x-coordinates of sampled actions
                    sampled_action[:, 1],  # y-coordinates of sampled actions
                    color='green',         # Color for the sampled actions
                    marker='x',            # Marker style for the sampled actions
                    s=50,                  # Size of the marker
                    alpha=0.5,
                    label='Sampled Actions'  # Legend label
                )
            
            # Save plot
            plt.savefig(f'{saved_path}_step_{step}.png')
            logger.info('save  %s', step)
            plt.close()

            # Optionally save computed data
            if saved_data_path is not None:
                np.savez(
                    f'{saved_data_path}_step_{step}.npz',
                    x=action_space_flat[:, 0].cpu().numpy(),
                    y=action_space_flat[:, 1].cpu().numpy(),
                    noise_vectors=noise_pred,
                    magnitudes=magnitudes,
                    robot_action=robot_action,
                    teacher_action=teacher_action,
                    sampled_action = sampled_action
                )

def visualize_diffusion_vectors_from_saved(
    saved_data_path_prefix,
    saved_plot_path_prefix,
    steps=[0, 5, 10, 20, 40, 60, 80, 100],
    xlim=(-1, 1),
    ylim=(-1, 1),
    quiver_scale=10,
    quiver_width=0.007,    # adjust shaft thickness
    headwidth=4,           # arrow-head width (multiples of width)
    headlength=6,          # arrow-head length (multiples of width)
    circle_color='gray',
    circle_linestyle='-',
    circle_linewidth=2,
    sampled_action_ = None,
    visual_bc = False, # if true, donot draw a- and desiredA
):
    for step in steps:
        data = np.load(f'{saved_data_path_prefix}_step_{step}.npz')
        x = data['x']; y = data['y']
        noise_pred    = data['noise_vectors']
        magnitudes    = data['magnitudes']
        robot_action  = data['robot_action']
        teacher_action= data['teacher_action']
        if sampled_action_ is None:
            sampled_action = data['sampled_action']
        else:
            sampled_action = sampled_action_

        downsample_rate = 4 # should be 2, 4, ..
        if downsample_rate > 1:
            idx = np.arange(0, x.shape[0], downsample_rate)
            x = x[idx]
            y = y[idx]
            noise_pred = noise_pred[idx]
            magnitudes = magnitudes[idx]

        # compute radius
        diff   = teacher_action - robot_action
        radius_ratio = 0.3
        radius = np.linalg.norm(radius_ratio * diff)

        fig, ax = plt.subplots(figsize=(7,7))
        Q = ax.quiver(
            x, y,
            noise_pred[:,0], noise_pred[:,1],
            magnitudes,                  # this is C, the color‐array
            cmap='coolwarm',
            scale=quiver_scale,
            scale_units='xy',
            angles='xy',
            width=quiver_width,
            headwidth=headwidth,
            headlength=headlength
        )

        # # Estimate density via 2D histogram
        # contour_bins=500
        # H, xedges, yedges = np.histogram2d(
        #     sampled_action[:, 0], sampled_action[:, 1],
        #     bins=contour_bins, range=[xlim, ylim]
        # )
        # # Compute grid for contour
        # Xc = (xedges[:-1] + xedges[1:]) / 2
        # Yc = (yedges[:-1] + yedges[1:]) / 2
        # X, Y = np.meshgrid(Xc, Yc)
        # # Draw contour lines
        # contour_levels=7,      # number of contour levels
        # contour_colors='black',
        # contour_linestyle='--' # style for contour lines
        # ax.contour(
        #     X, Y, H.T,
        #     levels=contour_levels,
        #     colors=contour_colors,
        #     linestyles=contour_linestyle
        # )
        
        # now explicitly pass Q into colorbar
        # cbar = fig.colorbar(Q, ax=ax)
        # cbar.set_label(f'Noise Magnitude at Step {step}')

        # scatter key actions
        if not visual_bc:
            ax.scatter(*robot_action,   color='yellow', s=100, edgecolor='k', label='Robot Action')
        ax.scatter(*teacher_action, color='green',  s=100, edgecolor='k', label='Teacher Action')

        # # sampled actions if any
        # if sampled_action is not None:
        #     ax.scatter(
        #         sampled_action[:,0], sampled_action[:,1],
        #         marker='x', s=50, alpha=0.5, label='Sampled Actions'
        #     )

        # draw the circle
        if not visual_bc:
            circle = plt.Circle(
                teacher_action,
                radius,
                edgecolor=circle_color,
                facecolor='none',
                linestyle=circle_linestyle,
                linewidth=circle_linewidth
            )
            ax.add_patch(circle)

        ax.set_aspect('equal', 'box')
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel('Action Dimension 1')
        ax.set_ylabel('Action Dimension 2')
        # ax.set_title(f'Diffusion Noise Vector Field (Step {step})')
        # ax.legend()

        out_path = f'{saved_plot_path_prefix}_step_{step}_draw.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f'Saved plot for step {step} → {out_path}')

def visualize_diffusion_vectors_action_prob(
    model, state, saved_path, saved_data_path = None, sampled_action = None,
    resolution=30, steps=[0, 5, 10, 20, 40, 60, 80, 100], device='cuda'
):
    """
    Visualize diffusion noise vectors at specified denoising steps using PyTorch with vector fields.

    Parameters:
    - model: PyTorch model implementing diffusion denoising.
    - state: Input state tensor.
    - robot_action: The robot action (negative example).
    - teacher_action: The teacher action (positive example).
    - saved_path: Path prefix to save plots.
    - resolution: Number of points per dimension.
    - steps: List of denoising steps to visualize.
    - device: PyTorch device ('cpu' or 'cuda').
    - saved_data_path: Path to optionally save the computed noise data.
    """

    # Define action space bounds
    (x_min, x_max), (y_min, y_max) = (-1, 1), (-1, 1)
    # (x_min, x_max), (y_min, y_max) = (-0, 0.6), (-0, 0.6)

    # Generate a 2D grid for the action space
    x = torch.linspace(x_min, x_max, resolution)
    y = torch.linspace(y_min, y_max, resolution)
    action_x, action_y = torch.meshgrid(x, y, indexing='ij')
    action_space_flat = torch.stack([action_x.flatten(), action_y.flatten()], dim=1).to(device)
    action_space_flat_unsqueeze = action_space_flat.unsqueeze(1)
    # Repeat state across all actions
    if hasattr(state, 'numpy'):
        state = state.numpy()
    state = torch.from_numpy(state).float().to(device)
    logger.debug('state shape:  %s', state.shape)
    # state_repeated = state.unsqueeze(0).repeat(action_space_flat.size(0), 1).to(device)
    state_repeated = state.repeat(action_space_flat.size(0), 1).to(device)
    # state_repeated = state_repeated.unsqueeze(1)
    # Ensure the model is in eval mode
    model.eval()

    # Visualize for each specified step
    for step in steps:
        with torch.no_grad():
            # pdb.set_trace()
            # Compute diffusion noise vectors at the given denoising step
            noise_pred = model(action_space_flat_unsqueeze, step * torch.ones_like(action_space_flat[:,0]), 
            local_cond=None, global_cond=state_repeated)

            noise_pred = noise_pred.squeeze(1).cpu().numpy()
            # Plot using quiver (vector field)
            plt.figure(figsize=(7, 7))
            cmap = plt.get_cmap('coolwarm')
            magnitudes = np.linalg.norm(noise_pred, axis=1)
            logger.debug('noise_pred : %s', noise_pred.shape)
            plt.quiver(
                action_space_flat[:, 0].cpu().numpy(),
                action_space_flat[:, 1].cpu().numpy(),
                noise_pred[:, 0],
                noise_pred[:, 1],
                magnitudes,
                cmap=cmap,
                scale=10,  # 2
                scale_units='xy',
                angles='xy',
            )
            plt.colorbar(label=f'Noise Magnitude at Step {step}')
        
            plt.xlabel('Action Dimension 1')
            plt.ylabel('Action Dimension 2')
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.title(f'Diffusion Noise Vector Field at Step {step}')
            plt.legend()

            if sampled_action is not None:
                plt.scatter(
                    sampled_action[:, 0],  # x-coordinates of sampled actions
                    sampled_action[:, 1],  # y-coordinates of sampled actions
                    color='green',         # Color for the sampled actions
                    marker='x',            # Marker style for the sampled actions
                    s=50,                  # Size of the marker
                    alpha=0.5,
                    label='Sampled Actions'  # Legend label
                )
            
            # Save plot
            plt.savefig(f'{saved_path}_step_{step}.png')
            plt.close()

            # Optionally save computed data
            if saved_data_path is not None:
                np.savez(
                    f'{saved_data_path}_step_{step}.npz',
                    x=action_space_flat[:, 0].cpu().numpy(),
                    y=action_space_flat[:, 1].cpu().numpy(),
                    noise_vectors=noise_pred,
                    magnitudes=magnitudes
                )
