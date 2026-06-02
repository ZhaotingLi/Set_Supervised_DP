
import torch
import matplotlib.pyplot as plt

# --- reuse functions from previous response (ball reflection, cube check, etc.) ---
def check_inside_cube(x, tol=1e-6):
    inside = torch.all(x <= (1.0 + tol), dim=1) & torch.all(x >= (-1.0 - tol), dim=1)
    return inside

def iterative_reflection_cube(xs, x_out, max_iter=2, tol=1e-6):
    for _ in range(max_iter):
        inside_mask = check_inside_cube(x_out, tol=tol)
        if inside_mask.all():
            break
        out_idx = (~inside_mask).nonzero(as_tuple=True)[0]
        x_out[out_idx] = torch.clamp(x_out[out_idx], -1.0, 1.0)
    return x_out

def check_inside_ball(x0, radius, x, tol=1e-6):
    if not torch.is_tensor(radius):
        radius = torch.tensor(radius, device=x.device, dtype=x.dtype)
    else:
        radius = radius.to(device=x.device, dtype=x.dtype)
    N = x.shape[0]
    if radius.ndim == 0:
        r = radius.view(1, 1).expand(N, 1)
    elif radius.ndim == 1:
        r = radius.view(-1, 1)
    elif radius.ndim == 2:
        r = radius
    diff = x - x0
    dist_sq = torch.sum(diff * diff, dim=1, keepdim=True)
    inside = dist_sq <= (r + tol) ** 2
    return inside.squeeze(1)

def line_sphere_intersection_batch(x0_i, radius_i, xs_i, x_out_i, tol=1e-8):
    device = xs_i.device
    dtype = xs_i.dtype
    eps = torch.tensor(tol, device=device, dtype=dtype)
    if not torch.is_tensor(radius_i):
        radius_i = torch.tensor(radius_i, device=device, dtype=dtype)
    else:
        radius_i = radius_i.to(device=device, dtype=dtype)
    M = xs_i.shape[0]
    if radius_i.ndim == 0:
        r = radius_i.view(1, 1).expand(M, 1)
    elif radius_i.ndim == 1:
        r = radius_i.view(-1, 1)
    elif radius_i.ndim == 2:
        r = radius_i
    d = x_out_i - xs_i
    m = xs_i - x0_i
    a = torch.sum(d * d, dim=1)
    b = 2.0 * torch.sum(m * d, dim=1)
    c = torch.sum(m * m, dim=1) - (r.squeeze(1) ** 2)
    a_safe = torch.where(a.abs() <= eps, torch.ones_like(a), a)
    disc = b**2 - 4.0 * a_safe * c
    disc_clamped = torch.clamp(disc, min=0.0)
    sqrt_disc = torch.sqrt(disc_clamped)
    denom = 2.0 * a_safe
    t1 = (-b - sqrt_disc) / denom
    t2 = (-b + sqrt_disc) / denom
    t_candidates = torch.stack((t1, t2), dim=1)
    valid = (t_candidates >= 0.0) & (t_candidates <= 1.0)
    t_masked = torch.where(valid, t_candidates, -torch.ones_like(t_candidates))
    t_selected = t_masked.max(dim=1).values
    t_selected = torch.where(t_selected <= -0.1, torch.zeros_like(t_selected), t_selected)
    intersection = xs_i + t_selected.unsqueeze(1) * d
    return intersection

def iterative_reflection_ball(x0, radius, xs, x_out):
    max_iter = 4
    tol = 1e-6
    device = x0.device
    dtype = x0.dtype
    eps = torch.tensor(1e-8, device=device, dtype=dtype)
    x0 = x0.squeeze()
    xs = xs.squeeze()
    x_out = x_out.squeeze()
    if not torch.is_tensor(radius):
        radius = torch.tensor(radius, device=device, dtype=dtype)
    else:
        radius = radius.to(device=device, dtype=dtype)
    N = x_out.shape[0]
    if radius.ndim == 0:
        r_all = radius.view(1, 1).expand(N, 1)
    elif radius.ndim == 1:
        r_all = radius.view(-1, 1)
    elif radius.ndim == 2:
        r_all = radius
    for i in range(max_iter):
        inside_cube = check_inside_cube(x_out)
        inside_ball_mask = check_inside_ball(x0, r_all, x_out)
        intersected = ~inside_ball_mask
        intersected_indices = intersected.nonzero(as_tuple=True)[0]
        if (~inside_cube).sum() == 0 and (~inside_ball_mask).sum() == 0:
            break
        if intersected_indices.numel() > 0:
            x0_i = x0[intersected_indices]
            xs_i = xs[intersected_indices]
            x_out_i = x_out[intersected_indices]
            radius_i = r_all[intersected_indices]
            intersection = line_sphere_intersection_batch(x0_i, radius_i, xs_i, x_out_i)
            d_x = (x_out_i - xs_i)
            delta0 = torch.norm(d_x, dim=1, keepdim=True)
            delta_inter = torch.norm(intersection - xs_i, dim=1, keepdim=True)
            diff = delta0 - delta_inter
            grad = -(d_x) / torch.clamp(torch.norm(d_x, dim=1, keepdim=True), min=eps)
            x_out[intersected_indices] = intersection + diff * grad
        if i > 1 and intersected_indices.numel() > 0:
            x_out[intersected_indices] = xs[intersected_indices]
            x_out = torch.clamp(x_out, -1, 1)
            break
        x_out = iterative_reflection_cube(xs=xs, x_out=x_out)
    return x_out.unsqueeze(1)

# --- Test + Visualization in 2D ---
def test_reflection_ball_2d():
    N = 200
    dim = 2
    # Ball center
    x0 = torch.zeros(N, dim) + 0.5
    # Ball radius
    radius = 0.8
    # xs: inside points
    xs = torch.zeros(N, dim)  # just center for simplicity
    # random candidate points (some outside cube, some outside ball)
    x_out = torch.empty(N, dim).uniform_(-1.5, 1.5)

    x_out_reflected = iterative_reflection_ball(x0, radius, xs, x_out.clone())
    x_out_ref = x_out_reflected.squeeze(1).detach().numpy()
    x_out_np = x_out.detach().numpy()

    # Plot
    fig, ax = plt.subplots(figsize=(6,6))
    # draw cube [-1,1]^2
    ax.plot([-1,1,1,-1,-1], [-1,-1,1,1,-1])
    # draw ball
    theta = torch.linspace(0, 2*3.14159, 200)
    cx = radius * torch.cos(theta) + x0[:, 0]
    cy = radius * torch.sin(theta) + x0[:, 1]
    ax.plot(cx.numpy(), cy.numpy())
    # original
    ax.scatter(x_out_np[:,0], x_out_np[:,1], s=10, alpha=0.3, label='original')
    # reflected
    ax.scatter(x_out_ref[:,0], x_out_ref[:,1], s=10, label='reflected')
    ax.set_aspect('equal', 'box')
    ax.legend()
    ax.set_title("Reflection into Ball ∩ Cube in 2D")
    plt.show()



def iterative_reflection_cube_with_traj(xs, x_out, max_iter=2, tol=1e-6):
    traj = [x_out.clone()]
    for _ in range(max_iter):
        inside_mask = check_inside_cube(x_out, tol=tol)
        if inside_mask.all():
            break
        out_idx = (~inside_mask).nonzero(as_tuple=True)[0]
        x_out[out_idx] = torch.clamp(x_out[out_idx], -1.0, 1.0)
        traj.append(x_out.clone())
    return x_out, traj

def iterative_reflection_ball_with_traj(x0, radius, xs, x_out):
    max_iter = 4
    tol = 1e-6
    device = x0.device
    dtype = x0.dtype
    eps = torch.tensor(1e-8, device=device, dtype=dtype)
    x0 = x0.squeeze()
    xs = xs.squeeze()
    x_out = x_out.squeeze()
    if not torch.is_tensor(radius):
        radius = torch.tensor(radius, device=device, dtype=dtype)
    else:
        radius = radius.to(device=device, dtype=dtype)
    N = x_out.shape[0]
    if radius.ndim == 0:
        r_all = radius.view(1, 1).expand(N, 1)
    elif radius.ndim == 1:
        r_all = radius.view(-1, 1)
    elif radius.ndim == 2:
        r_all = radius
    traj = [x_out.clone()]
    for i in range(max_iter):
        inside_cube = check_inside_cube(x_out)
        inside_ball_mask = check_inside_ball(x0, r_all, x_out)
        intersected = ~inside_ball_mask
        intersected_indices = intersected.nonzero(as_tuple=True)[0]
        if (~inside_cube).sum() == 0 and (~inside_ball_mask).sum() == 0:
            break
        if intersected_indices.numel() > 0:
            x0_i = x0[intersected_indices]
            xs_i = xs[intersected_indices]
            x_out_i = x_out[intersected_indices]
            radius_i = r_all[intersected_indices]
            intersection = line_sphere_intersection_batch(x0_i, radius_i, xs_i, x_out_i)
            d_x = (x_out_i - xs_i)
            delta0 = torch.norm(d_x, dim=1, keepdim=True)
            delta_inter = torch.norm(intersection - xs_i, dim=1, keepdim=True)
            diff = delta0 - delta_inter
            grad = -(d_x) / torch.clamp(torch.norm(d_x, dim=1, keepdim=True), min=eps)
            x_out[intersected_indices] = intersection + diff * grad
        traj.append(x_out.clone())
        if i > 1 and intersected_indices.numel() > 0:
            x_out[intersected_indices] = xs[intersected_indices]
            x_out = torch.clamp(x_out, -1, 1)
            traj.append(x_out.clone())
            break
        x_out, cube_traj = iterative_reflection_cube_with_traj(xs=x_out, x_out=x_out)
        traj.extend(cube_traj[1:])
    return x_out.unsqueeze(1), traj

# --- Modified Test: Draw trajectory of one point ---
def test_reflection_ball_2d_trajectory():
    N = 200
    dim = 2
    x0 = torch.zeros(N, dim) +0.5
    radius = 0.8
    xs = torch.zeros(N, dim)  -1.0
    x_out = torch.empty(N, dim).uniform_(-1.5, 1.5)
    final, traj = iterative_reflection_ball_with_traj(x0, radius, xs, x_out.clone())
    idx = 0
    point_init = x_out[idx].detach().numpy()
    traj_np = [t[idx].detach().numpy() for t in traj]
    fig, ax = plt.subplots(figsize=(6,6))
    ax.plot([-1,1,1,-1,-1], [-1,-1,1,1,-1])
    theta = torch.linspace(0, 2*3.14159, 200)
    cx = radius * torch.cos(theta) + x0[:, 0]
    cy = radius * torch.sin(theta) + x0[:, 1]
    ax.plot(cx.numpy(), cy.numpy())
    ax.scatter(point_init[0], point_init[1], color='blue', label='start')
    xs_np = xs[idx].detach().numpy()
    ax.scatter(xs_np[0], xs_np[1], color='green', label='xs')
    traj_np_arr = torch.tensor(traj_np).numpy()
    ax.plot(traj_np_arr[:,0], traj_np_arr[:,1], marker='o', label='trajectory')
    ax.set_aspect('equal', 'box')
    ax.legend()
    ax.set_title(f"Trajectory of sample index {idx}")
    plt.show()

if __name__ == "__main__":
    # Run test
    # test_reflection_ball_2d()
    test_reflection_ball_2d_trajectory()
