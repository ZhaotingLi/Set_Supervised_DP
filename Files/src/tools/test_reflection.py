import logging

import pdb


import torch
import time

logger = logging.getLogger(__name__)


# def line_cone_intersection_batch(x0, v_normalized, alpha, xs, x_out, intersected_indices, iter):
def line_cone_intersection_batch(x0_i, v_i, alpha,  xs_i, x_out_i, iter):

    eps = torch.tensor(1e-8, device=x0_i.device, dtype=x0_i.dtype)
    cos_half_alpha = torch.cos(alpha / 2.0)
    cos_half_alpha_sq = cos_half_alpha**2
    r, d = xs_i - x0_i, x_out_i - xs_i

    A = torch.sum(d * v_i, dim=1)
    B = torch.sum(r * v_i, dim=1)
    d_norm_sq = torch.sum(d * d, dim=1)
    r_norm_sq = torch.sum(r * r, dim=1)
    r_dot_d = torch.sum(r * d, dim=1)

    a = A**2 - cos_half_alpha_sq* d_norm_sq
    b = 2 * (A * B - cos_half_alpha_sq * r_dot_d)
    c = B**2 - cos_half_alpha_sq * r_norm_sq

    t_solutions = torch.ones((x_out_i.shape[0]), device=x0_i.device, dtype=x0_i.dtype)

    degenerate_mask = a.abs() <= eps
    if degenerate_mask.any():
        degenerate_mask_indices = torch.nonzero(degenerate_mask, as_tuple=True)[0]  
        b_deg = b[degenerate_mask_indices]
        c_deg = c[degenerate_mask_indices]
        valid_b_mask = torch.nonzero(b_deg.abs() > eps, as_tuple=True)[0]  
        t_solutions[degenerate_mask_indices][valid_b_mask] = -c_deg[valid_b_mask] / b_deg[valid_b_mask]

    non_degenerate_mask = ~degenerate_mask
    # if non_degenerate_mask.any():
    non_degenerate_mask_indices = torch.nonzero(non_degenerate_mask, as_tuple=True)[0] 
    a_nd, b_nd, c_nd = a[non_degenerate_mask_indices], b[non_degenerate_mask_indices], c[non_degenerate_mask_indices]
    disc = b_nd**2 - 4 * a_nd * c_nd
    # if (disc < 0).any():
    #     # raise ValueError("Negative discriminant encountered.")
    #     print("Negative discriminant encountered.")
    # # sqrt_disc = torch.sqrt(disc)
    sqrt_disc = torch.sqrt(torch.clamp(disc, min=0))

    denom = 2 * a_nd 
    t1 = (-b_nd + sqrt_disc) / denom
    t2 = (-b_nd - sqrt_disc) / denom
    t_candidates = torch.stack((t1, t2), dim=1)
    valid_candidates = (t_candidates >= 0) & (t_candidates < 1)
    masked_t_candidates = torch.where(valid_candidates, t_candidates, -torch.ones_like(t_candidates))
    t_selected = masked_t_candidates.max(dim=1).values
    # invalid_solution_mask = t_selected <= -0.1
    # if invalid_solution_mask.any():
    #     invalid_solution_mask = torch.nonzero(invalid_solution_mask, as_tuple=True)[0] 
    #     t_selected[invalid_solution_mask] = 0.0
    t_selected = torch.where(t_selected <= -0.1, torch.zeros_like(t_selected), t_selected)
    t_solutions[non_degenerate_mask_indices] = t_selected


    intersection_points = xs_i + t_solutions.unsqueeze(1) * d

    return intersection_points



def check_inside_cone(x0, v_normalized, alpha, xs):
    # check xs inside the cone, if ouside, raise error
    eps = torch.tensor(1e-8, device=x0.device, dtype=x0.dtype)
    # Normalize v for each batch element.
   
    cos_half_alpha = torch.cos(alpha / 2.0)
    vec_x_s = xs - x0           # shape (N, dim)
    norm_vec_x_s = torch.norm(vec_x_s, dim=1)  # shape (N,)
    # For those away from the apex, check if the angle is less than alpha/2:
    inside_mask_xs = (norm_vec_x_s > eps) & ((torch.sum(vec_x_s * v_normalized, dim=1) / norm_vec_x_s) >= cos_half_alpha)
    # num_outside_xs = torch.sum(~inside_mask_xs)
    if inside_mask_xs.all():
        return True
    else:
        return False
    

def iterative_refelection_cone(x0, v, alpha, xs, x_out):

    max_iter = 4    # maximum number of iterations
    tol = 1e-6       # convergence tolerance
    eps = torch.tensor(1e-8, device=x0.device, dtype=x0.dtype)
    cos_half_alpha = torch.cos(alpha / 2.0)
    x0 = x0.squeeze()
    v = v.squeeze()
    xs = xs.squeeze()
    x_out = x_out.squeeze()
    v_normalized = v / torch.clamp(torch.norm(v, dim=1, keepdim=True), min=eps)

    # total_start_time = time.time()

    for i in range(max_iter):


        inside = check_inside_cube(x_out)
        num_outside_cube = torch.sum(~inside)

        vec_x_out = x_out - x0
        norm_vec_x_out = torch.norm(vec_x_out, dim=1)
        inside_mask = (norm_vec_x_out > eps) & ((torch.sum(vec_x_out * v_normalized, dim=1) / norm_vec_x_out) >= cos_half_alpha)
        intersected = ~inside_mask
        intersected_indices = torch.nonzero(intersected, as_tuple=True)[0]     # Convert the boolean mask to integer indices
        num_outside_cone = torch.sum(intersected)


        if num_outside_cone == 0 and num_outside_cube == 0:
            break
        
        if num_outside_cone > 0:
            x0_i, v_i, xs_i, x_out_i = x0[intersected_indices], v_normalized[intersected_indices], xs[intersected_indices], x_out[intersected_indices]
            intersection = line_cone_intersection_batch(x0_i, v_i, alpha, xs_i, x_out_i, iter=i)

            d_x = (x_out_i - xs_i)
            delta0 = torch.norm(d_x, dim=1, keepdim=True)
            delta_inter = torch.norm(intersection - xs_i, dim=1, keepdim=True)
            diff = delta0 - delta_inter
            grad = -(d_x) / torch.clamp(torch.norm(d_x, dim=1, keepdim=True), min=1e-8)

            damp_coefficent = 1.0 
            x_out[intersected_indices] = intersection + damp_coefficent * diff * grad

        if i > 1:
            x_out[intersected_indices] = xs[intersected_indices]
            x_out = torch.clamp(x_out, -1, 1)
            break

        x_out = iterative_reflection_cube(xs=xs, x_out=x_out)


    # total_elapsed_time = time.time() - total_start_time
    # print(f"Total elapsed time: {total_elapsed_time:.6f} seconds.")

    return x_out.unsqueeze(1)
    


def check_inside_cube(x, tol=1e-6):
    """
    Check if each point in x (of shape (N, dim)) lies within the cube defined by [-1, 1] in every dimension.
    
    Parameters:
        x:   Tensor of shape (N, dim) containing the points.
        tol: Tolerance for the boundary check.
    
    Returns:
        A Boolean tensor of shape (N,) where True indicates the point is within [-1,1] (with tolerance) in all dimensions.
    """
    # Each coordinate must be <= 1+tol and >= -1-tol.
    inside = torch.all(x <= (1.0 + tol), dim=1) & torch.all(x >= (-1.0 - tol), dim=1)
    return inside


def line_cube_intersection_batch(xs_i, x_out_i, tol=1e-8):
    """
    Compute the intersection of the ray from xs (an interior point) to x_out with the cube boundaries 
    (each coordinate must be between -1 and 1). 
    
    For each sample in the batch:
      - If x_out is already inside the cube, the function returns x_out.
      - Otherwise, for each dimension the candidate t is computed as:
            if (x_out - xs)_i > 0: t_i = (1 - xs_i) / (x_out_i - xs_i)
            if (x_out - xs)_i < 0: t_i = (-1 - xs_i) / (x_out_i - xs_i)
        and then t = min(t_i) (over all i) is chosen to yield the first intersection.
    
    Parameters:
        xs:   Tensor of shape (N, dim) known to be inside the cube.
        x_out:Tensor of shape (N, dim) candidate points (possibly outside the cube).
        tol:  A small tolerance to avoid division by zero.
    
    Returns:
        intersection: Tensor of shape (N, dim) with the computed intersection points on the cube boundary.
        out_mask:     Boolean tensor of shape (N,) that is True for samples where x_out was outside the cube.
    """
    # Check which samples have x_out inside the cube.
    # inside = check_inside_cube(x_out)
    # intersection = x_out.clone()
    
        
    # Compute the direction vectors from xs to x_out (shape: (N, dim)).

    d = x_out_i - xs_i

    # Prepare a tensor to hold candidate t for each dimension; initialize with infinity.
    t_candidates = torch.full_like(d, float('inf'))
    
    # For dimensions with a positive direction, compute t = (1 - xs) / d.
    # pos_mask = d > tol
    # if pos_mask.any():
    #     pos_mask = torch.nonzero(pos_mask, as_tuple=True)[0] 
    #     t_candidates[pos_mask] = (1.0 - xs[pos_mask]) / d[pos_mask]
    
    # # For dimensions with a negative direction, compute t = (-1 - xs) / d.
    # neg_mask = d < -tol
    # if neg_mask.any():
    #     neg_mask = torch.nonzero(neg_mask, as_tuple=True)[0] 
    #     t_candidates[neg_mask] = (-1.0 - xs[neg_mask]) / d[neg_mask]

    # Compute boolean masks for positive and negative directions.
    pos_mask = d > tol
    neg_mask = d < -tol

    # Update t_candidates using boolean indexing directly.
    t_candidates[pos_mask] = (1.0 - xs_i[pos_mask]) / d[pos_mask]
    t_candidates[neg_mask] = (-1.0 - xs_i[neg_mask]) / d[neg_mask]

    
    # For each sample (row), select the minimum t among dimensions.
    t, _ = torch.min(t_candidates, dim=1)  # shape (N,)
    t = torch.clamp(t, min=0.0, max=1.0)
    intersection = xs_i + t.unsqueeze(1) * d

    return intersection


def iterative_reflection_cube(xs, x_out, max_iter=2, tol=1e-6):
    """
    Iteratively reflect candidate points x_out (per batch sample) into the interior of the cube defined by [-1, 1]
    in each dimension.
    
    Here, xs is an inside point (serving as the origin of the reflection ray) and x_out is the candidate (possibly outside).
    The reflection is performed by:
        1. Computing the intersection point of the ray (from xs to x_out) with the cube boundary.
        2. Reflecting x_out across the boundary via: x_out_new = 2 * intersection - x_out.
    The process is repeated until all x_out are within the cube (or a maximum number of iterations is reached).
    
    Parameters:
        xs:    Tensor of shape (N, dim) known to be inside the cube.
        x_out: Tensor of shape (N, dim) candidate points (possibly outside the cube).
        max_iter: Maximum number of iterations.
        tol:      Tolerance level for the inside check.
    
    Returns:
        x_out: The updated tensor (shape (N, dim)) with all points reflected inside the cube.
    """
    for i in range(max_iter):
        
        inside_mask = check_inside_cube(x_out, tol=tol)
        if inside_mask.all():
            # print(f"Cube reflection converged in {i} iterations.")
            break
        out_mask = ~inside_mask  # Samples that require reflection
        out_mask_indices = torch.nonzero(out_mask, as_tuple=True)[0] 
        x_out_i = x_out[out_mask_indices]
        xs_i = xs[out_mask_indices]
        intersection = line_cube_intersection_batch(xs_i, x_out_i, tol=tol)

        x_out_selected = x_out[out_mask_indices]
        xs_selected = xs[out_mask_indices]
        d_x = (x_out_selected -xs_selected)
        delta0 = torch.norm(d_x, dim=1, keepdim=True)
        delta_inter = torch.norm(intersection - xs_selected, dim=1, keepdim=True)
        # Compute the (signed) difference relative to the initial distance.
        diff = delta0- delta_inter  # shape: (N, 1)

        grad = - (d_x) / torch.clamp(torch.norm(d_x, dim=1, keepdim=True), min=1e-8)
        x_out[out_mask_indices]= intersection + diff * grad

    else:
        logger.warning("Warning: maximum iterations reached before all points were reflected inside the cube.")
    
    return x_out


# === Example Usage in Batch Mode ===
if __name__ == "__main__":
    # Create a batch of 4 samples in 3D.
    N, dim = 4, 3
    # Define apex points x0 (one per sample).
    x0 = torch.zeros(N, dim)
    
    # Define central directions; here we use the same direction for simplicity.
    v = torch.tensor([[0.0, 0.0, 1.0]]).repeat(N, 1) / 10
    
    # Define a full cone angle (in radians). For example, 60° full angle yields 30° half-angle.
    alpha = torch.tensor(60.0 * (3.141592653589793 / 180.0))
    
    # Define xs (inside the cone); one per sample.
    xs = torch.tensor([[0.0, 0.0, 2.0],
                       [0.0, 0.0, 3.0],
                       [0.1, 0.1, 2.0],
                       [0.0, -0.1, 2.5]]) /10
    
    # Define x_out for each sample.
    # Sample 0: Outside the cone.
    # Sample 1: Outside the cone.
    # Sample 2: Already inside the cone.
    # Sample 3: Outside the cone.
    x_out = torch.tensor([[-5.0, 1.0, 2.0],
                          [-1.5, 1.0, 3.0],
                          [0.2, -3, -2.0],
                          [-5.0, -1.0, 2.5]])/10
    # '''1st Iteration'''
    # delta0 = torch.norm(x_out - x0, dim=1, keepdim=True)
    # intersection, intersected = line_cone_intersection_batch(x0, v, alpha, xs, x_out)
    # # intersection, intersected = line_cone_intersection_batch(x0, v, alpha, xs, x_out)
    # print("Batch Intersection Points:")
    # print(intersection)
    # print("intersected: ", intersected)
    # check = check_intersection_batch(x0, v, alpha, intersection)
    # print("Intersection valid on boundary (batch check):")

    # '''2st Iteration'''
    # delta_inter_x0 = torch.norm(intersection - x0, dim=1, keepdim=True)
    # gradient = -1 *(x_out-xs) / torch.clamp(torch.norm(x_out-xs, dim=1, keepdim=True), min=1e-8)
    # xs1 = intersection
    # x_out1 = intersection + (delta0 - delta_inter_x0) * gradient 
    # intersection, intersected = line_cone_intersection_batch(x0, v, alpha, xs1, x_out1)
    # # x1 = intersection +  
    # # intersection, intersected = line_cone_intersection_batch(x0, v, alpha, xs, x_out)
    # print("Batch Intersection Points:")
    # print(intersection)
    # print("intersected: ", intersected)
    # check = check_intersection_batch(x0, v, alpha, intersection)
    # print("Intersection valid on boundary (batch check):")
    # print(check)

    # '''Iteration continues, and stops when delta_inter_x0 <= 0 '''

    iterative_refelection_cone(x0, v, alpha, xs, x_out)