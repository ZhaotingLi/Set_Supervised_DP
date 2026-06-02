import logging
import numpy as np

logger = logging.getLogger(__name__)

# wrong_oracle_action = True
wrong_oracle_action = False
wrong_oracle_action_count = 0

def sample_orthogonal_vector(h_human):
    dim_a = h_human.shape[0]

    # Normalize h_human to make the vector a unit vector
    h_human_normalized = h_human / np.linalg.norm(h_human)

    # Initialize an empty list to hold the basis vectors
    basis_vectors = []

    # Use Gram-Schmidt process to generate basis vectors
    for i in range(dim_a - 1):  # We need dim_a - 1 vectors to define the plane
        # Generate a random vector
        random_vector = np.random.normal(size=dim_a)
        # Subtract the component that is in the direction of h_human
        projection = np.dot(random_vector, h_human_normalized)
        random_vector -= projection * h_human_normalized
        # Orthonormalize with respect to the existing basis vectors
        for basis_vector in basis_vectors:
            projection = np.dot(random_vector, basis_vector)
            random_vector -= projection * basis_vector
        # Normalize the new basis vector
        random_vector /= np.linalg.norm(random_vector)
        # Add the new basis vector to the list
        basis_vectors.append(random_vector)

    # Stack the basis vectors
    basis_vectors = np.vstack(basis_vectors)

    # Now, sample one set of coefficients to generate one orthogonal vector
    sample_h_size = 1
    coefficients = np.random.uniform(low=-1, high=1, size=(sample_h_size, dim_a - 1))
    sampled_orthogonal_vector = np.dot(coefficients, basis_vectors)

    # Normalize the sampled vector and scale it to the norm of h_human
    norm_h_human = np.linalg.norm(h_human)
    sampled_orthogonal_vector = sampled_orthogonal_vector / np.linalg.norm(sampled_orthogonal_vector)
    sampled_orthogonal_vector *= norm_h_human

    # Ensuring the shape of the sampled vector is the same as h_human
    sampled_orthogonal_vector = np.squeeze(sampled_orthogonal_vector)  # Remove any singleton dimensions
    return sampled_orthogonal_vector


def sample_orthogonal_vector_determinisitic(h_human, fixed_basis=None, fixed_coefficients=None):
    dim_a = h_human.shape[0]

    # Normalize h_human to make the vector a unit vector
    h_human_normalized = h_human / np.linalg.norm(h_human)

    # Initialize the list with the fixed basis vectors
    if fixed_basis is None:
        # If no fixed basis is provided, create a simple orthogonal basis
        fixed_basis = np.eye(dim_a)  # This creates an identity matrix, which represents orthogonal unit vectors
    else:
        # Ensure the fixed basis is orthogonal and has the correct dimension
        assert fixed_basis.shape == (dim_a, dim_a), "Fixed basis must be a square matrix of the same dimension as h_human"

    # Remove the component of the fixed basis vectors that is in the direction of h_human
    for i in range(dim_a):
        projection = np.dot(fixed_basis[i], h_human_normalized)
        fixed_basis[i] -= projection * h_human_normalized
        fixed_basis[i] /= np.linalg.norm(fixed_basis[i])  # Normalize

    # Use fixed coefficients if provided, else use a default fixed coefficient
    if fixed_coefficients is None:
        fixed_coefficients = np.ones((1, dim_a))  # A simple fixed coefficient
        fixed_coefficients = fixed_coefficients/ np.linalg.norm(fixed_coefficients)
        # fixed_coefficients[-1] =0
        randomized_coefficients = np.random.uniform(low=-1, high=1, size=(1, dim_a)) 
        randomized_coefficients = randomized_coefficients / np.linalg.norm(randomized_coefficients)
        # fixed_coefficients = 0.5 * fixed_coefficients + 0.5 * randomized_coefficients # A simple fixed coefficient
        fixed_coefficients = 0.2 * fixed_coefficients + 0.8 * randomized_coefficients # A simple fixed coefficient
    else:
        # Ensure the fixed coefficients have the correct shape
        assert fixed_coefficients.shape == (1, dim_a), "Fixed coefficients must be of shape (1, dim_a)"

    sampled_orthogonal_vector = np.dot(fixed_coefficients, fixed_basis)

    # Normalize the sampled vector and scale it to the norm of h_human
    norm_h_human = np.linalg.norm(h_human)
    sampled_orthogonal_vector = sampled_orthogonal_vector / np.linalg.norm(sampled_orthogonal_vector)
    sampled_orthogonal_vector *= norm_h_human

    # Ensuring the shape of the sampled vector is the same as h_human
    sampled_orthogonal_vector = np.squeeze(sampled_orthogonal_vector)  # Remove any singleton dimensions
    return sampled_orthogonal_vector


def get_wrong_oracle(action_teacher, action):
    action_max = np.max(abs(action_teacher))
    if action_max > 1:
        action_teacher =  action_teacher / action_max
    difference = action_teacher - action
    # generate wrong_anction_teacher which is wrong_anction_teacher = action - difference
    wrong_anction_teacher = action - difference
    return wrong_anction_teacher


# for HG-DAgger and policy contrastive intervention
def non_perfect_oracle(action_teacher, action, config):
    oracle_teacher_noise = config['oracle_teacher_noise']
    oracle_teacher_magnitude_noise = config['oracle_teacher_magnitude_noise']
    oracle_teacher_magnitude_noise_parameter = config['oracle_teacher_magnitude_noise_parameter']
    if_biased_directional_noise = config['if_biased_directional_noise']
    oracle_teacher_Gaussian_noise = config['oracle_teacher_Gaussian_noise']
    abs_action = config['use_abs_action']

    # rand_number = np.random.random(1)[0]
    # if wrong_oracle_action and rand_number < 0.1:
    if wrong_oracle_action:
        global wrong_oracle_action_count
        wrong_oracle_action_count = wrong_oracle_action_count + 1
        if wrong_oracle_action_count > 10:
            wrong_oracle_action_count = 0
            action_teacher = get_wrong_oracle(action_teacher, action)
    # print("wrong_oracle_action_count: ", wrong_oracle_action_count)

    action_max = np.max(abs(action_teacher))
    if action_max > 1:
        action_teacher =  action_teacher / action_max

    # if oracle_teacher_noise == 0 and oracle_teacher_magnitude_noise == False:  # if no noise, directly output teacher's action
    #     return action_teacher
    
    difference = action_teacher - action

    difference = np.array(difference)
    
    if oracle_teacher_noise != 0:
        # sample the noise (same maginitude but different orientation)
        # you can also vary the maginitude 
        difference_vectical_vector = sample_orthogonal_vector(difference)
        # difference_vectical_vector = sample_orthogonal_vector_determinisitic(difference)
        # alpha = 60 * np.pi / 180
        if if_biased_directional_noise:
            biased_angle = np.sign(np.random.uniform(low=-1, high=1)) * 15
            alpha = (oracle_teacher_noise+biased_angle) * np.pi / 180
        else:
            alpha = oracle_teacher_noise * np.pi / 180
        # alpha = 0 * np.pi / 180
        # alpha = np.random.randint(low=0, high=45) * np.pi / 180
        difference_with_noise = difference * np.cos(alpha) + difference_vectical_vector * np.sin(alpha)
    else:
        difference_with_noise = difference

    # introduce the noise for the magnitude of corrections
    if oracle_teacher_magnitude_noise:
        # oracle_teacher_magnitude_noise_parameter = 4.0
        # magnitude_coefficients = np.random.uniform(low=0.5, high=2)
        # magnitude_coefficients = np.random.uniform(low=1.0/oracle_teacher_magnitude_noise_parameter, high= oracle_teacher_magnitude_noise_parameter)
        
        # prob_magnitude_noise = np.sign(np.random.uniform(low=-1, high=1))
        # if prob_magnitude_noise > 0:
        #     magnitude_coefficients = oracle_teacher_magnitude_noise_parameter
        # else:
        #     magnitude_coefficients = 1.0 / oracle_teacher_magnitude_noise_parameter
        
        magnitude_coefficients = np.random.uniform(low=0.001, high=oracle_teacher_magnitude_noise_parameter)
    else:
        magnitude_coefficients = 1

    non_optimal_action_teacher = magnitude_coefficients * difference_with_noise + action

    if oracle_teacher_Gaussian_noise:
        # non_optimal_action_teacher = non_optimal_action_teacher + np.random.normal(loc=0.0, scale=np.linalg.norm(difference/1.0), size=action.shape)
        if abs_action:
            # gaussian_noise_scale = np.minimum(np.linalg.norm(difference/2.0), 0.025)
            gaussian_noise_scale = np.minimum(np.linalg.norm(difference/2.0), 0.04)
            logger.debug('gaussian_noise_scale:  %s', gaussian_noise_scale)
            non_optimal_action_teacher = non_optimal_action_teacher + np.random.normal(loc=0.0, scale=gaussian_noise_scale, size=action.shape)
        else:
            non_optimal_action_teacher = non_optimal_action_teacher + np.random.normal(loc=0.0, scale=np.linalg.norm(difference/2.0), size=action.shape)
    # non_optimal_action_teacher = difference_with_noise + action
    logger.debug('Gaussian Noise:  %s oracle_teacher_noise:  %s  oracle_teacher_magnitude_noise:  %s  oracle_teacher_magnitude_noise_parameter:  %s  magnitude_coefficients:  %s', oracle_teacher_Gaussian_noise, oracle_teacher_noise, oracle_teacher_magnitude_noise, oracle_teacher_magnitude_noise_parameter, magnitude_coefficients)
    # clip the action if some dim of it is out of its limits
    dim_a = action.shape[0]
    lower_limits = np.full(dim_a, -1)  # Lower limits for each dimension, replace with actual values
    upper_limits = np.full(dim_a, 1)   # Upper limits for each dimension, replace with actual values

    # Clip the actions to be within the limits
    clipped_actions = np.clip(non_optimal_action_teacher, lower_limits, upper_limits)
    return clipped_actions


# for policy contrastive method
def non_perfect_oracle_normed(action_teacher, action, config):
    oracle_teacher_noise = config['oracle_teacher_noise']


    action_max = np.max(abs(action_teacher))
    if action_max > 1:
        action_teacher =  action_teacher / action_max
    difference = action_teacher - action

    difference = np.array(difference)

    # sample the noise (same maginitude but different orientation)
    # you can also vary the maginitude 
    difference_vectical_vector = sample_orthogonal_vector(difference)
    # difference_vectical_vector = sample_orthogonal_vector_determinisitic(difference)
    # alpha = 60 * np.pi / 180
    alpha = oracle_teacher_noise * np.pi / 180
    # alpha = 0 * np.pi / 180
    # alpha = np.random.randint(low=0, high=45) * np.pi / 180
    difference_with_noise = difference * np.cos(alpha) + difference_vectical_vector * np.sin(alpha)
    

    non_optimal_action_teacher = difference_with_noise + action
    # non_optimal_action_teacher = non_optimal_action_teacher + np.random.normal(loc=0.0, scale=np.linalg.norm(difference/1.0), size=action.shape)
    # non_optimal_action_teacher = non_optimal_action_teacher + np.random.normal(loc=0.0, scale=0.4, size=action.shape)
    # print("non_perfect_oracle_normed: ", non_optimal_action_teacher, " action agent: ", action, " action teacher: ", action_teacher)

    dim_a = action.shape[0]
    lower_limits = np.full(dim_a, -1)  # Lower limits for each dimension, replace with actual values
    upper_limits = np.full(dim_a, 1)   # Upper limits for each dimension, replace with actual values

    # Clip the actions to be within the limits
    clipped_actions = np.clip(non_optimal_action_teacher, lower_limits, upper_limits)

    # [ATTENTION!] we don't clip the action because the difference will be normalized and multiplied by e

    # clip the action if some dim of it is out of its limits
    # dim_a = action.shape[0]
    # lower_limits = np.full(dim_a, -1)  # Lower limits for each dimension, replace with actual values
    # upper_limits = np.full(dim_a, 1)   # Upper limits for each dimension, replace with actual values

    # Clip the actions to be within the limits
    # clipped_actions = np.clip(non_optimal_action_teacher, lower_limits, upper_limits)
    return clipped_actions


def oracle_gimme_feedback(action_teacher, action, h, config):
    theta = config['theta']
    feedback_threshold = config['feedback_threshold']
    oracle_teacher_only_one_dim = config['oracle_teacher_only_one_dim'] # used for the multi-line following task

    dim_a = len(action)
    if action_teacher is None:
        return [0] * dim_a
    action_max = np.max(abs(action_teacher))
    if action_max > 1:
        action_teacher =  action_teacher / action_max
    difference = action_teacher - action
    # print("difference: ", difference, " action_max: ", action_max, " action_teacher:", action_teacher)

    arg_max = np.argmax(abs(difference))  # try another simulated teacher, which only give feedback along one direction
    difference = np.array(difference)

    h = [0] * dim_a
    # if np.linalg.norm(difference) > 0.2:  
    if np.linalg.norm(difference) > feedback_threshold:

        h = [0] * dim_a
        if oracle_teacher_only_one_dim:
            # for i, name in enumerate(h):
            #     # [Attention! ] made for line-following task
            #     if abs(difference[i]) > 1 * theta and i == arg_max:
            #         h[i] = np.sign(difference[i])
            diff_abs = abs(difference)
            control_first_ball = diff_abs[0]**2 + diff_abs[1]**2 > diff_abs[2]**2 + diff_abs[3]**2
            # for i, name in enumerate(h):
            #     if abs(difference[i]) > 1 * theta and i == arg_max:
            #         h[i] = action_teacher[i]
            if control_first_ball:
                # if abs(difference[0]) > 1 * theta:
                #     h[0] =  np.sign(difference[0])
                # if abs(difference[1]) > 1 * theta:
                #     h[1] =  np.sign(difference[1])
                h[0:2] = difference[0:2]
                h = h / np.linalg.norm(h)
            else:
                # if abs(difference[2]) > 1 * theta:
                #     h[2] =  np.sign(difference[2])
                # if abs(difference[3]) > 1 * theta:
                #     h[3] =  np.sign(difference[3])
                h[2:] = difference[2:]
                h = h / np.linalg.norm(h)
        else:
            # h = difference 
            h = non_perfect_oracle_normed(action_teacher.copy(), action.copy(), config)  - action
            # print("np.linalg.norm(h): ", np.linalg.norm(h))
            h = h / np.linalg.norm(h)

    h_no_threshold = non_perfect_oracle_normed(action_teacher.copy(), action.copy(), config)  - action
    h_no_threshold = h_no_threshold / np.linalg.norm(h_no_threshold)


    return h, h_no_threshold #h: can be zero if the norm is below threshold. h_no_threshold: no threshold considered


def oracle_feedback_HGDagger(action_teacher, action, h, config):

    feedback_threshold = config['feedback_threshold']
    oracle_teacher_only_one_dim = config['oracle_teacher_only_one_dim'] # used for the multi-line following task


    if action_teacher is None:
        return None
    action_max = np.max(abs(action_teacher))
    if action_max > 1:
        action_teacher =  action_teacher / action_max
    difference = action_teacher - action
    #print("difference: ", difference)

    arg_max = np.argmax(abs(difference))  # try another simulated teacher, which only give feedback along one direction
    difference = np.array(difference)
    # h is a list of zero, with the same dimension as action
    dim_a = len(action)
    h = [0] * dim_a
    # h_no_threshold = h.copy()
    h_no_threshold = action_teacher.copy()
    logger.debug('difference:  %s', difference)
    if np.linalg.norm(difference) > feedback_threshold:  
        
        if oracle_teacher_only_one_dim:  
            
            # for multi-line follower task
            # ##  Attention!! 
            # # assume we can only control one dimension at a time, we select the one with the largest difference
            # diff_abs = abs(difference)
            # control_first_ball = diff_abs[0]**2 + diff_abs[1]**2 > diff_abs[2]**2 + diff_abs[3]**2
            # # for i, name in enumerate(h):
            # #     if abs(difference[i]) > 1 * theta and i == arg_max:
            # #         h[i] = action_teacher[i]
            # if control_first_ball:
            #     if np.linalg.norm(difference[0:2]) > feedback_threshold:
            #         h[0:2] = action_teacher[0:2]
            #         '''For HG-Dagger in multi-line following, if no feedback for another robot, keep the h to the same as the robot's action'''
            #         h[2:] = action[2:]
            # else:
            #     if np.linalg.norm(difference[2:]) > feedback_threshold:
            #         h[2:] = action_teacher[2:]
            #         h[0:2] = action[0:2]
            '''Robosuite task 6-dim'''
            # # h = action_teacher.copy()
            # position_diff = np.linalg.norm(difference[0:3])
            # rotation_diff = np.linalg.norm(difference[3:6])
            # if position_diff < rotation_diff:
            #     h[0:3] = action[0:3]
            #     print("h[0:3] = action[0:3]")
            # else:
            #     h[3:6] = action[3:6]
            #     print("h[3:6] = action[3:6]")
            # h[-1] = action[-1]
            h = action.copy()
            if difference.shape[0] == 7:
                position_diff = np.linalg.norm(difference[0:3])
                rotation_diff = np.linalg.norm(difference[3:6])
                if position_diff > rotation_diff:
                    h[0:3] = action_teacher[0:3]
                    # print("h[0:3] = action[0:3]")
                else:
                    h[3:6] = action_teacher[3:6]
                    # print("h[3:6] = action[3:6]")
                h[-1] = action_teacher[-1]
            elif difference.shape[0] == 14:
                r1_diff = np.linalg.norm(difference[0:7])
                r2_diff = np.linalg.norm(difference[7:14])
                if r1_diff > r2_diff:
                    h[0:7] = action_teacher[0:7]
                else:
                    h[7:14] = action_teacher[7:14]
                # print("partial feedback")

        else:
            h = non_perfect_oracle(action_teacher.copy(), action.copy(), config)

    h_no_threshold = non_perfect_oracle(action_teacher.copy(), action.copy(), config) 
    
    # print("h: ", h, " action: ", action, " action_teacher:", action_teacher)

    return h, h_no_threshold #h: can be zero if the norm is below threshold. h_no_threshold: no threshold considered

# caculate the feedback from the oracle teacher for policy-contrastive-intervention method
def oracle_feedback_intervention_diff(action_teacher, action, h, config):
    feedback_threshold = config['feedback_threshold']
    oracle_teacher_only_one_dim = config['oracle_teacher_only_one_dim'] # used for the multi-line following task

    if action_teacher is None:
        return None
    action_max = np.max(abs(action_teacher))
    if action_max > 1:
        action_teacher =  action_teacher / action_max
    difference = action_teacher - action
    # print("difference: ", difference)

    arg_max = np.argmax(abs(difference))  # try another simulated teacher, which only give feedback along one direction
    difference = np.array(difference)
    # h is a list of zero, with the same dimension as action
    dim_a = len(action)
    h = [0] * dim_a
    # h_no_threshold = h.copy()
    h_no_threshold = difference.copy()
    # print("feedback_threshold: ", feedback_threshold)
    if np.linalg.norm(difference) > feedback_threshold:  
    
        ##  Attention!! 
        # assume we can only control one dimension at a time, we select the one with the largest difference
        # for i, name in enumerate(h):
        #     if abs(difference[i]) > 1 * theta : # and i == arg_max:
        #         h[i] = difference[i]
        # h = difference
        
        if oracle_teacher_only_one_dim:
            '''simple line-following task'''
            # diff_abs = abs(difference)
            # control_first_ball = diff_abs[0]**2 + diff_abs[1]**2 > diff_abs[2]**2 + diff_abs[3]**2
            # # for i, name in enumerate(h):
            # #     if abs(difference[i]) > 1 * theta and i == arg_max:
            # #         h[i] = action_teacher[i]
            # if control_first_ball:
            #     # if abs(difference[0]) > 1 * theta:
            #     #     h[0] =  difference[0]
            #     # if abs(difference[1]) > 1 * theta:
            #     #     h[1] =  difference[1]
            #     h[0:2] = difference[0:2]
            # else:
            #     # if abs(difference[2]) > 1 * theta:
            #     #     h[2] =  difference[2]
            #     # if abs(difference[3]) > 1 * theta:
            #     #     h[3] =  difference[3]
            #     h[2:] = difference[2:]

            # [!] to do: 1. the magnitude of position error and rotation error
            # 2. first control rotation, then position?
            # 3. first 10 episodes do not use this?
            '''Robosuite task 6-dim'''
            if difference.shape[0] == 7:
                position_diff = np.linalg.norm(difference[0:3])
                rotation_diff = np.linalg.norm(difference[3:6])
                if position_diff > rotation_diff:
                    h[0:3] = difference[0:3]
                else:
                    h[3:6] = difference[3:6]
                h[-1] = difference[-1]
            elif difference.shape[0] == 14:
                r1_diff = np.linalg.norm(difference[0:7])
                r2_diff = np.linalg.norm(difference[7:14])
                if r1_diff > r2_diff:
                    h[0:7] = difference[0:7]
                else:
                    h[7:14] = difference[7:14]
                # print("partial feedback")

        else:
            # print("non_perfect_oracle")
            h = non_perfect_oracle(action_teacher.copy(), action.copy(), config)  - action
    
    h_no_threshold = non_perfect_oracle(action_teacher.copy(), action.copy(), config)  - action


    # print("h: ", h, " action: ", action, " action_teacher:", action_teacher)


    # note: h_no_threshold is only used in main-receding_horizon when Ta > 1. 
    # The idea is once the teacher provides corrections/interventions, it will continue to provide for the remaining n* Ta steps,
    # no matter whether h is below the threshold or not.
    return h, h_no_threshold #h: can be zero if the norm is below threshold. h_no_threshold: no threshold considered

