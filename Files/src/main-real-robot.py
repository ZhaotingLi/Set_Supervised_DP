import logging
import os
import time
import random
import datetime

import numpy as np

import hydra
from omegaconf import DictConfig

from agents.selector_policy import agent_selector
from env.env_selector_real_robot import env_selector_real_robot
from env.robotsuite.env_robosuite import extract_latest_obs_dict, axisangle_to_rot6d
from tools.buffer_trajectory import TrajectoryBuffer
from tools.oracle_feedback import (
    oracle_gimme_feedback,
    oracle_feedback_HGDagger,
    oracle_feedback_intervention_diff,
)
from tools.real_robot_helpers import (
    evaluation_saving_results_process,
    generate_random_numbers,
    is_env_done,
)

logger = logging.getLogger(__name__)


def resolve_config_path(project_root, path, field_name, required=False):
    if path is None:
        if required:
            raise ValueError(f"{field_name} must be set for this mode.")
        return None
    path = os.path.expanduser(os.path.expandvars(str(path)))
    if os.path.isabs(path):
        return path
    return os.path.join(project_root, path)

# ==============================================================================
#      TRAINING LOGIC: Online interactive imitation learning (action horizion > 1
#                      Used for collecting corrections. 
# ==============================================================================

def train_interactive_learning_repetition(env, policy_oracle, feedback_receiver, feedback_receiver_keyboard, i_repetition, config_general, config_agent, render_savefig_flag = True):  # used in delftblue
    traj_buffer = TrajectoryBuffer()  # used to save trajectory-level data for preference learning
    import rospy
    from env.franka.pose_transform_functions import get_euler_from_quaternion, get_quaternion_from_euler, euler_to_axis_angle
    """Run one repetition of training: init seeds, run episodes, eval & save."""
    SEED = 48 + i_repetition
    random.seed(SEED)
    np.random.seed(SEED)

    max_num_of_episodes = config_general['max_num_of_episodes']
    max_time_steps_episode = config_general['max_time_steps_episode']
    executed_human_correction = config_general['executed_human_correction']
    oracle_teacher = config_general['oracle_teacher']
    human_teacher = config_general['human_teacher']

    environment_name = config_general['environment'] # pendulum(PushT), metaworld, robosuite, obs_avoidance, cartpole, mountaincar
    task = config_general['task']

    save_policy = config_agent['save_policy']
    agent_algorithm = config_agent['algorithm']
    agent_type = config_agent['agent']

    # Pendulum environment seeding example
    if environment_name in ['PushT']:
        env.set_seed(SEED)
        env_seeds = generate_random_numbers(max_num_of_episodes)
        env_seeds_eval = generate_random_numbers(max_num_of_episodes * 10)

    # Initialize agent
    current_agent = agent_selector(agent_type, config_agent)
    # if agent_type != 'Diffusion' or agent_type != 'Set_Supervised_Diffusion':
    # if agent_type not in ['Diffusion', 'Set_Supervised_Diffusion', 'CLIC']:
    # # if agent_type not in ['Diffusion', 'Set_Supervised_Diffusion']:
    #     current_agent.createModels(init_neural_network())

    load_policy = config_agent.load_policy
    if load_policy:
        logger.info('LOAD MODELS')
        from hydra.utils import get_original_cwd
        project_root = get_original_cwd()
        current_agent.load_dir = resolve_config_path(
            project_root,
            current_agent.load_dir,
            "AGENT.load_dir",
            required=True,
        )
        logger.debug('load path:  %s', current_agent.load_dir)
        current_agent.load_model()
        #TODO load buffer

        # filename = current_agent.load_dir + 'buffer_data.pkl'
        # current_agent.buffer.load_from_file(filename)
        # # # for tactile sensor task, set sensor_readings to zero to see if sensor info helps 
        # # for i in range(len(current_agent.buffer.buffer)):
        # #     current_agent.buffer.buffer[i][0][:,-5:] = np.zeros(5)

        # print("length of buffer: ", current_agent.buffer.length())

    # print("self.load_policy_flag: ", self.load_policy_flag)
    # # also load the buffer
    # print("LOAD buffer from load_dir")
    # filename = self.load_dir + 'buffer_data.pkl'
    # self.buffer.load_from_file(filename)
    # print("length of buffer: ", self.buffer.length())

    # print('LOAD MODELS')
    # from hydra.utils import get_original_cwd
    # project_root = get_original_cwd()
    # # import pdb; pdb.set_trace()
    # full_path = os.path.join(project_root, self.load_dir)
    # self.load_dir = full_path
    # # load model now
    # # Check if the model file exists
    # self.load_model()
                    

    # Tracking
    t_total = 1
    episode_counter = 0
    cumm_feedback = 0
    eval_time_acc = 0
    repetition_done = False

    Ta = current_agent.horizon # length of executed actions
    Ta_executed = config_general['Ta_executed']
    Ta_i_teacher =0
    Ta_i = Ta
    action_agent_Ta = None

    # For saving
    ep_history, ts_history, time_history = [], [], []
    fb_history, success_history, error_history = [], [], []

    start_time = time.time()

    control_frequency = 10
    rate = rospy.Rate(control_frequency)

    # ===== EPISODES LOOP =====
    for i_episode in range(max_num_of_episodes):
        logger.info('i_episode:  %s', i_episode)
        ## i_episode % 2 == 1 is feedback mode
        # no_feedback_mode = i_episode % 2 == 0
        # record_preference_data = True
        no_feedback_mode = False


        observation, info = env.reset()

        # from script.visualize_obs_encoder_decoder_output import visualize_reconstruction_for_indices
        # visualize_reconstruction_for_indices([observation], [0], current_agent.obs_encoder)

        # wait for human to start the robot
        logger.info("press space key to start")
        while not feedback_receiver_keyboard.ask_for_done():
            env.hold_on_mode()


        episode_counter += 1
        h_counter = 0  # how many times feedback was given
        last_action = None
        h = None
        done_restart = False  # set by human feedback
        done = False

        current_agent.evaluation = False
        t = 0
        # for t in range(1, max_time_steps_episode + 1):

        obs_list = []
        action_positive_list = []
        action_negative_list = []
        intervention_signal = [] # boolean, indicates whether the teacher starts to give feedback
  
        while True:  # no max episode length, only reset from human feedback
            # print("agent.evaluation: ", current_agent.evaluation)
            t = t + 1
            env.render_mode = 'human'
            if render_savefig_flag: env.render()
            # Time used for logging
            elapsed = (time.time() - start_time - eval_time_acc)
            time_str = str(datetime.timedelta(seconds=elapsed))

            # obs_proc = process_observation(observation) # TODO remove this line, check whether CLIC breaks
            obs_proc = observation

            # key = 'robot0_eef_pos_vel'
            # print("obs_proc[key] before: ", obs_proc[key])
            # obs_proc[key] = np.clip((obs_proc[key] + 1.0) * 127.5, 0, 255).astype(np.uint8)  #TODO remove this line!!!!
            # print("obs_proc[key]: ", obs_proc[key])
            
            # Agent's action or teacher's correction
            if not executed_human_correction:
                action_agent_Ta = current_agent.action(obs_proc)
                # action = action_agent_Ta[0]
                Ta_i = 0
                action = action_agent_Ta
                action_agent_i = action[Ta_i, :]
            else:
                # If "execute_human_correction" is True
                # h = feedback_receiver.get_h() if human_teacher else h
                if h is not None and np.any(h):
                    # action_agent = current_agent.action(obs_proc)
                    action_agent_Ta = None
                    if agent_type in ['HG_DAgger','Implicit_BC','IWR','PVP','Diffusion', 'Set_Supervised_Diffusion']:
                        action_agent_i = h
                    else:
                        action_agent_i = last_action + np.matmul(current_agent.e, h)
                    
                    # action_agent_i = np.clip(action_agent_i, -1, 1)
                    # TODO for now we use teacher action at current state, instead of the positive action at last state
                    h_counter += 1
                    Ta_i_teacher = Ta_i_teacher + 1
                    logger.debug('(1) action:  %s  last_action:  %s', action_agent_i, last_action)
                else:
                    # action_agent = current_agent.action(obs_proc)
                    start = time.time()
                    if Ta_i >= Ta_executed -1 or Ta_i == 0:
                        action_agent_Ta = current_agent.action(obs_proc)
                        Ta_i = 0
                    end   = time.time()

                    action_agent_i = action_agent_Ta[Ta_i, :]

                    logger.debug('%s  Ta_i:  %s', f"Elapsed time: {end - start:.6f} seconds", Ta_i)
                    # action_agent_i = action_agent_Ta[0]
                    # action = action_agent_i
                    # print("action_agent: ", action)
            
            # Step environment
            observation, reward, done, _, info = env.step(action_agent_i)
            last_action = action_agent_i 

            if human_teacher:
                # Real-time user feedback (example)
                # h = feedback_receiver.get_h()
                if feedback_receiver.ask_for_done():
                    done = True

                h_raw = feedback_receiver.get_h()
                '''test adding noise to the robot'''
                logger.debug('(2) h_raw:  %s', h_raw)
                done_restart = feedback_receiver.ask_for_done()
                if np.any(h_raw):
                    if agent_algorithm in ["Policy_Contrastive_intervention", "Policy_Contrastive_sphere_intervention", "CLIC_EBM"]: # for policy contrastive intervention, we assume human teacher always demonstration
                        if action_agent_Ta is None:
                            action_agent_Ta = current_agent.action(obs_proc)  # key: the h is the difference between the action from the agent and the action from the human teacher!
                                                                    #         not its past action, which could be the action from the human teacher!!!
                            action_agent_i = action_agent_Ta[0]
                        if config_agent.use_abs_action:
                            h_raw = h_raw * env.scale + env.ee_pose[:2]
                            h_raw = env.normalize_abs_action(h_raw, max_list=env.action_max, min_list=env.action_min)
                        h_raw, h_raw_nothreshold = oracle_feedback_intervention_diff(h_raw.copy(), action_agent_i, h, config=config_general)
                        
                        # h_raw = h_raw - action_agent
                        current_agent.e = np.identity(current_agent.dim_a)
                        ## if np.linalg.norm(h_raw) < 0.1:
                        # if np.linalg.norm(h_raw[0:2]) < 0.05:  #TODO fix it for abs_action
                        #     h_raw = None
                        # else:
                        #     last_action = action_agent
                        # # last_action = action_agent # not tested
                    elif agent_type in ['HG_DAgger','Implicit_BC','IWR','PVP','Diffusion', 'Set_Supervised_Diffusion']:
                        if action_agent_Ta is None or Ta_i >= Ta_executed:
                            action_agent_Ta = current_agent.action(obs_proc) 
                            action_agent_i = action_agent_Ta[0]
                            Ta_i = 0
                        else:      
                            action_agent_i = action_agent_Ta[Ta_i, :]  # we compare the action in the previous step
                            logger.debug('take action Ta:  %s', Ta_i)
                        if config_agent.use_abs_action:
                            #kuka
                            # h_raw = h_raw * env.scale + env.ee_pose[:2]
                            # h_raw = env.normalize_abs_action(h_raw, max_list=env.action_max, min_list=env.action_min)
                            position_goal = h_raw[:3] * env.scale +  np.array(env.robot.curr_pos).copy() 
                            quat_goal=np.quaternion(env.robot.curr_ori[0],env.robot.curr_ori[1],env.robot.curr_ori[2],env.robot.curr_ori[3])
                            q_delta_array=get_quaternion_from_euler(h_raw[3], h_raw[4],h_raw[5])
                            q_delta=np.quaternion(q_delta_array[0],q_delta_array[1],q_delta_array[2],q_delta_array[3]) 
                            quat_goal=q_delta*quat_goal
                            euler_goal = get_euler_from_quaternion(quat_goal)
                            axis_angle_goal = euler_to_axis_angle(euler_goal)

                            rota6d_goal = axisangle_to_rot6d(axis_angle_goal)

                            gripper_goal = h_raw[-1] * env.gripper_scale + env.goal_gripper_command
                            action_human = np.concatenate([position_goal, rota6d_goal, np.array([gripper_goal])], axis=0)

                            h_raw = env.normalize_abs_action(action_human, max_list=env.action_max, min_list=env.action_min)

                        logger.debug('(2.0.0) h_raw:  %s', h_raw)

                        if np.any(h_raw):  # TODO ,test next time
                            last_action = action_agent_i
                            
                h = h_raw
                logger.debug('(2.1) h:  %s', h)

            Ta_i = Ta_i + 1  # step Ta after checking whether to give feedback
            if not bool(np.any(h_raw)): # no feedback
                Ta_i = 0  # reset to query the robot policy

                obs_list = []
                action_positive_list = []
                action_negative_list = []
                intervention_signal = [] # boolean, indicates whether the teacher starts to give feedback
        
            else:
                obs_list.append(obs_proc)
                action_positive_list.append(h)
                intervention_signal.append(True)
                action_negative_list.append(action_agent_i)


            '''save data into buffer'''
            obs_proc_Ta = None
            negative_action_Ta = None
            h_Ta = None

            teacher_outputs_enough_data = len(obs_list) >= Ta
            if teacher_outputs_enough_data:
            # only start to record when teacher gives feedback
                # data_id = len(obs_list) - Ta
                data_id = 1 
                '''
                o1, o2, o3, o4, ...
                a1, a2, a3, a4, ...
                record [o1, o2] and [a1 : a1+horizon]   instead of [a2: a2+horizon]!         
                '''
                obs_proc_Ta = obs_list[data_id]
                # intervention_signal_Ta = any(intervention_signal[data_id-1 : data_id-1 + Ta]) 
                intervention_signal_Ta = all(intervention_signal[data_id-1 : data_id-1 + Ta])
                logger.debug('intervention_signal:  %s', intervention_signal)
                
                if intervention_signal_Ta:
                    h_counter += 1
                    negative_action_Ta = action_negative_list[data_id-1 : data_id-1 + Ta]   # length == Ta
                    negative_action_Ta = np.stack([np.asarray(a, dtype=np.float32) for a in negative_action_Ta], axis=0) 

                    positive_action_Ta = action_positive_list[data_id-1 : data_id-1 + Ta]   # length == Ta
                    h_Ta = np.stack([np.asarray(a, dtype=np.float32) for a in positive_action_Ta], axis=0) 

                    logger.info('-----------Add one feedback, len(obs_list):  %s', len(obs_list))
                    logger.debug('len(intervention_signal):  %s  len(action_negative_list):  %s  len +: %s', len(intervention_signal), len(action_negative_list), len(action_positive_list))
                
                obs_list.pop(0)
                intervention_signal.pop(0)
                action_negative_list.pop(0)
                action_positive_list.pop(0)

            # Collect data if h is none; Also train the policy 
            current_agent.collect_data_and_train(
                last_action=negative_action_Ta,
                h=h_Ta,
                obs_proc=obs_proc_Ta,
                next_obs=observation, # this line is wrong, TODO 
                t=t_total,
                done=done,
                agent_algorithm=agent_algorithm,
                agent_type=agent_type,
                i_episode=i_episode
            )

            env_done_fake, success_ep = is_env_done(info)
            done = done or env_done_fake or done_restart
            
            t_total += 1
            rate.sleep()

            
            traj_buffer.add_transition(
                # obs={'low_dim':obs_proc},      # used for low-dim state
                obs = extract_latest_obs_dict(obs_proc), # used for image observation
                teacher_action=h if np.any(h) else np.zeros_like(last_action),
                done=done,
                timestep=t,
                no_robot_action = bool(np.any(h)),
                no_teacher_action = not bool(np.any(h)),
                robot_action=action_agent_i if action_agent_i is not None else last_action,                   # the feedback signal (if any)
                episode_id= i_episode,
                if_success = False,
            )

            if done:
                base_dir = config_agent['base_dir']
                current_agent.save_dir = base_dir
                current_agent.load_dir = current_agent.saved_dir
                if not os.path.exists(current_agent.saved_dir):
                    os.makedirs(current_agent.saved_dir)  
                
                traj_buffer.finish_trajectory()
                traj_buffer.save_to_file("trajectory_buffer_"+str(i_repetition))
                cumm_feedback += h_counter
                
                # Optionally save policy or buffer
                if save_policy:
                    current_agent.save_model()   

                with open(os.path.join(current_agent.saved_dir, 'data.txt'), 'w') as f:
                    f.write(f"Steps: {t_total}\nEpisode: {i_episode}\nTime: {time_str}\nFeedbacks: {cumm_feedback}\n")
                break
        

        # Evaluate agent periodically
        evalutation_frequency = 10

        if i_episode % evalutation_frequency == 0 or repetition_done:
            skip_evaluation = None
            logger.warning("press 's' key to skip the whole evaluation, 'q' for not skip")
            while skip_evaluation is None:
                skip_evaluation = feedback_receiver_keyboard.ask_whether_skip_evaluation()
            logger.info('skip_evaluation:  %s', skip_evaluation)
            logger.info("press space key to start")
            while not feedback_receiver_keyboard.ask_for_done():
                env.hold_on_mode()

            '''Start evaluation here'''
            if not skip_evaluation:
                history = [ep_history, ts_history, time_history, fb_history, success_history, error_history]
                success_rate, mean_error = None, None
                data =    [episode_counter, t_total, time_str, cumm_feedback, success_rate, mean_error]

                eval_time_acc, history_new = evaluation_saving_results_process(eval_agent=current_agent, eval_env= env, feedback_receiver=feedback_receiver,feedback_receiver_keyboard=feedback_receiver_keyboard, i_episode=i_episode, i_repetition=i_repetition,
                                                    max_steps=max_time_steps_episode, render_savefig_flag = render_savefig_flag, 
                                                    history=history, data=data, eval_time_acc = eval_time_acc, SEED_id=SEED, config_general=config_general, config_agent=config_agent)
                ep_history, ts_history, time_history, fb_history, success_history, error_history = history_new

        if repetition_done:
            break

    logger.info(f"== Finished Repetition {i_repetition} with {episode_counter} episodes ==")


# ==============================================================================
#      Collecting Demonstrations by teleoperating 
# ==============================================================================
def collect_dataset_repetition(env, policy_oracle, feedback_receiver, feedback_receiver_keyboard, i_repetition, config_general, config_agent, render_savefig_flag = True):  # used in delftblue
    traj_buffer = TrajectoryBuffer()  # used to save trajectory-level data for preference learning
    import rospy
    from env.franka.pose_transform_functions import get_euler_from_quaternion, get_quaternion_from_euler, euler_to_axis_angle

    """Run one repetition of training: init seeds, run episodes, eval & save."""
    SEED = 48 + i_repetition
    random.seed(SEED)
    np.random.seed(SEED)

    max_num_of_episodes = config_general['max_num_of_episodes']
    max_time_steps_episode = config_general['max_time_steps_episode']
    executed_human_correction = config_general['executed_human_correction']
    oracle_teacher = config_general['oracle_teacher']
    human_teacher = config_general['human_teacher']

    environment_name = config_general['environment'] # pendulum(PushT), metaworld, robosuite, obs_avoidance, cartpole, mountaincar
    task = config_general['task']

    save_policy = config_agent['save_policy']
    agent_algorithm = config_agent['algorithm']
    agent_type = config_agent['agent']


    # Initialize agent
    current_agent = agent_selector(agent_type, config_agent)
    # if agent_type != 'Diffusion' or agent_type != 'Set_Supervised_Diffusion':
    
    # Tracking
    t_total = 1
    episode_counter = 0
    cumm_feedback = 0
    eval_time_acc = 0
    repetition_done = False
    

    # For saving
    ep_history, ts_history, time_history = [], [], []
    fb_history, success_history, error_history = [], [], []

    start_time = time.time()

    control_frequency = 10
    rate = rospy.Rate(control_frequency)

    # TODO define saving freq for the traj data

    # ===== EPISODES LOOP =====
    for i_episode in range(max_num_of_episodes):
        logger.info('i_episode:  %s', i_episode)
        ## i_episode % 2 == 1 is feedback mode
        # no_feedback_mode = i_episode % 2 == 0
        # record_preference_data = True
        no_feedback_mode = False


        observation, info = env.reset()

        # wait for human to start the robot
        logger.info("press space key to start")
        while not feedback_receiver_keyboard.ask_for_done():
            env.hold_on_mode()


        episode_counter += 1
        h_counter = 0  # how many times feedback was given
        last_action = None
        h = None
        done = False
        done_restart = False

        current_agent.evaluation = False
        t = 0
        # for t in range(1, max_time_steps_episode + 1):
        while True:  # no max episode length, only reset from human feedback
            # print("agent.evaluation: ", current_agent.evaluation)
            t = t + 1
            env.render_mode = 'human'
            if render_savefig_flag: env.render()
            # Time used for logging
            elapsed = (time.time() - start_time - eval_time_acc)
            time_str = str(datetime.timedelta(seconds=elapsed))

            # obs_proc = process_observation(observation) # TODO remove this line, check whether CLIC breaks
            obs_proc = observation
            
            # Agent's action or teacher's correction
            h = feedback_receiver.get_h()
            # if feedback_receiver.ask_for_done():   # TODO this two lines makes done always false, very strange. should find out why
            #     done = True
            done_restart = feedback_receiver.ask_for_done()
            if np.any(h):
                if config_agent.use_abs_action:
                    # KUka
                    # action = h * env.scale + env.ee_pose[:2]
                    # print("action before normalization: ", action)
                    # action = env.normalize_abs_action(action, max_list=env.action_max, min_list=env.action_min)

                    # Franka
                    position_goal = h[:3] * env.scale +  np.array(env.robot.curr_pos).copy() 
                    quat_goal=np.quaternion(env.robot.curr_ori[0],env.robot.curr_ori[1],env.robot.curr_ori[2],env.robot.curr_ori[3])
                    q_delta_array=get_quaternion_from_euler(h[3], h[4], h[5])
                    q_delta=np.quaternion(q_delta_array[0],q_delta_array[1],q_delta_array[2],q_delta_array[3]) 
                    quat_goal=q_delta*quat_goal
                    euler_goal = get_euler_from_quaternion(quat_goal)
                    axis_angle_goal = euler_to_axis_angle(euler_goal)  # transform euler angle to rot6d as the latter has value range between [-1, 1]
                    rota6d_goal = axisangle_to_rot6d(axis_angle_goal)

                    gripper_goal = h[-1] * env.gripper_scale + env.goal_gripper_command
                    action = np.concatenate([position_goal, rota6d_goal, np.array([gripper_goal])], axis=0)

                    action = env.normalize_abs_action(action, max_list=env.action_max, min_list=env.action_min)
                else:
                    action = h
            else:
                if config_agent.use_abs_action:
                    # KUka
                    # action = env.ee_pose[:2]
                    # print("action before normalization: ", action)
                    
                    # Franka
                    position_goal = np.array(env.robot.curr_pos).copy() 
                    logger.debug('position_goal:  %s', position_goal)
                    quat_goal=np.quaternion(env.robot.curr_ori[0],env.robot.curr_ori[1],env.robot.curr_ori[2],env.robot.curr_ori[3])
                    euler_goal = get_euler_from_quaternion(quat_goal)
                    axis_angle_goal = euler_to_axis_angle(euler_goal)
                    rota6d_goal = axisangle_to_rot6d(axis_angle_goal)
                    gripper_goal = env.goal_gripper_command
                    action = np.concatenate([position_goal, rota6d_goal, np.array([gripper_goal])], axis=0)
                    logger.debug('action before normalization:  %s', action)
                    action = env.normalize_abs_action(action, max_list=env.action_max, min_list=env.action_min)
                else:
                    action = np.zeros(current_agent.dim_a)
            logger.debug('h:  %s  done:  %s  action:  %s', h, done, action)
            # Step environment
            observation, reward, done, _, info = env.step(action)

            
            env_done_fake, success_ep = is_env_done(info)
            done = done or env_done_fake or done_restart 

            logger.debug('h:  %s  done:  %s', h, done)
            last_action = action 
            
            # if receive_feedback_phase:
            #     teacher_action_to_buffer = teacher_action_agent_i
            #     robot_action_to_buffer = action_agent_i
            #     no_robot_action = False
            #     no_teacher_action = False
            # else:
            #     robot_action_to_buffer = action_agent_i
            #     teacher_action_to_buffer = np.zeros_like(robot_action_to_buffer)
            #     no_robot_action = False
            #     no_teacher_action = True
            
            # save to trajectory buffer
            
            if np.any(h):  # donot record the data if the human doesn't take any actions
                teacher_action_to_buffer = action
                traj_buffer.add_transition(
                    obs=extract_latest_obs_dict(obs_proc),      # or processed observation if you prefer
                    teacher_action=teacher_action_to_buffer,
                    done=done,
                    timestep=t,
                    no_robot_action = True,
                    no_teacher_action = False,
                    robot_action=np.zeros_like(teacher_action_to_buffer),                   # the feedback signal (if any)
                    episode_id= i_episode,
                    if_success= False,
                )

            if done:
                traj_buffer.finish_trajectory()
                traj_buffer.save_to_file("trajectory_buffer_"+str(i_repetition))

            if done:
                logger.info("leaave the while")
                base_dir = config_agent['base_dir']
                current_agent.saved_dir = os.path.join(base_dir, config_agent['experiment_id'], f"repetition_{i_repetition:03d}/") 
                current_agent.load_dir = current_agent.saved_dir
                if not os.path.exists(current_agent.saved_dir):
                    os.makedirs(current_agent.saved_dir)  
                cumm_feedback += h_counter
                # Optionally save policy or buffer
            
            if done:
                break
            
            rate.sleep()


        if repetition_done:
            break

    logger.info(f"== Finished Repetition {i_repetition} with {episode_counter} episodes ==")

# ==============================================================================
#      Replaying Demonstrations/Correction dataset
# ==============================================================================
def Replay_collected_dataset(env, policy_oracle, feedback_receiver, feedback_receiver_keyboard, i_repetition, config_general, config_agent, render_savefig_flag = True):  # used in delftblue
    traj_buffer = TrajectoryBuffer()  # used to save trajectory-level data for preference learning
    import rospy
    """Run one repetition of training: init seeds, run episodes, eval & save."""
    SEED = 48 + i_repetition
    random.seed(SEED)
    np.random.seed(SEED)

    max_num_of_episodes = config_general['max_num_of_episodes']
    max_time_steps_episode = config_general['max_time_steps_episode']
    executed_human_correction = config_general['executed_human_correction']
    oracle_teacher = config_general['oracle_teacher']
    human_teacher = config_general['human_teacher']

    environment_name = config_general['environment'] # pendulum(PushT), metaworld, robosuite, obs_avoidance, cartpole, mountaincar
    task = config_general['task']

    save_policy = config_agent['save_policy']
    agent_algorithm = config_agent['algorithm']
    agent_type = config_agent['agent']


    logger.debug('LOAD Actions')
    from hydra.utils import get_original_cwd
    project_root = get_original_cwd()
    # import pdb; pdb.set_trace()
    action_path = 'outputs_docker/teacher_action.pkl'
    full_path = os.path.join(project_root, action_path)
    logger.debug('load path:  %s', full_path)
    import pickle
    with open(full_path, 'rb') as f:
        loaded_teacher_action = pickle.load(f)
    logger.debug('loaded_teacher_action:  %s', loaded_teacher_action)
    # Initialize agent
    current_agent = agent_selector(agent_type, config_agent)

    # Tracking
    t_total = 1
    episode_counter = 0
    repetition_done = False
    

    control_frequency = 20
    rate = rospy.Rate(control_frequency)

    # ===== EPISODES LOOP =====
    for i_episode in range(max_num_of_episodes):
        logger.info('i_episode:  %s', i_episode)

        observation, info = env.reset()

        # wait for human to start the robot
        logger.info("press space key to start")
        while not feedback_receiver_keyboard.ask_for_done():
            env.hold_on_mode()

        episode_counter += 1
        done = False
        done_restart = False

        current_agent.evaluation = False
        t = 0

        while True:  # no max episode length, only reset from human feedback
            t = t + 1
   
            done_restart = feedback_receiver.ask_for_done()

            action = loaded_teacher_action[t] if t<len(loaded_teacher_action) else loaded_teacher_action[-1]
            # action = env.normalize_abs_action(action, max_list=env.action_max, min_list=env.action_min)
            # Step environment
            observation, reward, done, _, info = env.step(action)
            
            env_done_fake, success_ep = is_env_done(info)
            done = done or env_done_fake or done_restart 

            if done:
                break
            
            rate.sleep()

        if repetition_done:
            break

    logger.info(f"== Finished Repetition {i_repetition} with {episode_counter} episodes ==")


# ==============================================================================
#      TRAINING LOGIC: Offline learning
# ==============================================================================
def train_offline_repetition(i_repetition, config_general, config_agent, render_savefig_flag = False):  # used in delftblue
    """Run one repetition of training: init seeds, run episodes, eval & save."""
    SEED = 48 + i_repetition
    random.seed(SEED)
    np.random.seed(SEED)

    max_num_of_episodes = config_general['max_num_of_episodes']
    max_time_steps_episode = config_general['max_time_steps_episode']

    environment_name = config_general['environment'] # pendulum(PushT), metaworld, robosuite, obs_avoidance, cartpole, mountaincar

    save_policy = config_agent['save_policy']
    agent_algorithm = config_agent['algorithm']
    agent_type = config_agent['agent']


    # Initialize agent
    current_agent = agent_selector(agent_type, config_agent)

    logger.info("LOAD buffer")
    
    from hydra.utils import get_original_cwd
    project_root = get_original_cwd()
    
    def _to_abs(path: str) -> str:
        if path is None:
            raise ValueError("AGENT.buffer_dataset_path must be set before loading an offline buffer.")
        path = os.path.expanduser(os.path.expandvars(str(path)))
        if os.path.isabs(path):
            return path
        return os.path.join(project_root, path)
    
    buffer_path = _to_abs(config_agent.buffer_dataset_path)

    '''Method 1 of Loading buffer: from traj_dataset'''
    # current_agent.buffer.ingest_trajectory_hdf5(traj_filename =buffer_path )
    
    '''Method 2 of Loading buffer: from buffer.h5 (generated files by Method1)'''

    intervention_replay_buffer_type = getattr(
        current_agent,
        "intervention_replay_buffer_type",
        None,
    )
    if intervention_replay_buffer_type is None:
        if getattr(current_agent, "use_hdf5_dataset", False):
            intervention_replay_buffer_type = "hdf5_obs_action_buffer"
        elif getattr(current_agent, "use_traj_ref_buffer", False):
            intervention_replay_buffer_type = "traj_ref_buffer"
        else:
            intervention_replay_buffer_type = "pickle_obs_action_buffer"

    if intervention_replay_buffer_type == "hdf5_obs_action_buffer":
        current_agent.buffer.load_from_file(buffer_path, read_only=True)
    elif intervention_replay_buffer_type == "traj_ref_buffer":
        current_agent.buffer.ingest_trajectory_hdf5_to_Intervention_buffer_Ta(
            traj_filename=buffer_path,
            show_progress=True,
        )
    elif intervention_replay_buffer_type == "pickle_obs_action_buffer":
        current_agent.buffer.load_from_h5_buffer_file(buffer_path, action_horizon = config_agent.Ta)
    else:
        raise ValueError(
            "Unsupported intervention_replay_buffer_type="
            f"{intervention_replay_buffer_type!r}."
        )

    '''Method 3 of loading buffer: load pkl (if have)'''
    # from hydra.utils import get_original_cwd
    # project_root = get_original_cwd()
    # # import pdb; pdb.set_trace()
    # full_path = os.path.join(project_root, current_agent.load_dir)
    # print("load path: ", full_path)
    # filename = current_agent.load_dir + 'buffer_data.pkl'
    # current_agent.buffer.load_from_file(filename)

    logger.info('length of buffer:  %s', current_agent.buffer.length())

    # Tracking
    t_total = 1
    episode_counter = 0
    cumm_feedback = 0
    eval_time_acc = 0
    repetition_done = False
    obs_proc = None
    time_str = 0

    start_time = time.time()

    # ===== EPISODES LOOP =====
    for i_episode in range(max_num_of_episodes):
        logger.info('i_episode:  %s', i_episode)


        episode_counter += 1
        h_counter = 0  # how many times feedback was given
        last_action = None
        h = None
        done = True

        current_agent.evaluation = False

        
        # Train agent
        current_agent.collect_data_and_train(
            last_action= None, 
            h=None,
            obs_proc=None,
            next_obs=None, # this line is wrong, TODO 
            t=t_total,
            done=True,
            agent_algorithm=agent_algorithm,
            agent_type=agent_type,
            i_episode=i_episode
        )
        
        base_dir = config_agent['base_dir']
        current_agent.saved_dir = os.path.join(base_dir, config_agent['experiment_id'], f"repetition_{i_repetition:03d}/") 
        current_agent.load_dir = current_agent.saved_dir
        if not os.path.exists(current_agent.saved_dir):
            os.makedirs(current_agent.saved_dir)  

        if save_policy:
            current_agent.save_model()   

        if repetition_done:
            break

    logger.info(f"== Finished Repetition {i_repetition} with {episode_counter} episodes ==")


# ==============================================================================
#      Evaluation on real-robot
# ==============================================================================
def evalaution_without_training(env, policy_oracle, feedback_receiver, feedback_receiver_keyboard, i_repetition, config_general, config_agent):
    """Run one repetition of training using offline data."""

    SEED = 100001 + i_repetition
    random.seed(SEED)
    np.random.seed(SEED)
    agent_algorithm = config_agent['algorithm']
    agent_type = config_agent['agent']
    max_num_of_episodes = config_general['max_num_of_episodes']
    max_time_steps_episode = config_general['max_time_steps_episode']

    # Initialize agent
    eval_agent = agent_selector(agent_type, config_agent)  

    logger.info('LOAD MODELS')
    from hydra.utils import get_original_cwd
    project_root = get_original_cwd()
    eval_agent.load_dir = resolve_config_path(
        project_root,
        eval_agent.load_dir,
        "AGENT.load_dir",
        required=True,
    )
    logger.debug('load path:  %s', eval_agent.load_dir)
    eval_agent.load_model()

    # Tracking
    t_total = 1
    episode_counter = 0
    cumm_feedback = 0
    eval_time_acc = 0
    time_str = 0
    repetition_done = False

    # For saving
    ep_history, ts_history, time_history = [], [], []
    fb_history, success_history, error_history = [], [], []

    # ===== EPISODES LOOP =====
    for i_episode in range(max_num_of_episodes):
        SEED = 100006 + i_episode

        random.seed(SEED)
        np.random.seed(SEED)
        logger.info('i_episode:  %s', i_episode)
        episode_counter += 1
    
        history = [ep_history, ts_history, time_history, fb_history, success_history, error_history]
        success_rate, mean_error = None, None
        data =    [episode_counter, t_total, time_str, cumm_feedback, success_rate, mean_error]


        eval_time_acc, history_new = evaluation_saving_results_process(eval_agent=eval_agent, eval_env= env, feedback_receiver=feedback_receiver,feedback_receiver_keyboard=feedback_receiver_keyboard, i_episode=i_episode, i_repetition=i_repetition,
                                                    max_steps=max_time_steps_episode, render_savefig_flag = False, 
                                                    history=history, data=data, eval_time_acc = eval_time_acc, SEED_id=SEED, config_general=config_general, config_agent=config_agent)

        ep_history, ts_history, time_history, fb_history, success_history, error_history = history_new

        if repetition_done:
            break

    logger.info(f"== Finished Repetition {i_repetition} with {episode_counter} episodes ==")



# ==============================================================================
#      Collecting self-play dataset for training the image auto-encoder. 
# ==============================================================================
def collect_self_play_dataset(env, policy_oracle, feedback_receiver, feedback_receiver_keyboard, i_repetition, config_general, config_agent, render_savefig_flag = True):  # used in delftblue
    traj_buffer = TrajectoryBuffer()  # used to save trajectory-level data for preference learning
    import rospy
    from env.franka.pose_transform_functions import get_euler_from_quaternion, get_quaternion_from_euler, euler_to_axis_angle
    """Run one repetition of training: init seeds, run episodes, eval & save."""
    SEED = 48 + i_repetition
    random.seed(SEED)
    np.random.seed(SEED)

    max_num_of_episodes = config_general['max_num_of_episodes']

    # Tracking
    t_total = 1
    episode_counter = 0
    cumm_feedback = 0
    eval_time_acc = 0
    repetition_done = False


    start_time = time.time()

    control_frequency = 10
    rate = rospy.Rate(control_frequency)

    # ===== EPISODES LOOP =====
    for i_episode in range(max_num_of_episodes):
        logger.info('i_episode:  %s', i_episode)

        observation, info = env.reset()

        # wait for human to start the robot
        logger.info("press space key to start")
        while not feedback_receiver_keyboard.ask_for_done():
            env.hold_on_mode()

        episode_counter += 1
        h = None
        done_restart = False  # set by human feedback
        done = False

        t = 0

        while True:  # no max episode length, only reset from human feedback
            # print("agent.evaluation: ", current_agent.evaluation)
            t = t + 1
            env.render_mode = 'human'
            if render_savefig_flag: env.render()
            # Time used for logging
            elapsed = (time.time() - start_time - eval_time_acc)
            time_str = str(datetime.timedelta(seconds=elapsed))

            obs_proc = observation
            
            # Agent's action or teacher's correction
            h = feedback_receiver.get_h()
            done_restart = feedback_receiver.ask_for_done()

            if np.any(h):
                if config_agent.use_abs_action:
                    position_goal = h[:3] * env.scale +  np.array(env.robot.curr_pos).copy() 
                    quat_goal=np.quaternion(env.robot.curr_ori[0],env.robot.curr_ori[1],env.robot.curr_ori[2],env.robot.curr_ori[3])
                    q_delta_array=get_quaternion_from_euler(h[3], h[4], h[5])
                    q_delta=np.quaternion(q_delta_array[0],q_delta_array[1],q_delta_array[2],q_delta_array[3]) 
                    quat_goal=q_delta*quat_goal
                    euler_goal = get_euler_from_quaternion(quat_goal)
                    axis_angle_goal = euler_to_axis_angle(euler_goal)  # transform euler angle to rot6d as the latter has value range between [-1, 1]
                    rota6d_goal = axisangle_to_rot6d(axis_angle_goal)

                    gripper_goal = h[-1] * env.gripper_scale + env.goal_gripper_command
                    action = np.concatenate([position_goal, rota6d_goal, np.array([gripper_goal])], axis=0)

                    action = env.normalize_abs_action(action, max_list=env.action_max, min_list=env.action_min)
                else:
                    action = h
            else:
                if config_agent.use_abs_action:
                    position_goal = np.array(env.robot.curr_pos).copy() 
                    logger.debug('position_goal:  %s', position_goal)
                    quat_goal=np.quaternion(env.robot.curr_ori[0],env.robot.curr_ori[1],env.robot.curr_ori[2],env.robot.curr_ori[3])
                    euler_goal = get_euler_from_quaternion(quat_goal)
                    axis_angle_goal = euler_to_axis_angle(euler_goal)
                    rota6d_goal = axisangle_to_rot6d(axis_angle_goal)
                    gripper_goal = env.goal_gripper_command
                    action = np.concatenate([position_goal, rota6d_goal, np.array([gripper_goal])], axis=0)
                    logger.debug('action before normalization:  %s', action)
                    action = env.normalize_abs_action(action, max_list=env.action_max, min_list=env.action_min)
                else:
                    action = np.zeros(config_agent.dim_a)
            logger.debug('h:  %s  done:  %s  action:  %s', h, done, action)
            # Step environment
            observation, reward, done, _, info = env.step(action)

            
            env_done_fake, success_ep = is_env_done(info)
            done = done or env_done_fake or done_restart 

            logger.debug('h:  %s  done:  %s', h, done)
            
            record_frame = feedback_receiver_keyboard.ask_for_done()
            if record_frame:  # donot record the data if the human doesn't take any actions
                teacher_action_to_buffer = action
                traj_buffer.add_transition(
                    obs=extract_latest_obs_dict(obs_proc),      # or processed observation if you prefer
                    teacher_action=teacher_action_to_buffer,
                    done=done,
                    timestep=t,
                    no_robot_action = False,
                    no_teacher_action = True,
                    robot_action=np.zeros_like(teacher_action_to_buffer),                   # the feedback signal (if any)
                    episode_id= i_episode,
                    if_success= False,
                )

            if done:
                traj_buffer.finish_trajectory()
                traj_buffer.save_to_file("trajectory_buffer_self_play"+str(i_repetition))
 
            if done:
                break
            
            rate.sleep()

        if repetition_done:
            break

    logger.info(f"== Finished Repetition {i_repetition} with {episode_counter} episodes ==")


# run by using: python main-v3-cleaned_new_config.py --config-name =train_Set_Supervised_Diffusion_image_Ta8
@hydra.main(config_path="config_real", config_name="train_Set_Supervised_Diffusion_image_Ta8")
def main(cfg: DictConfig):
    config_general  = cfg.GENERAL
    config_agent    = cfg.AGENT
    config_feedback = cfg.FEEDBACK
    config_task = cfg.task

    """Main entry point: run multiple training repetitions."""
    logger.info("Starting training...\n")
    for rep_idx in range(config_general['number_of_repetitions']):

        if config_agent.get('offline_data_collection', False):
            import rospy
            env, policy_oracle, feedback_receiver, feedback_receiver_keyboard = env_selector_real_robot(config_general, config_feedback, config_task)
            collect_dataset_repetition(env, policy_oracle, feedback_receiver,feedback_receiver_keyboard, rep_idx, config_general, config_agent)
            # collect_self_play_dataset(env, policy_oracle, feedback_receiver,feedback_receiver_keyboard, rep_idx, config_general, config_agent)
            # Replay_collected_dataset(env, policy_oracle, feedback_receiver,feedback_receiver_keyboard, rep_idx, config_general, config_agent)
        elif config_agent['evaluate']:
            import rospy
            env, policy_oracle, feedback_receiver, feedback_receiver_keyboard = env_selector_real_robot(config_general, config_feedback, config_task)
            evalaution_without_training(env, policy_oracle, feedback_receiver, feedback_receiver_keyboard, rep_idx, config_general, config_agent)
        elif config_agent['offline_training']:
            train_offline_repetition( rep_idx, config_general, config_agent)   # offline training with dataset
        else:
            import rospy
            env, policy_oracle, feedback_receiver, feedback_receiver_keyboard = env_selector_real_robot(config_general, config_feedback, config_task)
            train_interactive_learning_repetition(env, policy_oracle, feedback_receiver,feedback_receiver_keyboard, rep_idx, config_general, config_agent) 
           
  
if __name__ == "__main__":
    main()
