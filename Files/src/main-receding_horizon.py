import logging
import os
import time
import random
import datetime
import numpy as np

import hydra
from omegaconf import DictConfig

from agents.selector_policy import agent_selector
from env.env_selector import env_selector
from tools.buffer_trajectory import TrajectoryBuffer
from tools.feedback_window_buffer import FeedbackWindowBuffer
from tools.oracle_feedback import (
    oracle_gimme_feedback,
    oracle_feedback_HGDagger,
    oracle_feedback_intervention_diff,
)
from tools.receding_horizon_helpers import (
    evaluation_saving_results_process,
    generate_random_numbers,
    get_teacher_action,
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


def test_teacher_policy(env, policy_oracle, feedback_receiver, i_repetition, config_general, config_agent):
    """Test the teacher policy."""
    evaluations_per_training_ = config_general['evaluations_per_training']
    environment_name = config_general['environment']
    max_num_of_episodes = config_general['max_num_of_episodes']
    max_time_steps_episode = config_general['max_time_steps_episode']
    episode_counter = 0

    # ===== EPISODES LOOP =====
    for i_episode in range(max_num_of_episodes):
        SEED = 100001 + i_episode * evaluations_per_training_ 
        random.seed(SEED)
        np.random.seed(SEED)

        if environment_name in ['PushT']:
            env.seed(SEED)
            logger.debug('seed:  %s', SEED)

        observation, info = env.reset()

        episode_counter += 1
        action_agent = np.zeros(config_agent.dim_a)
        for t in range(1, max_time_steps_episode + 1):
            env.render_mode = 'human'
            env.render()

            teacher_action = get_teacher_action(environment_name, observation, action_agent=action_agent,  env=env, policy_oracle=policy_oracle)
            # test the noise
            action_agent = teacher_action #+ np.random.normal(loc=0.0, scale=0.025, size=teacher_action.shape)

            # Step environment
            observation, reward, done, _, info = env.step(action_agent)

            env_done_fake, success_ep = is_env_done(info)
            done = done or env_done_fake or t == max_time_steps_episode
            if done and environment_name in ['robosuite']: 
                policy_oracle.reset() # reset the state machine

            if done:
                break

# ==========================
# TRAINING LOGIC: Online interactive imitation learning
#       In each episode, the robot policy performs a rollout and receives corrective feedback from an oracle teacher.
#       These corrections are aggregated into the dataset.
#       Training is performed both during the episode and again after the episode ends.
# ==========================
def train_single_repetition(env, policy_oracle, feedback_receiver, i_repetition, config_general, config_agent, render_savefig_flag = False):  
    """Run one repetition of training: init seeds, run episodes, eval & save."""
    traj_buffer = TrajectoryBuffer()  # used to save trajectory-level data for preference learning
    SEED = 48 + i_repetition
    random.seed(SEED)
    np.random.seed(SEED)

    max_num_of_episodes = config_general['max_num_of_episodes']
    max_time_steps_episode = config_general['max_time_steps_episode']
    oracle_teacher = config_general['oracle_teacher']
    human_teacher = config_general['human_teacher']

    environment_name = config_general['environment'] # pendulum(PushT), metaworld, robosuite, obs_avoidance, cartpole, mountaincar
    task = config_general['task']

    save_policy = config_agent['save_policy']
    agent_algorithm = config_agent['algorithm']
    agent_type = config_agent['agent']

    use_abs_action = config_agent['use_abs_action']
    # Pendulum environment seeding example
    if environment_name in ['PushT']:
        env.set_seed(SEED)
        env_seeds = generate_random_numbers(max_num_of_episodes)
        env_seeds_eval = generate_random_numbers(max_num_of_episodes * 10)

    # Initialize agent
    current_agent = agent_selector(agent_type, config_agent)

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
    base_dir = config_agent['base_dir']
    current_agent.saved_dir = os.path.join(base_dir, f"repetition_{i_repetition:03d}/")
    if not os.path.exists(current_agent.saved_dir):
        os.makedirs(current_agent.saved_dir)

    from hydra.utils import get_original_cwd
    project_root = get_original_cwd()
    if config_agent.load_policy:
        current_agent.load_dir = resolve_config_path(
            project_root,
            current_agent.load_dir,
            "AGENT.load_dir",
            required=True,
        )
        logger.info('current_agent.load_dir:  %s', current_agent.load_dir)
        current_agent.load_model()

    record_traj_dataset = getattr(config_general, "record_traj_dataset", False) # used to record traj dataset

    # ===== EPISODES LOOP =====
    for i_episode in range(max_num_of_episodes):
        logger.info('i_episode:  %s', i_episode)
        no_feedback_mode = False  
        if environment_name in ['PushT']:
            env.set_seed(env_seeds[i_episode])

        observation, info = env.reset()

        episode_counter += 1
        h_counter = 0  # how many times feedback was given
        last_action = None
        h = None
    
        Ta = current_agent.horizon # length of executed actions
        Ta_i_teacher =0
        Ta_i = Ta
        Tr = getattr(config_general, "Tr", 2) if config_general is not None else 2 # Queried action-chunk length during intervention
        reset_Ta_i = False
        action_agent_Ta = None
        receive_feedback_phrase = False
        success_ep = 0
        feedback_window = FeedbackWindowBuffer()
        
        current_agent.evaluation = False
        for t in range(1, max_time_steps_episode + 1):
            env.render_mode = 'human'
            if render_savefig_flag: env.render()
            # Time used for logging
            elapsed = (time.time() - start_time - eval_time_acc)
            time_str = str(datetime.timedelta(seconds=elapsed))

            obs_proc = observation

            # Agent's action or teacher's correction
            if h is not None and np.any(h):
                if agent_type in ['HG_DAgger','Implicit_BC','IWR','PVP','Diffusion', 'Set_Supervised_Diffusion']:
                    action_i = h
                else:
                    action_i = last_action + np.matmul(current_agent.e, h)

                action_i = get_teacher_action(environment_name, obs_proc, action_agent=action_i,  env=env, policy_oracle=policy_oracle) # query the teacher here

                if not use_abs_action:
                    action_i = np.clip(action_i, -1, 1)
                Ta_i_teacher = Ta_i_teacher + 1
            else:                
                if Ta_i >= Ta -1 and receive_feedback_phrase is False:
                    receive_feedback_phrase = True
                    
                if Ta_i >= Ta -1 or (receive_feedback_phrase is False and Ta_i == 0):
                    action_agent_Ta = current_agent.action(obs_proc)
                    Ta_i = 0
                
                action_i = action_agent_Ta[Ta_i, :]
                # if receive_feedback_phrase and Ta_i_teacher > 0:
                if receive_feedback_phrase:
                    Ta_i_teacher = Ta_i_teacher + 1
                
            teacher_action_i = get_teacher_action(environment_name, obs_proc, action_agent=action_i,  env=env, policy_oracle=policy_oracle)
            # Step environment
            observation, reward, done, _, info = env.step(action_i)
            
            env_done_fake, success_ep = is_env_done(info)
            done = done or env_done_fake or t == max_time_steps_episode
            last_action = action_i 

            # Decide if feedback is given by the teacher
            if Ta_i_teacher > 2*Ta - 1 and receive_feedback_phrase is True:
                receive_feedback_phrase = False
                Ta_i_teacher = 0
                reset_Ta_i = True

            h = None
            if not no_feedback_mode and oracle_teacher and (receive_feedback_phrase is True):
                # execute the teacher action for Ta steps
                if action_agent_Ta is None or Ta_i >= Tr:
                    action_agent_Ta = current_agent.action(obs_proc)
                    action_agent_i = action_agent_Ta[0, :]
                    Ta_i = 0
                else:      
                    action_agent_i = action_agent_Ta[Ta_i, :]  # we compare the action in the previous step

                if agent_type in ['HG_DAgger','Implicit_BC','IWR','PVP','Diffusion']:
                    if agent_algorithm in ['Diffusion_policy_relative','ibc_relative','pvp_relative']:
                        tmp_h, h_no_threshold = oracle_gimme_feedback(teacher_action_i, action_agent_i, None, config=config_general)
                        h = current_agent.e * np.array(tmp_h) + action_agent_i
                    else:
                        h, h_no_threshold = oracle_feedback_HGDagger(teacher_action_i, action_agent_i, None, config=config_general)
                else:
                    if agent_algorithm in ["Policy_Contrastive_intervention", "Policy_Contrastive_sphere_intervention", "CLIC_EBM"
                                               , 'Set_Supervised_Diffusion_intervention', 'Diffusion_DPO']:
                        h, h_no_threshold = oracle_feedback_HGDagger(teacher_action_i, action_agent_i, None, config=config_general)
                        current_agent.e = np.identity(current_agent.dim_a)
                    else:
                        tmp_h, h_no_threshold = oracle_gimme_feedback(teacher_action_i, action_agent_i, None, config=config_general)
                        h = current_agent.e * np.array(tmp_h) + action_agent_i

                last_action = action_agent_i

            Ta_i = Ta_i + 1  # step Ta after checking whether to give feedback
            if reset_Ta_i:
                Ta_i = 0
                reset_Ta_i = False

            h, Ta_i_teacher = feedback_window.append_step(  # append feedback or reset the buffer to empty
                receive_feedback_phrase=receive_feedback_phrase,
                obs_proc=obs_proc,
                h=h,
                h_no_threshold=h_no_threshold,
                last_action=last_action,
                teacher_action_i=teacher_action_i,
                ta_i_teacher=Ta_i_teacher,
            )

            if human_teacher and not oracle_teacher:
                # Real-time user feedback (example)
                h = feedback_receiver.get_h()
                if feedback_receiver.ask_for_done():
                    done = True

            '''save to traj buffer'''
            if record_traj_dataset:
                # if receive_feedback_phrase and Ta_i_teacher > 0:
                if receive_feedback_phrase and feedback_window.latest_intervention_active():  # check whether the latest buffered step is a real intervention
                    teacher_action_to_buffer = teacher_action_i
                    robot_action_to_buffer = action_agent_i
                    no_robot_action = False
                    no_teacher_action = False  # it is just ~intervention_signal[-1]
                else:
                    robot_action_to_buffer = action_i
                    teacher_action_to_buffer = np.zeros_like(robot_action_to_buffer)
                    no_robot_action = False
                    no_teacher_action = True
                
                # save to trajectory buffer
                traj_buffer.add_transition(
                    obs=obs_proc,     
                    teacher_action=teacher_action_to_buffer,
                    done=done,
                    timestep=t,
                    no_robot_action = no_robot_action,
                    no_teacher_action = no_teacher_action,
                    robot_action=robot_action_to_buffer,                  
                    episode_id= i_episode,
                    if_success = success_ep,
                )

                if done:
                    traj_buffer.finish_trajectory()
                    traj_buffer.save_to_file("trajectory_buffer_"+str(i_repetition))

            # Collect feedback and Train agent
            if not no_feedback_mode:
                training_chunk = feedback_window.pop_training_chunk(Ta)  # pop the next Ta-length chunk for online training
                training_chunk.log_training_chunk()
                h_counter += training_chunk.feedback_count_delta()

                # Collect data if h is not none; Also train the policy 
                current_agent.collect_data_and_train(
                    last_action=training_chunk.training_last_action(agent_type), 
                    h=training_chunk.h_ta,
                    obs_proc=training_chunk.obs_proc,
                    next_obs=None,
                    t=t_total,
                    done=done,
                    agent_algorithm=agent_algorithm,
                    agent_type=agent_type,
                    i_episode=i_episode
                )
                
                if success_ep == 1 and feedback_window.can_flush_success_padding():   # padding data as the same as diffusion policy
                    while feedback_window.can_flush_success_padding():
                        padded_chunk = feedback_window.flush_success_padding_chunk(Ta)  # flush remaining buffered steps with padding after success
                        padded_chunk.log_success_padding_chunk()
                        h_counter += padded_chunk.feedback_count_delta()

                         # Collect data if h is not none; Also train the policy 
                        current_agent.collect_data_and_train(
                            last_action=padded_chunk.training_last_action(agent_type), 
                            h=padded_chunk.h_ta,
                            obs_proc=padded_chunk.obs_proc,
                            next_obs=None, 
                            t=t_total,
                            done=False,
                            agent_algorithm=agent_algorithm,
                            agent_type=agent_type,
                            i_episode=i_episode
                        )            

            t_total += 1
            if done and environment_name in ['robosuite']: 
                policy_oracle.reset() # reset the state machine of oracle policy

            if done:
                base_dir = config_agent['base_dir']
                current_agent.saved_dir = os.path.join(base_dir, f"repetition_{i_repetition:03d}/")
                current_agent.load_dir = current_agent.saved_dir
                if not os.path.exists(current_agent.saved_dir):
                    os.makedirs(current_agent.saved_dir)  
                
                cumm_feedback += h_counter
                # Optionally save policy or buffer
                if save_policy:
                    current_agent.save_model()   
              
                if config_agent['save_buffer']:
                    if getattr(config_agent, 'intervention_replay_buffer_type', None) in ['pickle_obs_action_buffer']:
                        pfile = os.path.join(current_agent.saved_dir, 'buffer_data.pkl')
                        current_agent.buffer.save_to_file(pfile)
                    with open(os.path.join(current_agent.saved_dir, 'data.txt'), 'w') as f:
                        f.write(f"Steps: {t_total}\nEpisode: {i_episode}\nTime: {time_str}\nFeedbacks: {cumm_feedback}\n")
                break
        
        # Evaluate agent periodically
        evaluation_frequency = 10
        if environment_name in ['robosuite', 'PushT']:   
            evaluation_frequency = 5 if i_episode > 150 else 10

        if i_episode % evaluation_frequency == 0 or repetition_done:
            '''Start evaluation here'''
            history = [ep_history, ts_history, time_history, fb_history, success_history, error_history]
            success_rate, mean_error = None, None
            data =    [episode_counter, t_total, time_str, cumm_feedback, success_rate, mean_error]

            eval_time_acc, history_new = evaluation_saving_results_process(eval_agent=current_agent, eval_env= env, policy_oracle=policy_oracle, i_episode=i_episode, i_repetition=i_repetition,
                                              max_steps=max_time_steps_episode, render_savefig_flag = render_savefig_flag, 
                                              history=history, data=data, eval_time_acc = eval_time_acc, SEED_id=SEED, config_general=config_general, config_agent=config_agent)
            ep_history, ts_history, time_history, fb_history, success_history, error_history = history_new

        if repetition_done:
            break
    logger.info(f"== Finished Repetition {i_repetition} with {episode_counter} episodes ==")


# ==========================
#      TRAINING LOGIC: Offline learning
# # ==========================
def train_offline_repetition(env, policy_oracle, feedback_receiver, i_repetition, config_general, config_agent, render_savefig_flag = False):  # used in delftblue
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

    # Pendulum environment seeding example
    if environment_name in ['PushT']:
        env.set_seed(SEED)
        env_seeds = generate_random_numbers(max_num_of_episodes)
        env_seeds_eval = generate_random_numbers(max_num_of_episodes * 10)

    # Initialize agent
    current_agent = agent_selector(agent_type, config_agent)
    # if agent_type != 'Diffusion' or agent_type != 'Set_Supervised_Diffusion':
    if agent_type not in ['Diffusion', 'Set_Supervised_Diffusion', 'CLIC']:
    # if agent_type not in ['Diffusion', 'Set_Supervised_Diffusion']:
        current_agent.createModels(init_neural_network())

    logger.info("LOAD buffer")
    base_dir = config_agent['base_dir']
    current_agent.saved_dir = os.path.join(base_dir, f"repetition_{i_repetition:03d}/")
    current_agent.load_dir = current_agent.saved_dir
    if not os.path.exists(current_agent.saved_dir):
        os.makedirs(current_agent.saved_dir)
    
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

    # For saving
    ep_history, ts_history, time_history = [], [], []
    fb_history, success_history, error_history = [], [], []

    start_time = time.time()

    # ===== EPISODES LOOP =====
    for i_episode in range(max_num_of_episodes):
        logger.info('i_episode:  %s', i_episode)

        if environment_name in ['PushT']:
            env.set_seed(env_seeds[i_episode])

        observation, info = env.reset()

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
        current_agent.saved_dir = os.path.join(base_dir, f"repetition_{i_repetition:03d}/")
        current_agent.load_dir = current_agent.saved_dir
        if not os.path.exists(current_agent.saved_dir):
            os.makedirs(current_agent.saved_dir)  

        if save_policy:
            current_agent.save_model()   

        # Evaluate agent periodically
        evaluation_frequency = 5 if i_episode > 150 else 20

        if i_episode % evaluation_frequency == 0 or repetition_done:
            '''Start evaluation here'''
            history = [ep_history, ts_history, time_history, fb_history, success_history, error_history]
            success_rate, mean_error = None, None
            data =    [episode_counter, t_total, time_str, cumm_feedback, success_rate, mean_error]

            eval_time_acc, history_new = evaluation_saving_results_process(eval_agent=current_agent, eval_env= env, policy_oracle=policy_oracle, i_episode=i_episode, i_repetition=i_repetition,
                                              max_steps=max_time_steps_episode, render_savefig_flag = render_savefig_flag, 
                                              history=history, data=data, eval_time_acc = eval_time_acc, SEED_id=SEED, config_general=config_general, config_agent=config_agent)
            ep_history, ts_history, time_history, fb_history, success_history, error_history = history_new

        if repetition_done:
            break

    logger.info(f"== Finished Repetition {i_repetition} with {episode_counter} episodes ==")


def evaluation_without_training(env, policy_oracle, feedback_receiver, i_repetition, config_general, config_agent, render_savefig_flag=True):
    """Run one repetition of training using offline data."""
    SEED = 100001 + i_repetition
    random.seed(SEED)
    np.random.seed(SEED)

    max_num_of_episodes = config_general['max_num_of_episodes']
    max_time_steps_episode = config_general['max_time_steps_episode']

    environment_name = config_general['environment'] # PushT, metaworld, robosuite, cartpole, mountaincar

    agent_type = config_agent['agent']

    if environment_name in ['PushT']:
        env.set_seed(SEED)

    # Initialize agent
    eval_agent = agent_selector(agent_type, config_agent)

    base_dir = config_agent['base_dir']
    eval_agent.saved_dir = os.path.join(base_dir, f"repetition_{i_repetition:03d}/")
    if not os.path.exists(eval_agent.saved_dir):
        os.makedirs(eval_agent.saved_dir)  

    # used to obtain the src path instead of hydra's temporary run directory
    from hydra.utils import get_original_cwd
    project_root = get_original_cwd()
    eval_agent.load_dir = resolve_config_path(
        project_root,
        eval_agent.load_dir,
        "AGENT.load_dir",
        required=True,
    )
    logger.info('eval_agent.load_dir:  %s', eval_agent.load_dir)
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

        eval_time_acc, history_new = evaluation_saving_results_process(eval_agent=eval_agent, eval_env= env, policy_oracle=policy_oracle, i_episode=i_episode, i_repetition=i_repetition,
                                                  max_steps=max_time_steps_episode, render_savefig_flag = render_savefig_flag, 
                                                  history=history, data=data, eval_time_acc = eval_time_acc, SEED_id=SEED, config_general=config_general, config_agent=config_agent)

        ep_history, ts_history, time_history, fb_history, success_history, error_history = history_new

        if repetition_done:
            break

    logger.info(f"== Finished Repetition {i_repetition} with {episode_counter} episodes ==")


# run by using: python main-receding_horizon.py --config-name=train_Set_Supervised_Diffusion_image_Ta8
@hydra.main(config_path="config", config_name="train_Set_Supervised_Diffusion_image_Ta8")
def main(cfg: DictConfig):
    config_general  = cfg.GENERAL
    config_agent    = cfg.AGENT
    config_feedback = cfg.FEEDBACK
    config_task = cfg.task
    # set up env here
    env, policy_oracle, feedback_receiver = env_selector(config_general, config_feedback, config_task)

    """Main entry point: run multiple training repetitions."""
    logger.info("Starting training...\n")
    for rep_idx in range(config_general['number_of_repetitions']):

        if config_agent['evaluate']:
            evaluation_without_training(env, policy_oracle, feedback_receiver, rep_idx, config_general, config_agent,  render_savefig_flag = config_general.get('render_savefig_flag', False))
        elif config_agent['offline_training']:
            train_offline_repetition(env, policy_oracle, feedback_receiver, rep_idx, config_general, config_agent)   # offline training with dataset
        else:
            train_single_repetition(env, policy_oracle, feedback_receiver, rep_idx, config_general, config_agent, render_savefig_flag = config_general.get('render_savefig_flag', False))  # online training

        # test_teacher_policy(env, policy_oracle, feedback_receiver, rep_idx, config_general, config_agent)


if __name__ == "__main__":
    main()
