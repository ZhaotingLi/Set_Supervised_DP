"""Helper functions for the real-robot training entrypoint."""
import logging

import os
import random
import time

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def is_env_done(info):
    """Return the environment done flag and success indicator from ``info``."""
    if info.get("success", 0) == 1:
        return True, 1
    return False, 0


def evaluation_saving_results_process(
    eval_agent,
    eval_env,
    feedback_receiver,
    feedback_receiver_keyboard,
    i_episode,
    i_repetition,
    max_steps,
    render_savefig_flag,
    history,
    data,
    eval_time_acc,
    SEED_id,
    config_general,
    config_agent,
):
    """Evaluate the agent, update histories, and persist progress if enabled."""
    eval_start = time.time()
    success_rate, mean_error = evaluate_agent(
        eval_agent,
        eval_env,
        feedback_receiver,
        feedback_receiver_keyboard,
        i_episode,
        use_image=False,
        max_steps=max_steps,
        config_general=config_general,
        verbose=False,
        render_savefig_flag=render_savefig_flag,
    )
    eval_end = time.time()
    eval_time_acc += eval_end - eval_start

    data[-2:] = success_rate, mean_error
    for his, d in zip(history, data):
        his.append(d)

    if config_general["save_results"]:
        save_training_progress(
            *history,
            eval_agent.e,
            config_general["task"],
            SEED_id,
            i_repetition,
            config_agent=config_agent,
        )
    return eval_time_acc, history


def evaluate_agent(
    agent,
    env,
    feedback_receiver,
    feedback_receiver_keyboard,
    i_episode,
    use_image,
    max_steps,
    config_general,
    render_savefig_flag=False,
    verbose=False,
    onlinetraining=True,
):
    """Evaluate the real-robot policy until success or operator reset."""
    import rospy

    successes, error_list = 0, []
    evaluations_per_training_ = config_general["evaluations_per_training"]
    environment_name = config_general["environment"]
    task = config_general["task"]

    if i_episode < 5 and onlinetraining:
        evaluations_per_training_ = 1

    agent.evaluation = True
    logger.info("start evalution")
    for i_episode_ in range(evaluations_per_training_):
        ep_success = 0
        ep_error = 0
        obs, info = env.reset()

        logger.info("press space key to start")
        while not feedback_receiver_keyboard.ask_for_done():
            env.hold_on_mode()

        last_action = np.zeros(agent.dim_a)
        steps_stuck = 0

        Ta = config_general["Ta_executed"]
        Ta_i = Ta

        control_frequency = 10
        rate = rospy.Rate(control_frequency)

        while True:
            obs_proc = obs
            if Ta == 1:
                action = agent.action(obs_proc)
            else:
                start = time.time()
                if Ta_i >= Ta:
                    action_Ta = agent.action(obs_proc)
                    Ta_i = 0
                action = action_Ta[Ta_i, :]

                end = time.time()
                logger.debug('%s  Ta_i:  %s', f"Elapsed time: {end - start:.6f} seconds", Ta_i)
                Ta_i = Ta_i + 1

            obs, reward, done, _, info = env.step(action)
            env_done_fake, success = is_env_done(info)

            done_restart = feedback_receiver.ask_for_done()
            done = done or env_done_fake or done_restart

            if success == 1:
                successes += 1
                ep_success = 1
                logger.info('-------------success:  %s  num:  %s', successes, i_episode_ + 1)

            last_action = action

            if done:
                break

            rate.sleep()

        error_list.append(ep_error)
        if verbose:
            logger.info(f"Evaluation -> success={bool(ep_success)}")

    success_rate = successes / evaluations_per_training_
    mean_error = np.mean(error_list) if error_list else 0.0

    logger.info("end evalution")
    return success_rate, mean_error


def save_training_progress(
    ep_list,
    ts_list,
    time_list,
    fb_list,
    success_list,
    error_list,
    e_mat,
    task_short,
    seed,
    rep_idx,
    config_agent,
):
    """Save training progress to a CSV file in ``./results``."""
    if not os.path.exists("./results"):
        os.makedirs("./results")

    df = pd.DataFrame(
        {
            "Episode": ep_list,
            "Timesteps": ts_list,
            "time": time_list,
            "Amount of feedback": fb_list,
            "Success rate": success_list,
            "error to simulated teacher": error_list,
        }
    )

    exp_id = config_agent["experiment_id"]
    agent_type = config_agent["agent"]
    agent_algorithm = config_agent["algorithm"]
    e_val = e_mat[0] if isinstance(e_mat, (list, np.ndarray)) else e_mat
    fname = (
        f"./results/{exp_id}"
        f"_{agent_type}"
        f"_Alg-{agent_algorithm}"
        f"_{rep_idx}.csv"
    )
    df.to_csv(fname, index=False)
    logger.info(f"Saved training progress to {fname}")


def generate_random_numbers(n):
    """Generate random integers between 0 and 1000 for environment seeds."""
    return [random.randint(0, 1000) for _ in range(n)]
