"""Helper functions for the receding-horizon training entrypoint."""
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


def get_teacher_action(
    environment_name,
    observation,
    action_agent=None,
    env=None,
    policy_oracle=None,
):
    """Return the teacher action for the configured environment."""
    if environment_name in ["metaworld", "robosuite"]:
        if environment_name in ["robosuite"]:
            action_teacher = policy_oracle.get_action(env.obs_extracted, env)
        else:
            action_teacher = policy_oracle.get_action(observation)

    if environment_name in ["mountaincar"]:
        action_teacher = env.control_policy(observation)

    if environment_name in ["PushT"]:
        action_teacher = env.control_policy(action_agent)

    if environment_name in ["obs_avoidance"]:
        action_teacher = env.control_policy(observation, action_agent)

    if environment_name in ["line_following"]:
        action_teacher = env.control_policy(state=None)
    return action_teacher


def evaluation_saving_results_process(
    eval_agent,
    eval_env,
    policy_oracle,
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
        policy_oracle,
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


def save_gif(frames, gif_path, fps=20):
    """Save a list of ``HxWx3`` uint8 frames as a GIF."""
    if len(frames) == 0:
        return
    import imageio.v2 as imageio

    os.makedirs(os.path.dirname(gif_path), exist_ok=True)
    imageio.mimsave(gif_path, frames, fps=fps)


def evaluate_agent(
    agent,
    env,
    policy_oracle,
    i_episode,
    use_image,
    max_steps,
    config_general,
    render_savefig_flag=False,
    verbose=False,
    onlinetraining=True,
    record_gif=True,
    gif_dir="eval_gifs",
    gif_fps=20,
    camera_name="agentview",
):
    """Evaluate an agent over the configured number of episodes."""
    successes, error_list = 0, []
    evaluations_per_training_ = config_general["evaluations_per_training"]
    environment_name = config_general["environment"]
    task = config_general["task"]

    if i_episode < 5 and onlinetraining:
        evaluations_per_training_ = 1

    if environment_name in ["robosuite", "PushT"]:
        if i_episode < 30 and onlinetraining:
            evaluations_per_training_ = 1

    agent.evaluation = True
    logger.info("start evalution")
    for i_episode_ in range(evaluations_per_training_):
        SEED = 100 + 100 * i_episode * evaluations_per_training_ + 10 * i_episode_
        random.seed(SEED)
        np.random.seed(SEED)

        if environment_name in ["PushT"]:
            env.seed(SEED)
        logger.debug('seed:  %s', SEED)

        ep_success = 0
        ep_error = 0
        obs, info = env.reset()

        frames = []

        last_action = np.zeros(agent.dim_a)
        steps_stuck = 0

        Ta = config_general["Ta_executed"]
        Ta_i = Ta
        for t_ev in range(max_steps):
            if render_savefig_flag:
                env.render_mode = "human"
                env.render()

            obs_proc = obs
            if Ta_i >= Ta:
                action_Ta = agent.action(obs_proc)
                Ta_i = 0

            action = action_Ta[Ta_i, :]
            Ta_i = Ta_i + 1

            if environment_name in ["PushT"]:
                teacher = np.array([0, 0])
            else:
                teacher = get_teacher_action(
                    environment_name,
                    obs_proc,
                    action_agent=action,
                    env=env,
                    policy_oracle=policy_oracle,
                )
            if teacher is not None:
                ep_error += np.linalg.norm(teacher - action, ord=2)

            obs, reward, done, _, info = env.step(action)
            env_done_fake, success = is_env_done(info)
            done = done or env_done_fake or t_ev == max_steps - 1

            if record_gif and environment_name in ["robosuite"]:
                if (t_ev % 3) == 0:
                    frame = env.render(
                        mode="rgb_array",
                        height=84,
                        width=84,
                        camera_name=camera_name,
                    )
                    frames.append(frame.astype(np.uint8))

            if success == 1:
                successes += 1
                ep_success = 1
                logger.info('-------------success:  %s  num:  %s  max_steps:  %s', successes, i_episode_ + 1, max_steps)

            if environment_name in ["metaworld"]:
                if np.linalg.norm(last_action - action) < 0.01:
                    steps_stuck += 1
                else:
                    steps_stuck = 0
                if steps_stuck > 200:
                    done = True
            last_action = action
            if environment_name in ["PushT"]:
                if done:
                    ep_error = reward
                else:
                    ep_error = 0
            if done:
                if environment_name in ["robosuite"]:
                    policy_oracle.reset()
                break

        error_list.append(ep_error)
        if verbose:
            logger.info(f"Evaluation -> success={bool(ep_success)}")

        if record_gif and environment_name in ["robosuite"]:
            gif_name = (
                f"eval_ep{i_episode:06d}_k{i_episode_:02d}_seed{SEED}"
                f"_succ{ep_success}.gif"
            )
            gif_path = os.path.join(gif_dir, gif_name)
            save_gif(frames, gif_path, fps=gif_fps)
            logger.info(f"[GIF] saved: {gif_path}  (frames={len(frames)})")

    success_rate = successes / evaluations_per_training_
    mean_error = np.mean(error_list) if error_list else 0.0

    if environment_name in ["robosuite"]:
        policy_oracle.reset()
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
    fname = f"./results/{exp_id}_{rep_idx}.csv"
    df.to_csv(fname, index=False)
    logger.info(f"Saved training progress to {fname}")


def generate_random_numbers(n):
    """Generate random integers between 0 and 1000 for environment seeds."""
    return [random.randint(0, 1000) for _ in range(n)]
