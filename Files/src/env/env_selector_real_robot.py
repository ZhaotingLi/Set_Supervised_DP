import logging

logger = logging.getLogger(__name__)



def env_selector_real_robot(config_general, config_feedback, config_task):
    environment_name = config_general['environment'] # pendulum, metaworld, robosuite, obs_avoidance, cartpole, mountaincar
    task = config_general['task']
    oracle_teacher = config_general['oracle_teacher']
    human_teacher = config_general['human_teacher']

    use_space_mouse = config_feedback['use_space_mouse']

    use_image = config_general['use_image']
    use_abs_action = config_general['use_abs_action']



    from tools.feedback_keyboard_3d import Feedback_keyboard_3d
    from tools.feedback_spacenav import Feedback_spaceNav
    from tools.feedback_spacenav_position import Feedback_spaceNav_position
    from tools.feedback_receiver_combined import Feedback_receiver_combined

    if environment_name in ['Kuka']:
        from env.kuka.kuka_env_box import KUKAenv
        from env.kuka.kuka_env_ball import KUKAenv_ball
        from env.kuka.kuka_env_ball_fixedX import KUKAenv_ball_fixedX 
        from env.kuka.kuka_env_6dEE_SoftH import KUKAenv_6dEE_SoftH
        from env.kuka.kuka_env_6dEE import KUKAenv_6dEE
        from env.kuka.kuka_env_PushT import KUKAenv_pushT
        from env.kuka.kuka_env_PushT_img import KUKAenv_pushT_img
        logger.debug('%s %s %s %s', "type of task: ", type(task), " ", config_general['task'])
        if config_general['task'] == "kuka-ball":
            # env = KUKAenv_ball()
            env = KUKAenv_ball_fixedX()
            logger.debug("use kuka-ball")
            if use_space_mouse:
                feedback_receiver = Feedback_spaceNav()
        elif config_general['task'] == 'kuka-6dEE':
            # env = KUKAenv_6dEE_SoftH()
            env = KUKAenv_6dEE()
            logger.debug("KUKAenv_6dEE_SoftH")
            if use_space_mouse:
                feedback_receiver = Feedback_spaceNav()
        elif config_general['task'] == 'kuka-pushT':
            env = KUKAenv_pushT()
            # env = None  # used for offline training when the robot is not connected
            logger.debug("KUKAenv_pushT")
            if use_space_mouse:
                feedback_receiver = Feedback_spaceNav_position()
        elif config_general['task'] == 'kuka-pushT-img':
            env = KUKAenv_pushT_img(config_task)
            if use_space_mouse:
                feedback_receiver = Feedback_spaceNav_position()
        else:
            env = KUKAenv()
            logger.debug("use kuka pushing box")
            if use_space_mouse:
                feedback_receiver = Feedback_spaceNav_position()
    elif environment_name in ['Franka']:
        from env.franka.franka_env_img import PANDAenv_pushT_img
        env = PANDAenv_pushT_img(config_task)
        if use_space_mouse:
                feedback_receiver = Feedback_receiver_combined()
    
    feedback_receiver_keyboard = Feedback_keyboard_3d()
    
            
    #env.init_varaibles()
    logger.debug('%s %s', "task: ", task)
    policy_oracle = None
    return env, policy_oracle, feedback_receiver, feedback_receiver_keyboard
