import logging

logger = logging.getLogger(__name__)



def env_selector(config_general, config_feedback, config_task):
    environment_name = config_general['environment'] # pendulum, metaworld, robosuite, obs_avoidance, cartpole, mountaincar
    task = config_general['task']
    oracle_teacher = config_general['oracle_teacher']
    human_teacher = config_general['human_teacher']

    use_space_mouse = config_feedback['use_space_mouse']

    use_image = config_general['use_image']
    use_abs_action = config_general['use_abs_action']

    if use_space_mouse:
        from tools.feedback_spacenav import Feedback_spaceNav

    if human_teacher is True: 
        from tools.feedback_keyboard_3d import Feedback_keyboard_3d
        from tools.feedback_keyboard_2d import Feedback_keyboard_2d

    if environment_name in ['obs_avoidance']:
        # from env.obs_avoidance.obs_avoidance_env import ObstacleAvoidanceEnv

        # env = ObstacleAvoidanceEnv()
        # task_short = task +'-a30-Bayesian_l1024_M2_QInitZero_QPenalty_sampleAll_Temp01'+agent_type 

        from env.DesiredActionSpace_ToyEnv.DesiredA_toyenv import DesiredA_ToyEnv
        from env.DesiredActionSpace_ToyEnv.DesiredA_toyenv_square import DesiredA_ToyEnv_square
        from env.DesiredActionSpace_ToyEnv.DesiredA_toyenv_TwoCircles import DesiredA_ToyEnv_TwoCircles
        if task == 'Desired_A':
            env = DesiredA_ToyEnv()
            # env = DesiredA_ToyEnv_TwoCircles()
        elif task == 'Desired_A_Square':
            env = DesiredA_ToyEnv_square()

        feedback_receiver = None
        policy_oracle = None

    if environment_name in ['PushT']:

        # pushT task
        from env.pusht.pusht_keypoints_env import PushTKeypointsEnv
        from env.pusht.pusht_env import PushTEnv
        from env.pusht.pusht_image_env import PushTImageEnv
        
        if use_image:
            env = PushTImageEnv(use_abs_action=use_abs_action, config=config_task)
        else:
            if task in ['pushT_abs_illustration']:
                from env.pusht.pusht_env_illustration import PushTEnv_illustration
                env = PushTEnv_illustration(use_abs_action=use_abs_action, config=config_task)
            else:
                env = PushTEnv(use_abs_action=use_abs_action, config=config_task)
            
        if human_teacher is False: 
            feedback_receiver = None
        else:
            feedback_receiver = Feedback_keyboard_2d()
        policy_oracle = None

    if environment_name in ['cartpole']:
        from tools.feedback import Feedback
        env = gym.make(environment_name)  # create environment
        task_short = "cartpole"
        if human_teacher is False: 
            feedback_receiver = None
        else:
            feedback_receiver = Feedback(env=env)


    if environment_name in ['robosuite']:
        if human_teacher is False: 
            feedback_receiver = None
        else:
            feedback_receiver = Feedback_keyboard_3d()
        
        from env.robotsuite.env_robosuite import EnvRobosuite
        
        env = EnvRobosuite(env_name=task, use_image_obs=use_image, 
                        use_abs_action= use_abs_action, config=config_task)
        if task == 'NutAssemblySquare':
            from env.robotsuite.nutassemblysquare_policy import NutAssemblyPolicy
            # from env.robotsuite.nutassemblysquare_policy_old24 import NutAssemblyPolicy
            policy_oracle = NutAssemblyPolicy(use_abs_action= use_abs_action)
        elif task == 'ToolHang':
            from env.robotsuite.toolhang_policy import ToolHangPolicy
            policy_oracle = ToolHangPolicy()
        elif task == 'PickPlaceCan':
            from env.robotsuite.pickCan_policy import PickCanPolicy
            policy_oracle = PickCanPolicy(use_abs_action= use_abs_action)
        elif task == 'TwoArmLift':
            from env.robotsuite.two_arm_lift_policy import TwoArmLiftPolicy
            policy_oracle = TwoArmLiftPolicy(use_abs_action=use_abs_action)

    if environment_name in ['mountaincar_continuous']:
        import gymnasium as gym
        env = gym.make('MountainCarContinuous-v0', render_mode="human")  
       
        from tools.feedback_keyboard_1d import Feedback_keyboard_1d
        feedback_receiver = Feedback_keyboard_1d()


    if environment_name in ['line_following']:
        # line following task
        from env.followLine.followLine_env import LineFollowerEnv
        # env = PushTEnv()
        if task == "line_following":
            env = LineFollowerEnv(random_lines = getattr(config_task, 'random_lines', False)) 
            policy_oracle = env.control_policy
            if human_teacher is False: 
                feedback_receiver = None
            else:
                feedback_receiver = Feedback_keyboard_3d()
            
        elif task == "multiLine-Following":
            from env.followLine.multifollowLine_env import MultiLineFollowerEnv
            from tools.feedback_keyboard_multi_2d import Feedback_keyboard_multi_2d
            env = MultiLineFollowerEnv()
            policy_oracle = None
            if human_teacher is False: 
                feedback_receiver = None
            else:
                feedback_receiver = Feedback_keyboard_multi_2d(robot_num=env.n_balls)
        else: # raise error
            logger.debug("task not found")
            raise ValueError
       

    elif environment_name in ['metaworld']:
        from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE)

        from metaworld.policies.sawyer_drawer_open_v2_policy import SawyerDrawerOpenV2Policy
        from metaworld.policies.sawyer_button_press_v2_policy import SawyerButtonPressV2Policy
        from metaworld.policies.sawyer_reach_v2_policy import SawyerReachV2Policy
        from metaworld.policies.sawyer_plate_slide_v2_policy import SawyerPlateSlideV2Policy
        from metaworld.policies.sawyer_button_press_topdown_v2_policy import SawyerButtonPressTopdownV2Policy
        from metaworld.policies.sawyer_push_v2_policy import SawyerPushV2Policy
        from metaworld.policies.sawyer_door_open_v2_policy import SawyerDoorOpenV2Policy
        from metaworld.policies.sawyer_lever_pull_v2_policy import SawyerLeverPullV2Policy

        # from metaworld.policies.sawyer_assembly_v2_policy import SawyerAssemblyV2Policy

        from metaworld.policies.sawyer_basketball_v2_policy import SawyerBasketballV2Policy
        from metaworld.policies.sawyer_shelf_place_v2_policy import SawyerShelfPlaceV2Policy
        from metaworld.policies.sawyer_pick_place_v2_policy import SawyerPickPlaceV2Policy
        from metaworld.policies.sawyer_soccer_v2_policy import SawyerSoccerV2Policy
        from metaworld.policies.sawyer_sweep_v2_policy import SawyerSweepV2Policy
        from metaworld.policies.sawyer_button_press_topdown_wall_v2_policy import SawyerButtonPressTopdownWallV2Policy


        from metaworld.policies.sawyer_peg_insertion_side_v2_policy import SawyerPegInsertionSideV2Policy

        from metaworld.policies.sawyer_door_lock_v2_policy import SawyerDoorLockV2Policy
        from metaworld.policies.sawyer_door_unlock_v2_policy import SawyerDoorUnlockV2Policy
        from metaworld.policies.sawyer_window_open_v2_policy import SawyerWindowOpenV2Policy
        from metaworld.policies.sawyer_window_close_v2_policy import SawyerWindowCloseV2Policy
        from metaworld.policies.sawyer_handle_press_side_v2_policy import SawyerHandlePressSideV2Policy
        # from metaworld.policies.sawyer_hammer_v2_policy import SawyerHammerV2Policy
        from env.metaworld_env.sawyer_hammer_v2_policy import SawyerHammerV2Policy
        from env.metaworld_env.sawyer_assembly_v2_policy import  SawyerAssemblyV2Policy


        # Create Environment
        # task = task.strip('"')
        # plate_slide_goal_observable_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task]
        # env = plate_slide_goal_observable_cls()
        # plate_slide_goal_observable_cls2 = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task]
        # env_eval = plate_slide_goal_observable_cls2()
        from env.metaworld_env.metaworld import MetaWorldSawyerEnv
        
        env = MetaWorldSawyerEnv(task)
        env_eval =MetaWorldSawyerEnv(task)
        # env = MetaWorldSawyerEnv()
        # env_eval = MetaWorldSawyerEnv(env_eval)
        if use_space_mouse:
            feedback_receiver = Feedback_spaceNav()
        if human_teacher is False: 
            feedback_receiver = None
        else:
            feedback_receiver = Feedback_keyboard_3d()
            
        # Create Oracle policy
        if task == "drawer-open-v2-goal-observable" or task == "drawer-open-v2":
            policy_oracle = SawyerDrawerOpenV2Policy()
           
        elif task == "hammer-v2-goal-observable" or task == "hammer-v2":
            policy_oracle = SawyerHammerV2Policy()
            
        elif task == "assembly-v2-goal-observable":
            policy_oracle = SawyerAssemblyV2Policy()
           
        elif task == "button-press-v2-goal-observable":
            policy_oracle = SawyerButtonPressV2Policy()
        elif task == "reach-v2-goal-observable":
            policy_oracle = SawyerReachV2Policy()
        elif task == "plate-slide-v2-goal-observable":
            policy_oracle = SawyerPlateSlideV2Policy()
        elif task == "button-press-topdown-v2-goal-observable":
            policy_oracle = SawyerButtonPressTopdownV2Policy()
        elif task == "push-v2-goal-observable":
            policy_oracle = SawyerPushV2Policy()
        elif task == "peg_insertion_side-v2-goal-observable":
            policy_oracle = SawyerPegInsertionSideV2Policy()
        elif task == "door-open-v2-goal-observable":
            policy_oracle = SawyerDoorOpenV2Policy()
        elif task == "basketball-v2-goal-observable":
            policy_oracle = SawyerBasketballV2Policy()
        elif task == "shelf-place-v2-goal-observable":
            policy_oracle = SawyerShelfPlaceV2Policy()
        elif task == "soccer-v2-goal-observable":
            policy_oracle = SawyerSoccerV2Policy()
        elif task == "pick-place-v2-goal-observable":
            policy_oracle = SawyerPickPlaceV2Policy()
        elif task == "sweep-v2-goal-observable":
            policy_oracle = SawyerSweepV2Policy()
        elif task == "button-press-topdown-wall-v2-goal-observable":
            policy_oracle = SawyerButtonPressTopdownWallV2Policy()
        elif task == "lever-pull-v2-goal-observable":
            policy_oracle = SawyerLeverPullV2Policy()
        elif task == "door-lock-v2-goal-observable":
            policy_oracle = SawyerDoorLockV2Policy()
        elif task == "door-unlock-v2-goal-observable":
            policy_oracle = SawyerDoorUnlockV2Policy()
        elif task == "window-open-v2-goal-observable":
            policy_oracle = SawyerWindowOpenV2Policy()
        elif task == "window-close-v2-goal-observable":
            policy_oracle = SawyerWindowCloseV2Policy()
        elif task == "handle-press-side-v2-goal-observable":
            policy_oracle = SawyerHandlePressSideV2Policy()

    return env, policy_oracle, feedback_receiver
