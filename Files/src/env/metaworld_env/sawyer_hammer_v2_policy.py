import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move

# SOURCE: https://github.com/Farama-Foundation/Metaworld/blob/c822f28f582ba1ad49eb5dcf61016566f28003ba/metaworld/policies/sawyer_hammer_v2_policy.py#L7 

class SawyerHammerV2Policy(Policy):
    def __init__(self):
        super().__init__()
        self.stage = 0  # Initialize the stage variable

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            "hand_pos": obs[:3],
            "gripper": obs[3],
            "hammer_pos": obs[4:7],
            "unused_info": obs[7:],
        }

    def get_action(self, obs):
        obs_ = obs[0, :]
        o_d = self._parse_obs(obs_)
        action = Action({"delta_pos": np.arange(3), "grab_effort": 3})

        action["delta_pos"] = move(
            o_d["hand_pos"], to_xyz=self._desired_pos(o_d), p=10.0
        )
        action["grab_effort"] = self._grab_effort(o_d)

        # print("teacher action: ", action)
        return action.array


    def _desired_pos(self, o_d):
        pos_curr = o_d["hand_pos"]
        pos_puck = o_d["hammer_pos"] + np.array([-0.04, 0.0, -0.01])
        pos_goal = np.array([0.24, 0.71, 0.11]) + np.array([-0.19, 0.0, 0.05])

        gripper_state = o_d["gripper"]
        # print("gripp er: ", gripper_state)
        # If error in the XY plane is greater than 0.02, place end effector above the puck
        if np.linalg.norm(pos_curr[:2] - pos_puck[:2]) > 0.02:
            if np.linalg.norm(pos_curr[:2] - pos_puck[:2]) < 0.03 and self.stage == 3:
                self.stage = 3
            else:
                self.stage = 1
        # Once XY error is low enough, drop end effector down on top of hammer
        elif abs(pos_curr[2] - pos_puck[2]) > 0.05 and pos_puck[-1] < 0.02:
            self.stage = 2
        elif abs(pos_curr[2] - pos_puck[2]) > 0.03 and pos_puck[-1] < 0.02 and gripper_state > 0.5:  # gripper_state < 0.5 usually indicates the grasping is good
            self.stage = 3
        # If not at the same X pos as the peg, move over to that plane
        elif np.linalg.norm(pos_curr[[0, 2]] - pos_goal[[0, 2]]) > 0.02:
            self.stage = 4
        # Move to the peg
        else:
            self.stage = 5
        
        if self.stage == 1:
            if pos_curr[2] - pos_puck[2] < 0.09:
                return pos_curr + np.array([0.0, 0.0, pos_puck[2] + 0.2])
            else:
                return pos_puck + np.array([0.0, 0.0, 0.1])
        elif self.stage == 2 or self.stage == 3:
            return pos_puck + np.array([0.0, 0.0, 0.01])
        elif self.stage == 4:
            return np.array([pos_goal[0], pos_curr[1], pos_goal[2]])
        elif self.stage ==  5:
            return pos_goal
        else:
            return pos_curr

    def _grab_effort(self, o_d):
        pos_curr = o_d["hand_pos"]
        pos_puck = o_d["hammer_pos"] + np.array([-0.04, 0.0, -0.01])
        pos_goal = np.array([0.24, 0.71, 0.11]) + np.array([-0.19, 0.0, 0.05])
        # print("grab effort function, pos_curr: ",pos_curr, " pos_puck: ", pos_puck)

        # # If error in the XY plane is greater than 0.02, place end effector above the puck
        # if np.linalg.norm(pos_curr[:2] - pos_puck[:2]) > 0.02 and self.stage <=2:    
        #     self.stage = 1  # go to the above of the hammer
        # # Once XY error is low enough, drop end effector down on top of hammer
        # elif abs(pos_curr[2] - pos_puck[2]) > 0.05 and pos_puck[-1] < 0.02:
        #     self.stage = 2  # try to reach the hammer from above
        # elif abs(pos_curr[2] - pos_puck[2]) > 0.03 and pos_puck[-1] < 0.02:
        #     self.stage = 3  # try to grasp it
        # # If not at the same X pos as the peg, move over to that plane
        # elif np.linalg.norm(pos_curr[[0, 2]] - pos_goal[[0, 2]]) > 0.02:
        #     self.stage = 4  # hammer grasped, try to align the x of the peg
        # # Move to the peg
        # else:
        #     self.stage = 5   # hammer grasped, go to the peg

        # print("stage: ", self.stage)
        # if (
        #     (np.linalg.norm(pos_curr[:2] - pos_puck[:2]) > 0.01
        #     or abs(pos_curr[2] - pos_puck[2]) > 0.1)
        #     or (stage == 1 or stage == 2)
        # ):
        #     return 0.0
        # # While end effector is moving down toward the hammer, begin closing the grabber
        # else:
        #     return 0.8
        if self.stage >= 3:
            return 1.0  # close gripper
        else:
            return -1  # open gripper