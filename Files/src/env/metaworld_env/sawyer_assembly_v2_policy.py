import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move

# reference: https://github.com/Farama-Foundation/Metaworld/blob/c822f28f582ba1ad49eb5dcf61016566f28003ba/metaworld/policies/sawyer_assembly_v2_policy.py#L7

class SawyerAssemblyV2Policy(Policy):
    def __init__(self):
        super().__init__()
        self.stage = 0  # Initialize the stage variable

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            "hand_pos": obs[:3],
            "gripper": obs[3],
            "wrench_pos": obs[4:7],
            "peg_pos": obs[-3:],
            "unused_info": obs[7:-3],
        }

    def get_action(self, obs):
        obs_ = obs[0, :]
        o_d = self._parse_obs(obs_)
        action = Action({"delta_pos": np.arange(3), "grab_effort": 3})

        action["delta_pos"] = move(
            o_d["hand_pos"], to_xyz=self._desired_pos(o_d), p=10.0
        )
        action["grab_effort"] = self._grab_effort(o_d)
        return action.array

    def _desired_pos(self, o_d):
        pos_curr = o_d["hand_pos"]
        pos_wrench = o_d["wrench_pos"] + np.array([-0.02, 0.0, 0.0])
        # pos_peg = o_d["peg_pos"] + np.array([0.12, 0.0, 0.14])
        pos_peg = o_d["peg_pos"] + np.array([0.12, 0.0, 0.06])

        # # If XY error is greater than 0.02, place end effector above the wrench
        # if np.linalg.norm(pos_curr[:2] - pos_wrench[:2]) > 0.02:
        #     return pos_wrench + np.array([0.0, 0.0, 0.1])
        # # (For later) if lined up with peg, drop down on top of it
        # elif np.linalg.norm(pos_curr[:2] - pos_peg[:2]) <= 0.02:
        #     return pos_peg + np.array([0.0, 0.0, -0.2])
        # # Once XY error is low enough, drop end effector down on top of wrench
        # elif abs(pos_curr[2] - pos_wrench[2]) > 0.05:
        #     return pos_wrench + np.array([0.0, 0.0, 0.03])
        # # If not at the same Z height as the goal, move up to that plane
        # elif abs(pos_curr[2] - pos_peg[2]) > 0.04:
        #     return np.array([pos_curr[0], pos_curr[1], pos_peg[2]])
        # # If XY error is greater than 0.02, place end effector above the peg
        # else:
        #     return pos_peg
        
        # print("pos_wrench: ", pos_wrench)
        gripper_state = o_d["gripper"]
        # print("gripp er: ", gripper_state)
        # If error in the XY plane is greater than 0.02, place end effector above the puck
        if np.linalg.norm(pos_curr[:2] - pos_wrench[:2]) > 0.02 and pos_wrench[-1] < 0.02:
            if np.linalg.norm(pos_curr[:2] - pos_wrench[:2]) < 0.08 and self.stage == 3:
                self.stage = 3
            else:
                self.stage = 1
        # Once XY error is low enough, drop end effector down on top of hammer
        elif abs(pos_curr[2] - pos_wrench[2]) > 0.05 and pos_wrench[-1] < 0.02 and gripper_state > 0.5:
            self.stage = 2
        elif abs(pos_curr[2] - pos_wrench[2]) > 0.03 and pos_wrench[-1] < 0.02 and gripper_state > 0.5:  # gripper_state < 0.5 usually indicates the grasping is good
            self.stage = 3
        # If not at the same Z height as the goal, move up to that plane
        # elif abs(pos_curr[2] - pos_peg[2]) > 0.04 and self.stage <= 4:
        elif pos_curr[2] - pos_peg[2] < - 0.02 and self.stage <=5:
            self.stage = 4
        elif pos_curr[2] - pos_peg[2] >  0.02:
            self.stage = 5
        # Move to the peg
        # elif np.linalg.norm(pos_curr[:2] - pos_peg[:2]) <= 0.02:
        #     self.stage = 5
        # else:
        #     self.stage = 6
        elif np.linalg.norm(pos_curr[:2] - pos_peg[:2]) > 0.02:
            if np.linalg.norm(pos_curr[:2] - pos_peg[:2]) < 0.03 and self.stage == 6:
                self.stage = 6
            else:
                self.stage = 5
        else:
            self.stage = 6
        
        if self.stage == 1:
            return pos_wrench + np.array([0.0, 0.0, 0.1])
        elif self.stage == 2 or self.stage == 3:
            return pos_wrench + np.array([0.0, 0.0, 0.01])
        elif self.stage == 4:
            # return np.array([pos_curr[0], pos_curr[1], pos_peg[2]])
            return pos_peg + np.array([0.0, 0.0, 0.02])
        elif self.stage ==  5:
            return pos_peg 
        else:
            # return pos_peg + np.array([0.0, 0.0, -0.2])
            return pos_peg + np.array([0.0, 0.0, -0.06])
        


    def _grab_effort(self, o_d):
        # print("stage: ", self.stage)
        if self.stage >= 3:
            return 1.0  # close gripper
        else:
            return -1  # open gripper