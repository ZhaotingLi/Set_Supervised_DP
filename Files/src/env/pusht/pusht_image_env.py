from gym import spaces
from env.pusht.pusht_env import PushTEnv
import numpy as np
import cv2
from env.robotsuite.env_robosuite import combine_obs_dicts

class PushTImageEnv(PushTEnv):
    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 10}

    def __init__(self,
            legacy=False,
            block_cog=None, 
            damping=None,
            render_size=96, use_abs_action = False,
            config = None):
        super().__init__(
            legacy=legacy, 
            block_cog=block_cog,
            damping=damping,
            render_size=render_size,
            render_action=False,
            use_abs_action = use_abs_action,
            config = config)
        ws = self.window_size
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0,
                high=1,
                shape=(3,render_size,render_size),
                dtype=np.float32
            ),
            'agent_pos': spaces.Box(
                low=0,
                high=ws,
                shape=(2,),
                dtype=np.float32
            )
        })
        self.render_cache = None
        self.obs_dict_past = None
    
    def _get_obs(self):
        img = super()._render_frame(mode='rgb_array')

        agent_pos = np.array(self.agent.position).copy()


        # cv2.imshow(f"Obs_", img)
        # cv2.waitKey(1)

        agent_pos = self.normalize_abs_action(agent_pos, action_max=self.eef_pos_max_list[0], action_min=self.eef_pos_min_list[0])
        img_obs = np.moveaxis(img.astype(np.float32) / 255, -1, 0)
        img_obs = (2.0 * img_obs - 1.0).astype(np.float32)
        obs = {
            'image': img_obs,
            'agent_pos': agent_pos
        }

        # draw action
        if self.latest_action is not None:
            action = np.array(self.latest_action)
            coord = (action / 512 * 96).astype(np.int32)
            marker_size = int(8/96*self.render_size)
            thickness = int(1/96*self.render_size)
            cv2.drawMarker(img, coord,
                color=(255,0,0), markerType=cv2.MARKER_CROSS,
                markerSize=marker_size, thickness=thickness)
        self.render_cache = img

        # combine past the current obs_dict
        # print("self.obs_dict_past is None: ", self.obs_dict_past is None)
        if self.obs_dict_past is None:
            self.obs_dict_past = obs
        
        obs_dict_combined = combine_obs_dicts(self.obs_dict_past, obs) 
        # obs_dict_combined[self.rgb_keys] -> (2, C, H, W)
        # obs_dict_combined[self.lowdim_keys] -> (2, lowdim_keys)

        self.obs_dict_past = obs
        # import pdb; pdb.set_trace()
        return obs_dict_combined

    # def render(self):
    #     mode = 'rgb_array'
    #     assert mode == 'rgb_array'

    #     if self.render_cache is None:
    #         self._get_obs()
        
    #     return self.render_cache
