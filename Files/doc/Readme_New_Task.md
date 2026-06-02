This file contains instructions on how to add new tasks based on this respository. 

### Create the wrapper for this new task

You can create a wrapper file to wrap some exisiting simulators. You can follow the example of 'env/robosuite/env_robosuite.py' or 'env/metaworld_env/metaworld.py'. Generally, the wrapper file should include the function 'step', 'reset' that is called in the main file. 

Once you succesfully define the wrapper, you can try to load it by calling it in 'env_selector.py'. At the same time, new configuration file w.r.t this new task should be defined below 'config/task'.

### Image observation format

If `use_image` is enabled for the task, the environment observation should be returned as a list/dict of named observation entries that match `shape_meta.obs` in the task config, including each RGB image key and any low-dimensional state keys.

To give feedback to the learning agent in simulation, we implement oracle teacher for each task. Therefore, the oracle should also be defined for this new task. 

