from shelves2d import Shelves2DEnv
import predicators.utils as utils
from predicators.structs import Object, Action
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use("tkagg")
utils.reset_config()
env = Shelves2DEnv()

task = env.get_task("train", 1)

print(utils.abstract(task.init, env.predicates))

fig = env.render_state_plt(task.init, task)

plt.show()