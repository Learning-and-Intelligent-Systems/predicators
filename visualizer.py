import pickle as pkl
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


dir = "C:/Users/quinc/Documents/LIS/predicators/saved_datasets/"
fname = "pybullet_multimodal_cover__demo__oracle__5000____100__None.data"


with open(dir + fname, 'rb') as f:
    out = pkl.load(f)

    params_x = []
    params_y = []
    print(len(out.trajectories))
    for trajectory in out.trajectories:
        for action in trajectory.actions:
            if action._option.name == "Pick":
                continue
            params_x.append(action._option.params[0])
            params_y.append(action._option.params[1])


    plt.scatter(params_x, params_y)
    plt.savefig("figure.png")