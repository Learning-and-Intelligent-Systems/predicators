import tkinter

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from predicators import utils
from predicators.envs.bookshelf import BookshelfEnv

matplotlib.use('TkAgg')


def test_init_sampler():
    print('Place bookshelf anywhere')
    utils.reset_config({"env": "bookshelf"})
    env = BookshelfEnv()
    # t = env._get_tasks(1, [200], env._train_rng)
    for _ in range(1):
        t = env._get_tasks(1, [6], [0], env._train_rng)
        state = t[0].init
        env.render_state_plt(state, t[0])
        # plt.figure()
        # im = env.grid_state(state)
        # plt.imshow(im)
        # plt.show()
        plt.savefig("books.pdf")
        plt.close()
    # print('Place bookshelf against some wall')
    # utils.reset_config({'bookshelf_against_wall': True})
    # env = BookshelfEnv()
    # for _ in range(10):
    #     t = env._get_tasks(1, [10], env._train_rng)
    #     state = t[0].init
    #     env.render_state_plt(state, t[0])
    #     plt.show()
    #     plt.close()

def test_navigate_bookshelf_action():
    utils.reset_config({"env": "bookshelf"})
    env = BookshelfEnv()
    t = env._get_tasks(1, [0], env._train_rng)
    state = t[0].init
    env.render_state_plt(state, t[0])
    plt.show()

    navigate_objects = [env._robot, env._shelf]
    navigate_params = np.array([-0.4, 0.5], dtype=np.float32)
    a = env._NavigateTo_policy(state, {}, navigate_objects, navigate_params)
    s = env.simulate(state, a)
    env.render_state_plt(s, t[0], caption='left')
    plt.show()

    navigate_params = np.array([1.4, 0.5], dtype=np.float32)
    a = env._NavigateTo_policy(state, {}, navigate_objects, navigate_params)
    s = env.simulate(state, a)
    env.render_state_plt(s, t[0], caption='right')
    plt.show()

    navigate_params = np.array([0.5, -1], dtype=np.float32)
    a = env._NavigateTo_policy(state, {}, navigate_objects, navigate_params)
    s = env.simulate(state, a)
    env.render_state_plt(s, t[0], caption='bottom')
    plt.show()

    navigate_params = np.array([0.5, 2], dtype=np.float32)
    a = env._NavigateTo_policy(state, {}, navigate_objects, navigate_params)
    s = env.simulate(state, a)
    env.render_state_plt(s, t[0], caption='top')
    plt.show()

    navigate_params = np.array([1.5, 2], dtype=np.float32)
    a = env._NavigateTo_policy(state, {}, navigate_objects, navigate_params)
    s = env.simulate(state, a)
    env.render_state_plt(s, t[0], caption='corner')
    plt.show()

    navigate_params = np.array([-0.5, 2], dtype=np.float32)
    a = env._NavigateTo_policy(state, {}, navigate_objects, navigate_params)
    s = env.simulate(state, a)
    env.render_state_plt(s, t[0], caption='corner')
    plt.show()

    navigate_params = np.array([1.5, -1], dtype=np.float32)
    a = env._NavigateTo_policy(state, {}, navigate_objects, navigate_params)
    s = env.simulate(state, a)
    env.render_state_plt(s, t[0], caption='corner')
    plt.show()

    navigate_params = np.array([-0.5, -1], dtype=np.float32)
    a = env._NavigateTo_policy(state, {}, navigate_objects, navigate_params)
    s = env.simulate(state, a)
    env.render_state_plt(s, t[0], caption='corner')
    plt.show()

    navigate_params = np.array([0, 0], dtype=np.float32)
    a = env._NavigateTo_policy(state, {}, navigate_objects, navigate_params)
    s = env.simulate(state, a)
    env.render_state_plt(s, t[0], caption='fail')
    plt.show()


def test_navigate_book_action():
    utils.reset_config({"env": "bookshelf"})
    env = BookshelfEnv()
    t = env._get_tasks(1, [1], env._train_rng)
    state = t[0].init
    env.render_state_plt(state, t[0])
    plt.show()

    book = [b for b in state if b.is_instance(env._book_type)][0]
    navigate_objects = [env._robot, book]
    navigate_params = np.array([-4, 0.5], dtype=np.float32)
    a = env._NavigateTo_policy(state, {}, navigate_objects, navigate_params)
    s = env.simulate(state, a)
    env.render_state_plt(s, t[0], caption='left')
    plt.show()

    navigate_params = np.array([5, 0.5], dtype=np.float32)
    a = env._NavigateTo_policy(state, {}, navigate_objects, navigate_params)
    s = env.simulate(state, a)
    env.render_state_plt(s, t[0], caption='right')
    plt.show()

    navigate_params = np.array([0.5, -2], dtype=np.float32)
    a = env._NavigateTo_policy(state, {}, navigate_objects, navigate_params)
    s = env.simulate(state, a)
    env.render_state_plt(s, t[0], caption='bottom')
    plt.show()

    navigate_params = np.array([0.5, 3], dtype=np.float32)
    a = env._NavigateTo_policy(state, {}, navigate_objects, navigate_params)
    s = env.simulate(state, a)
    env.render_state_plt(s, t[0], caption='top')
    plt.show()

    navigate_params = np.array([5, 3], dtype=np.float32)
    a = env._NavigateTo_policy(state, {}, navigate_objects, navigate_params)
    s = env.simulate(state, a)
    env.render_state_plt(s, t[0], caption='corner')
    plt.show()

    navigate_params = np.array([-4, 3], dtype=np.float32)
    a = env._NavigateTo_policy(state, {}, navigate_objects, navigate_params)
    s = env.simulate(state, a)
    env.render_state_plt(s, t[0], caption='corner')
    plt.show()

    navigate_params = np.array([5, -2], dtype=np.float32)
    a = env._NavigateTo_policy(state, {}, navigate_objects, navigate_params)
    s = env.simulate(state, a)
    env.render_state_plt(s, t[0], caption='corner')
    plt.show()

    navigate_params = np.array([-4, -2], dtype=np.float32)
    a = env._NavigateTo_policy(state, {}, navigate_objects, navigate_params)
    s = env.simulate(state, a)
    env.render_state_plt(s, t[0], caption='corner')
    plt.show()

    navigate_params = np.array([0, 0], dtype=np.float32)
    a = env._NavigateTo_policy(state, {}, navigate_objects, navigate_params)
    s = env.simulate(state, a)
    env.render_state_plt(s, t[0], caption='fail')
    plt.show()


def test_navigate_pick_place():
    utils.reset_config({"env": "bookshelf"})
    env = BookshelfEnv()
    t = env._get_tasks(1, [1], env._train_rng)
    state = t[0].init
    env.render_state_plt(state, t[0])
    plt.show()

    book = [b for b in state if b.is_instance(env._book_type)][0]
    navigate_objects = [env._robot, book]
    navigate_params = np.array([-4, 0.5], dtype=np.float32)
    a = env._NavigateTo_policy(state, {}, navigate_objects, navigate_params)
    s = env.simulate(state, a)
    env.render_state_plt(s, t[0], caption='left')
    assert env._CanReach_holds(s, [book, env._robot])
    plt.show()

    state = s
    pick_objects = [env._robot, book]
    pick_params = np.array([0.25, 0.0], dtype=np.float32)
    a = env._PickBook_policy(state, {}, pick_objects, pick_params)
    s = env.simulate(state, a)
    env.render_state_plt(s, t[0], caption='pick')
    assert env._Holding_holds(s, [book])
    plt.show()

    state = s
    navigate_params = np.array([-0.4, 0.5], dtype=np.float32)
    navigate_objects = [env._robot, env._shelf]
    a = env._NavigateTo_policy(state, {}, navigate_objects, navigate_params)
    s = env.simulate(state, a)
    env.render_state_plt(s, t[0], caption='right')
    assert env._CanReach_holds(s, [env._shelf, env._robot])
    plt.show()

    state = s
    place_params = np.array([1.0], dtype=np.float32)
    place_objects = [env._robot, book, env._shelf]
    a = env._PlaceBookOnShelf_policy(state, {}, place_objects, place_params)
    s = env.simulate(state, a)
    env.render_state_plt(s, t[0], caption='right')
    assert env._OnShelf_holds(s, [book, env._shelf])
    plt.show()

def test_ebm():
    from predicators.ground_truth_nsrts import _get_options_by_names
    utils.reset_config({"env": "bookshelf"})
    env = BookshelfEnv()
    t = env._get_tasks(1, [1], env._train_rng)
    state = t[0].init

    book = [b for b in state if b.is_instance(env._book_type)][0]
    navigate_objects = [env._robot, book]
    NavigateTo, = _get_options_by_names("bookshelf", ["NavigateTo"])
    
    X = []
    Y = []

    for _ in range(1000):
        navigate_params = NavigateTo.params_space.sample()
        a = env._NavigateTo_policy(state, {}, navigate_objects, navigate_params)
        s = env.simulate(state, a)
        x = np.r_[state[book], state[env._robot]]
        y = env._CanReach_holds(s, [book, env._robot]) 
        X.append(np.r_[x, navigate_params])
        Y.append(y)

    import torch
    from torch import nn
    from torch.optim import Adam
    import torch.autograd as autograd

    def sample_langevin(x_fixed, x, model, stepsize, n_steps, noise_scale=None, intermediate_samples=False):
        # Note: taken https://github.com/swyoon/pytorch-energy-based-model/blob/master/langevin.py
        """Draw samples using Langevin dynamics
        x: torch.Tensor, initial points
        model: An energy-based model
        stepsize: float
        n_steps: integer
        noise_scale: Optional. float. If None, set to np.sqrt(stepsize * 2)
        """
        if noise_scale is None:
            noise_scale = np.sqrt(stepsize * 2)

        l_samples = []
        l_dynamics = []
        x.requires_grad = True
        for _ in range(n_steps):
            l_samples.append(x.detach().to('cpu'))
            noise = torch.randn_like(x) * noise_scale
            out = model(torch.cat((x_fixed, x), dim=1))
            grad = autograd.grad(out.sum(), x)[0]
            dynamics = stepsize * grad + noise
            x = x + dynamics
            l_samples.append(x.detach().to('cpu'))
            l_dynamics.append(dynamics.detach().to('cpu'))

        if intermediate_samples:
            return l_samples, l_dynamics
        else:
            return l_samples[-1]

    X = torch.from_numpy(np.stack(X)).float()
    Y = torch.from_numpy(np.array(Y)[:, np.newaxis]).float()

    mlp = nn.Sequential(
        nn.Linear(X.shape[1] , 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 1)
    )
    mlp.train()
    loss = nn.BCEWithLogitsLoss()
    optimizer = Adam(mlp.parameters())
    for epoch in range(1000):
        Yhat = mlp(X)
        l = loss(Yhat, Y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    mlp.eval()
    Yhat = mlp(X)
    acc = ((Yhat > 0).float() == Y).float().mean()
    print(acc)

    Xhat = sample_langevin(X[:,:-2], torch.randn_like(X[:,-2:]), mlp, 0.01, 5000)
    # Xhat = sample_langevin(X[:,:-2], torch.zeros_like(X[:,-2:]), mlp, 0.01, 5000)

    real_pos_x = np.empty(X.shape[0])
    real_pos_y = np.empty(X.shape[0])
    sample_pos_x = np.empty(X.shape[0])
    sample_pos_y = np.empty(X.shape[0])
    for i in range(real_pos_x.shape[0]):
        a = env._NavigateTo_policy(state, {}, navigate_objects, X[i, -2:]).arr
        real_pos_x[i] = a[0]
        real_pos_y[i] = a[1]
        a = env._NavigateTo_policy(state, {}, navigate_objects, Xhat[i]).arr
        sample_pos_x[i] = a[0]
        sample_pos_y[i] = a[1]
    x, y = navigate_params
    env.render_state_plt(state, t[0], caption='right')
    plt.scatter(real_pos_x[(Y==1).squeeze()], real_pos_y[(Y==1).squeeze()], c='blue', zorder=-10)
    plt.scatter(real_pos_x[(Y==0).squeeze()], real_pos_y[(Y==0).squeeze()], c='red', zorder=-10)
    plt.scatter(sample_pos_x, sample_pos_y, c='green', alpha=0.3, zorder=-5)
    plt.show()

if __name__ == '__main__':
    test_init_sampler()
    # test_navigate_bookshelf_action()
    # test_navigate_book_action()
    # test_navigate_pick_place()
    # test_ebm()
