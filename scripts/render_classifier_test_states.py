"""Script to render images of the states used in testing interactively learned
predicate classifiers."""


from predicators.scripts.evaluate_interactive_approach_classifiers import \
    create_states_cover
from predicators.src import utils
from predicators.src.envs import create_new_env
from predicators.src.envs.cover import CoverEnv
from predicators.src.settings import CFG

if __name__ == "__main__":
    args = utils.parse_args()
    utils.update_config(args)
    env = create_new_env(CFG.env, do_cache=True)
    task = env.get_test_tasks()[0]
    if CFG.env == "cover":
        assert isinstance(env, CoverEnv)
        states, _, _ = create_states_cover(env)
    else:
        raise NotImplementedError(f"No implementation yet for {CFG.env}")
    for i, s in enumerate(states):
        img = env.render_state(s, task)  # task is unused
        outfile = f"{CFG.env}__test_state_{i+1}.png"
        utils.save_video(outfile, img)
        print(f"Wrote image out to {outfile}")
