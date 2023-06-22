from mujoco_kitchen.utils import make_env, primitive_and_params_to_primitive_action, OBS_ELEMENT_INDICES
import numpy as np

##

env = make_env("kitchen", "microwave", {"usage_kwargs": {"max_path_length": 50, "use_raw_action_wrappers": False, "unflatten_images": False}})

# Display Useful Information
print("#"*30)
print("Env", env)
print("Primitive Funcs", env.primitive_name_to_func.keys())
print("Primitive Index -> Names", env.primitive_idx_to_name)
print("Primitive Names -> Action Index", env.primitive_name_to_action_idx)
print("Number of Parameters", env.max_arg_len)
print("Number of Primitives", env.num_primitives)
print("Action Space", env.action_space)
print("#"*30)

env.reset()
for _ in range(100):
    env.render()
    # Move to Kettle
    action = primitive_and_params_to_primitive_action('move_delta_ee_pose', env.get_site_xpos("kettle_site") - env.get_site_xpos("end_effector")) #env.action_space.sample()

    # Parse Action to Primitive and Parameters
    # Only needed for printing (This is done in env)
    primitive_idx, primitive_args = (
        np.argmax(action[: env.num_primitives]),
        action[env.num_primitives :],
    )
    primitive_name = env.primitive_idx_to_name[primitive_idx]
    parameters = None
    for key, val in env.primitive_name_to_action_idx.items():
        if key == primitive_name:
            if type(val) ==  int:
                parameters = [primitive_args[val]]
            else:
                parameters = [primitive_args[i] for i in val]
    assert parameters is not None
    print(primitive_name, parameters, "\n")

    state, reward, done, info = env.step(action)

    # Parse State into Object Centric State
    for key, val in OBS_ELEMENT_INDICES.items():
        print(key, [state[i] for i in val])
    print()
    important_sites = ["hinge_site1", "hinge_site2", "kettle_site", "microhandle_site", "knob1_site", "knob2_site", "knob3_site", "knob4_site", "light_site", "slide_site", "end_effector"]
    for site in important_sites:
        print(site, env.get_site_xpos(site))
        # Potentially can get this ^ from state

    if done:
        print("Done")
        break