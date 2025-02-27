# Spot Utils

## How to run your own Spot environment

> Last Updated: 02/27/2025

**Steps:**
- Set up the codebase, perception pipeline, and Spot
  - You need to have access to a GPU server for the perception pipeline (e.g., the Detic-SAM pipeline)
  - You need to connect to Spot (through WiFi or ethernet cable). The Spot at LIS uses its own WiFi AP mode and is on IP `192.168.80.3`.
  - To connect to both Spot and GPU server, our current solution is to use WiFi for Spot and ethernet cable for the GPU server.
- If you want the spot to autonomously execute movement skills, then create a new map of the environment: See the `Mapping` section.
  - Prepare the metadata file: See the `Prepare Metadata` section.
- Otherwise, you can run a ''minimal'' environment where a human is asked to teleop the skills instead of having spot execute them automatically. This can be helpful for debugging new functionality, etc.
- Implement your task
- Start actual run. Examples:
```
# template with map (autonomous spot skills)
python predicators/main.py --spot_robot_ip <spot_ip> --spot_graph_nav_map <map_name> --env <env_name> --approach "spot_wrapper[oracle]"

# template without map (human teleop skills)
python predicators/main.py --env <env_name> --approach "spot_wrapper[oracle]" --spot_robot_ip <spot_ip> --perceiver spot_minimal_perceiver 

# an example to run LIS Spot with a room map
python predicators/main.py --spot_robot_ip 192.168.80.3 --spot_graph_nav_map b45-621 --env lis_spot_block_floor_env --approach spot_wrapper[oracle] --bilevel_plan_without_sim True --seed 0

# an example to run LIS spot without a map
python predicators/main.py --env spot_vlm_cup_table_env --approach "spot_wrapper[oracle]" --seed 0 --num_train_tasks 0 --num_test_tasks 1 --spot_robot_ip 192.168.80.3 --perceiver spot_minimal_perceiver --bilevel_plan_without_sim True --vlm_test_time_atom_label_prompt_type img_option_diffs_label_history --vlm_model_name gpt-4o --execution_monitor expected_atoms
```

### Implement Your Task

To create a simple task (that uses a map) before you can run, you only need to:

- Create a new environment in `envs/spot_envs.py`
  - In `spot_env.py`, subclass the `SpotRearrangementEnv` and define the necessary methods needed to override.
  - The simplest possible example is `SpotSodaFloorEnv` (and maybe you can directly use this!).
  - Doing this involves selecting some operators that you'll need.
- Add ground truth model
  - Add environment name into `SpotEnvsGroundTruthNSRTFactory`  in `ground_truth_models/spot_env/nsrt.py`
  - Add environment name into `SpotEnvsGroundTruthOptionFactory` in `ground_truth_models/spot_env/options.py`
- If you want, define a new `goal_description` string. Then, go to the _`create_goal` function of `spot_perceiver.py` and follow the example to convert a goal description string into an actual set of atoms needed to implement the goal.

To create a new task without using a map (and using human teleop skills), take a look at the `SimpleVLMCupEnv` in `spot_env.py` and make something similar.


## Mapping
> Last Updated: 11/14/2023

Our code is currently designed to operate given a saved map of a particular
environment. The map is required to define a coordinate system that persists
between robot runs. Moreover, each map contains associated metadata information
that we use for visualization, collision checking, and a variety of other
functions.

To create a new map of a new environment:
1. Print out and tape april tags around the environment. The tags are [here](https://support.bostondynamics.com/s/article/About-Fiducials)
2. Run the interactive script from the spot SDK to create a map, while walking
   the spot around the environment. The script is [here](https://github.com/boston-dynamics/spot-sdk/blob/master/python/examples/graph_nav_command_line/recording_command_line.py)   
3. Save the map files to spot_utils / graph_nav_maps / <your new env name>
4. Create a file named `metadata.yaml` if one doesn't already exist within the folder
associated with a map. See below for more details.
5. Set --spot_graph_nav_map to your new env name.


### Prepare Metadata

The metadata file is a yaml file that contains information about the map and is used by the codebase to make decisions about the environment. 
See `predicators/spot_utils/graph_nav_maps/floor8-v2/metadata.yaml` or `predicators/spot_utils/graph_nav_maps/floor8-sweeping/metadata.yaml` for an example and
explanation(s) of the various fields in the metadata file.

**Specifying the following required fields**

- `spot-home-pose`: a place in the room from which most of the room is visible and the robot can execute its object finding procedure.
- `allowed-regions`: these are (x,y) points that define 4 corners of a region that the robot will be allowed to be in. This is to prevent it from trying to navigate into a wall, or outside a door. In the case of 621, you should basically just put in the 4 corners of the room. _See below for a note._
- `known-immovable-objects`. These are the x, y, z positions of objects that the robot cannot manipulate (e.g. the floor). You'll probably want to add the floor and or any big tables in the room
- `static-object-features`. These are some hand-defined features for various objects that you might want to use (e.g., the object shape, width, length, height, etc.).



**Obtaining points for the `allowed-regions` in the metadata**

A challenging thing for the metadata is to define the points that yield `allowed-regions`.
The following workflow is one way to make this relatively easy.

1. Run [this script](https://github.com/boston-dynamics/spot-sdk/tree/master/python/examples/graph_nav_extract_point_cloud) on the pre-made map to yield an output `.ply` pointcloud file.
2. Install the [Open3D package](http://www.open3d.org/docs/release/getting_started.html) with `pip install open3d`.
3. Open up a python interpreter in your terminal, and run the following commands:
```
import open3d as o3d

pcd = o3d.io.read_point_cloud("<path to your pointcloud file>")
o3d.visualization.draw_geometries_with_editing([pcd])
```
4. Within the window, do SHIFT + left click on a point to print out its 3D coords to the terminal. Copy the first two (x, y) of all the
boundary points into the yaml. Note that you can do SHIFT + right click to unselect a point.

## Perception

> Last updated: 07/17/2023

We are currently using Detic + Segmentation Anything to provide text-conditioned bounding box + detection-conditioned segmentation mask.

We currently use REST interface from the [DETIC SAM BDAI repo](https://github.com/bdaiinstitute/detic-sam/).

The pipeline is as follows:
- Set up the repo
  - See the instructions in the above-linked repo. Note that you need to use the branch listed above.
- Run the server by running `server.py` from the above-linked repo
  - You only need to run the server once
  - It is recommended to run on a local computer (faster connection) with CUDA GPU (faster inference)
- Connect to server from local
  - Use SSH "local port forward"
  - `ssh -L 5550:localhost:5550 10.17.1.102`
- Request from your local computer
  - You can see perception_utils.py, or the `client.py` function in the BDAI repo.

## Simulation

> Last updated: 03/07/2024

Setting the `--bilevel_plan_without_sim` flag to `False` (which is the default value) will attempt to do full planning in Pybullet and then execution in the real world.
Importantly, this relies on having object models (urdfs, meshes, etc.) corresponding to each object that Spot needs to be aware of/manipulate in the world.
To add a new object model, add a new urdf and any relevant files to `envs/assets/urdf/`. Ensure that the urdf is named `<obj_name>.urdf` where `<obj_name>` is the name of the predicators `Object` that you want to instantiate corresponding to this urdf. 

The design pattern for interfacing with the simulator is that options have both a "real world" function implementing the relevant action on a Spot robot, and a "simulated" function that intends to mimic the real world behavior in the simulator. See the `_move_to_target_policy` function in `options.py` for the spot env for an example. Importantly, the simulated function should directly modify the simulation from within it, just like how the real world functions directly modify the real world by commanding the robot when invoked. 
