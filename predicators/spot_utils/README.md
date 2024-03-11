# Spot Utils

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
associated with a map. Populate this with details such as a `spot-home-pose`, etc.
See `predicators/spot_utils/graph_nav_maps/floor8-v2/metadata.yaml` for an example and
explanation(s) of the various fields in the metadata file.
5. Set --spot_graph_nav_map to your new env name.

### Obtaining points for the `allowed-regions` in the metadata
A challenging thing for the metadata is to define the points that yield `allowed-regions`.
The following workflow is one way to make this relatively easy.

1. Run [this script](https://github.com/boston-dynamics/spot-sdk/tree/master/python/examples/graph_nav_extract_point_cloud) on the pre-made
map to yield an output `.ply` pointcloud file.
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

Note that BDAI is currently transitioning to using `torchserve`, so we likely need to update the client code a bit. 

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
