"""A Kitchen environment wrapping robosuite kitchen."""
import copy
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, cast

import numpy as np
from gym.spaces import Box
import robosuite
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers import VisualizationWrapper
import robocasa.macros as macros
from robocasa.environments import ALL_KITCHEN_ENVIRONMENTS
from robocasa.utils.env_utils import create_env

from predicators import utils
from predicators.envs import BaseEnv
from predicators.settings import CFG
from predicators.structs import Action, EnvironmentTask, Image, Object, \
    Observation, Predicate, State, Type, Video
import matplotlib
from collections import OrderedDict
from termcolor import colored
import warnings
import os
import mujoco

# Constants from demo files
MAX_CARTESIAN_DISPLACEMENT = 0.2
MAX_ROTATION_DISPLACEMENT = 0.5

class RoboKitchenEnv(BaseEnv):
    """Kitchen environment using robosuite."""

    at_pre_turn_atol = 0.1  # tolerance for AtPreTurnOn/Off
    ontop_atol = 0.18  # tolerance for OnTop
    on_angle_thresh = -0.28  # -0.4  # dial is On if less than this threshold
    light_on_thresh = -0.39  # light is On if less than this threshold
    microhandle_open_thresh = -0.65
    hinge_open_thresh = 0.084
    cabinet_open_thresh = 0.02
    at_pre_pushontop_yz_atol = 0.1  # tolerance for AtPrePushOnTop
    at_pre_pullontop_yz_atol = 0.04  # tolerance for AtPrePullOnTop
    at_pre_pushontop_x_atol = 1.0  # other tolerance for AtPrePushOnTop

    gripper_open_thresh = 0.038 # m 
    gripper_closed_thresh = 0.02 # m
    close_distance_thresh = 0.02

    # Types (similar to original kitchen)
    handle_type = Type("handle_type", ["x", "y", "z", "qx", "qy", "qz", "qw"])
    gripper_type = Type("gripper_type", ["x", "y", "z", "qx", "qy", "qz", "qw", "angle"])
    hinge_type = Type("hinge_type", ["angle"])
    object_type = Type("object_type", ["x", "y", "z", "qx", "qy", "qz", "qw"])
    base_type = Type("base_type", ["x", "y", "z", "qx", "qy", "qz", "qw"])

    obj_name_to_type = {
        "handle": handle_type,
        "gripper": gripper_type,
        "hinge": hinge_type,
        "robot0_base": base_type,
    }

    tasks_extended = ['Lift', 'Stack', 'NutAssembly', 'NutAssemblySingle', 'NutAssemblySquare', 'NutAssemblyRound', 
                       'PickPlace', 'PickPlaceSingle', 'PickPlaceMilk', 'PickPlaceBread', 'PickPlaceCereal', 'PickPlaceCan', 
                       'Door', 'Wipe', 'ToolHang', 'TwoArmLift', 'TwoArmPegInHole', 'TwoArmHandover', 'TwoArmTransport', 'Kitchen',
                         'KitchenDemo', 'CupcakeCleanup', 'OrganizeBakingIngredients', 'PastryDisplay', 'FillKettle', 'HeatMultipleWater',
                           'VeggieBoil', 'ArrangeTea', 'KettleBoiling', 'PrepareCoffee', 'ArrangeVegetables', 'BreadSetupSlicing', 
                           'ClearingTheCuttingBoard', 'MeatTransfer', 'OrganizeVegetables', 'BowlAndCup', 'CandleCleanup', 
                           'ClearingCleaningReceptacles', 'CondimentCollection', 'DessertAssembly', 'DrinkwareConsolidation', 
                           'FoodCleanup', 'DefrostByCategory', 'MicrowaveThawing', 'QuickThaw', 'ThawInSink', 'AssembleCookingArray', 
                           'FryingPanAdjustment', 'MealPrepStaging', 'SearingMeat', 'SetupFrying', 'BreadSelection', 'CheesyBread', 
                           'PrepareToast', 'SweetSavoryToastSetup', 'PrepForTenderizing', 'PrepMarinatingMeat', 'ColorfulSalsa', 
                           'SetupJuicing', 'SpicyMarinade', 'HeatMug', 'MakeLoadedPotato', 'SimmeringSauce', 'WaffleReheat', 
                           'WarmCroissant', 'BeverageSorting', 'RestockBowls', 'RestockPantry', 'StockingBreakfastFoods', 
                           'CleanMicrowave', 'CountertopCleanup', 'PrepForSanitizing', 'PushUtensilsToSink', 'DessertUpgrade',
                             'PanTransfer', 'PlaceFoodInBowls', 'PrepareSoupServing', 'ServeSteak', 'WineServingPrep', 
                             'ArrangeBreadBasket', 'BeverageOrganization', 'DateNight', 'SeasoningSpiceSetup', 
                             'SetBowlsForSoup', 'SizeSorting', 'BreadAndCheese', 'CerealAndBowl', 'MakeFruitBowl', 
                             'VeggieDipPrep', 'YogurtDelightPrep', 'MultistepSteaming', 'SteamInMicrowave', 'SteamVegetables', 
                             'ManipulateDrawer', 'OpenDrawer', 'CloseDrawer', 'DrawerUtensilSort', 'OrganizeCleaningSupplies', 
                             'PantryMishap', 'ShakerShuffle', 'SnackSorting', 'DryDishes', 'DryDrinkware', 'PreSoakPan', 'SortingCleanup', 
                             'StackBowlsInSink', 'AfterwashSorting', 'ClearClutter', 'DrainVeggies', 'PrewashFoodAssembly', 'PnPCoffee', 
                             'CoffeeSetupMug', 'CoffeeServeMug', 'CoffeePressButton', 'ManipulateDoor', 'OpenDoor', 'OpenSingleDoor', 
                             'OpenDoubleDoor', 'CloseDoor', 'CloseSingleDoor', 'CloseDoubleDoor', 'MicrowavePressButton', 'TurnOnMicrowave',
                               'TurnOffMicrowave', 'NavigateKitchen', 'PnP', 'PnPCounterToCab', 'PnPCabToCounter', 'PnPCounterToSink', 
                               'PnPSinkToCounter', 'PnPCounterToMicrowave', 'PnPMicrowaveToCounter', 'PnPCounterToStove', 'PnPStoveToCounter', 
                               'ManipulateSinkFaucet', 'TurnOnSinkFaucet', 'TurnOffSinkFaucet', 'TurnSinkSpout', 'ManipulateStoveKnob',
                                 'TurnOnStove', 'TurnOffStove']

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        print(f"ALL_KITCHEN_ENVIRONMENTS: {ALL_KITCHEN_ENVIRONMENTS}")

        if self._using_gui:
            pass
            # assert not CFG.make_test_videos or CFG.make_failure_videos, \
            #     "Turn off --use_gui to make videos in robo kitchen env"

        self._pred_name_to_pred = self.create_predicates()
        self._env = None  # Will be created in reset
        self._env_raw = None
        self.task_selected = CFG.robo_kitchen_task
        if self.task_selected not in self.tasks_extended:
            raise ValueError(f"Task {self.task_selected} not supported")
        print(colored(f"Selected task: {self.task_selected}", "green"))

    def get_objects_of_interest(self, task_name: str) -> List[Object]:
        """Get the object of interest for the task."""
        if task_name == "OpenSingleDoor":
            return [self.object_name_to_object("handle"), 
                    self.object_name_to_object("door")]
        # by default, there are robot and gripper objects
        else:
            raise ValueError(f"Task {task_name} not supported")

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        """Create tasks for training."""
        using_recorded_data = True
        if using_recorded_data:
            tasks = []

            # Get the demo dataset path
            from robocasa.utils.dataset_registry import get_ds_path
            dataset_path = get_ds_path(self.task_selected, ds_type="human_raw")

            if dataset_path is None or not os.path.exists(dataset_path):
                print(colored(f"Unable to find dataset for {self.task_selected}. Downloading...", "yellow"))
                from robocasa.scripts.download_datasets import download_datasets
                download_datasets(tasks=[self.task_selected], ds_types=["human_raw"])
                dataset_path = get_ds_path(self.task_selected, ds_type="human_raw")

            # Load the demos
            import h5py
            f = h5py.File(dataset_path, "r")
            demos = list(f["data"].keys())

            # Sort demos by index
            inds = np.argsort([int(elem[5:]) for elem in demos])
            demos = [demos[i] for i in inds]

            # Create tasks from each demo
            for task_idx in range(CFG.num_train_tasks):
                if task_idx >= len(demos):
                    break

                # Get demo data
                demo = f[f"data/{demos[task_idx]}"]
                # Get initial state info from first timestep of datagen_info
                initial_state = {}
                for key in demo["datagen_info"].keys():
                    initial_state[key] = demo["datagen_info"][key][0]
                initial_state["model"] = demo.attrs["model_file"]
                initial_state["ep_meta"] = demo.attrs.get("ep_meta", None)

                # Create observation
                obs = {
                    "state_info": initial_state,
                    "obs_images": []
                }

                # Get goal description from task name
                goal_description = self.task_selected

                # Create task
                task = EnvironmentTask(obs, goal_description)
                tasks.append(task)

            f.close()
            return tasks
        else:   
            return self._get_tasks(num=CFG.num_train_tasks, train_or_test="train")

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        """Create tasks for testing."""
        return self._get_tasks(num=CFG.num_test_tasks, train_or_test="test")

    def _get_tasks(self, num: int, train_or_test: str) -> List[EnvironmentTask]:
        """Create a list of tasks"""
        tasks = []

        for task_idx in range(num):
            # For now just use OpenSingleDoor as the default task
            task_name = self.task_selected
            #check if task_name is in available_tasks
            if task_name not in self.tasks_extended:
                raise ValueError(f"Task {task_name} not supported")
            goal_description = task_name
            seed = task_idx
            
            # Get initial observation
            init_obs = self._reset_initial_state(seed, train_or_test, task_name)
            # let's not do that since we are not using reset from initial state
            # init_obs = {}
            task = EnvironmentTask(init_obs, goal_description)
            tasks.append(task)

        return tasks

    def goal_reached(self) -> bool:
        #print angle of handle
        state = self.state_info_to_state(
            self._current_observation["state_info"])
        hinge_angle = state.get(self.object_name_to_object("hinge"), "angle")

        # print(f"Hinge angle: {hinge_angle}")

        goal_desc = self._current_task.goal_description

        if goal_desc == "OpenSingleDoor":
            if hinge_angle > 0.9:
                return True
        else:
            return False
        


    def _reset_initial_state(self, seed: int, train_or_test: str, task_name: str) -> Observation:
        """Reset the environment to an initial state based on the seed."""
        # Create or recreate environment if needed
        warnings.warn("Resetting environment to initial state from seed not implemented for robosuite kitchen")
        if self._env is None:
            complex_config = False # easy config allows seed
            if complex_config:
                robot_type = "PandaOmron"
                controller_config = load_composite_controller_config(robot=robot_type)

                config = {
                    "env_name": task_name,
                    "robots": robot_type,
                    "controller_configs": controller_config,
                    "layout_ids": 0,
                    "style_ids": 0,
                    "translucent_robot": True,
                }

                print(colored(f"Initializing environment for task: {task_name}", "yellow"))

                self._env_raw = robosuite.make(
                    **config,
                    has_renderer=self._using_gui,
                    has_offscreen_renderer=False,
                    render_camera="robot0_frontview",
                    ignore_done=True,
                    use_camera_obs=False,
                    control_freq=20,
                    renderer="mjviewer", 
                )

                self._env = VisualizationWrapper(self._env_raw)
                self.ep_meta = self._env.get_ep_meta()
            else:
                print(f"Creating env for task: {task_name}, seed: {seed}, gui: {self._using_gui}")
                self._env = create_env(
                    env_name = task_name,
                    render_onscreen = self._using_gui,
                    seed = seed,
                )

        # Reset environment with seed
        obs = self._env.reset()
        
        # Update objects of interest based on task
        self.objects_of_interest = self.get_objects_of_interest(task_name)

        # Get contact information
        contact_set = self.get_object_level_contacts()

        # Return observation
        return {
            "state_info": obs,
            "obs_images": [],
            "contact_set": contact_set
        }

    def get_object_level_contacts(self) -> set[Tuple[Object, Object]]:
        """Get all contacts between objects in the environment, default to have robot and gripper, in addition to the objects of interest
        this has to be a method not a class method since we need env access """

        # only support panda robot for now
        contacts = set()
        # robot_contacts = self._env.get_contacts(self._env.robots[0].robot_model.models[0])
        gripper_contact = self._env.get_contacts(self._env.robots[0].robot_model.models[1])
        # filter down to only include objects of interest

        object_names = [obj.name for obj in self.objects_of_interest]
        # robot_obj = self.object_name_to_object("robot")
        gripper_obj = self.object_name_to_object("gripper")

        # for contact in robot_contacts: # each contact is a string
        #     for obj_name in object_names:
        #         if obj_name in contact:
        #             obj = self.object_name_to_object(obj_name)
        #             contacts.add((obj, robot_obj))
        for contact in gripper_contact:
            for obj_name in object_names:
                if obj_name in contact:
                    obj = self.object_name_to_object(obj_name)
                    contacts.add((obj, gripper_obj))
        # Filter out gripper-door contact if gripper-handle contact exists
        contacts = self._filter_door_handle_contacts(contacts, gripper_obj)
        return contacts

    def _filter_door_handle_contacts(self, contacts: set[Tuple[Object, Object]], gripper_obj: Object) -> set[Tuple[Object, Object]]:
        """Filter out gripper-door contact if gripper-handle contact exists."""
        handle_obj = self.object_name_to_object("handle")
        door_obj = self.object_name_to_object("door")
        gripper_handle_contact = (handle_obj, gripper_obj) in contacts or (gripper_obj, handle_obj) in contacts
        if gripper_handle_contact:
            contacts = {c for c in contacts if door_obj not in c}
        return contacts

    @classmethod
    def get_name(cls) -> str:
        return "robo_kitchen"

    @classmethod
    def create_predicates(cls) -> Dict[str, Predicate]:
        """Exposed for perceiver."""
        preds = {
            Predicate("ReadyGrabHandle", [cls.gripper_type, cls.handle_type], cls._ReadyGrabHandle_holds),
            Predicate("GripperOpen", [cls.gripper_type], cls._GripperOpen_holds),
            Predicate("GripperClosed", [cls.gripper_type], cls._GripperClosed_holds),
            Predicate("HingeOpen", [cls.hinge_type], cls._HingeOpen_holds),
            Predicate("HingeClosed", [cls.hinge_type], cls._HingeClosed_holds),
            Predicate("InContact", [cls.gripper_type, cls.handle_type], cls._InContact_holds),
        }

        return {p.name: p for p in preds}

    def simulate(self, state: State, action: Action) -> State:
        """Get next state from current state and action."""
        # Implement simulation logic
        raise NotImplementedError("Simulate not implemented for robosuite kitchen")

    def step(self, action: Action) -> Observation:
        """Execute action and return observation.
        
        Convert 7D predicators action [dx, dy, dz, droll, dpitch, dyaw, gripper]
        to 12D robocasa action [right_pose(6), right_gripper(1), base(3), torso(1), extra(1)]
        """
        # Debug print
        # print("\n" + "="*50)
        # print("STEP DEBUG INFO:")
        # print(f"Door state: {self._env_raw.door_fxtr.get_door_state(env=self._env_raw)}")
        # print(f"Action: {action.arr}")
        # print("="*50 + "\n")
        # Scale the action
        pos_delta = action.arr[:3] * MAX_CARTESIAN_DISPLACEMENT
        rot_delta = action.arr[3:6] * MAX_ROTATION_DISPLACEMENT
        gripper_cmd = action.arr[6]

        # Create 12D robocasa action:
        # - First 6D: right arm pose (position + rotation)
        # - Next 1D: right gripper
        # - Next 3D: base (no movement)
        # - Next 1D: torso (no movement)
        # - Last 1D: extra dimension (not used)
        env_action = np.zeros(12, dtype=np.float32)
        env_action[0:3] = pos_delta  # position control
        env_action[3:6] = rot_delta  # rotation control
        env_action[6] = gripper_cmd  # gripper control
        # env_action[7:10] are zeros (no base movement)
        # env_action[10] is zero (no torso movement)
        # env_action[11] is zero (extra dimension)

        # Execute action in environment (Robosuite:Mujoco Env)
        obs, _, _, _ = self._env.step(env_action)
        # self._add_debug_visualization() # not working!

        contact_set = self.get_object_level_contacts()

        observation = {
            "state_info": obs,
            "obs_images": [],
            "contact_set": contact_set
        }

        self._current_observation = observation
        return self._copy_observation(self._current_observation)

    def reset(self, train_or_test: str, task_idx: int) -> Observation:
        """Reset environment to initial state for the given task."""
        self._current_task = self.get_task(train_or_test, task_idx)
        task_name = self._current_task.goal_description
        warnings.warn("Resetting environment to initial state from not implemented, just reset the env")
        self._current_observation = self._reset_initial_state(
            seed = task_idx,
            train_or_test = train_or_test,
            task_name = task_name
        )   
        return self._copy_observation(self._current_observation)
    def render(self, action: Optional[Action] = None, # this renders the robot observation, not the viewer??
              caption: Optional[str] = None) -> Video:
        """Render current state."""
        return self._env.render()

    @property
    def action_space(self) -> Box:
        """7D action space: [dx, dy, dz, droll, dpitch, dyaw, gripper]"""
        return Box(-1.0, 1.0, (7,), np.float32)

    @property
    def goal_predicates(self) -> Set[Predicate]:
        """Get the subset of self.predicates that are used in goals."""
        goal_preds = {self._pred_name_to_pred["HingeOpen"], self._pred_name_to_pred["HingeClosed"]}
        return goal_preds

    @property
    def predicates(self) -> Set[Predicate]:
        """Get the set of predicates that are given with this environment."""
        # Initialize predicates similar to kitchen.py

        return set(self._pred_name_to_pred.values())

    @property
    def types(self) -> Set[Type]:
        """Get the set of types that are given with this environment."""
        return {
            self.object_type, 
            self.hinge_type,
            self.gripper_type,
            self.handle_type,
            self.base_type,
        }

    def get_observation(self) -> Observation:
        return self._copy_observation(self._current_observation)

    def _copy_observation(self, obs: Observation) -> Observation:
        """Create copy of observation."""
        return copy.deepcopy(obs)

    def render_state_plt(
        self,
        state: State,
        task: EnvironmentTask,
        action: Optional[Action] = None,
        caption: Optional[str] = None) -> matplotlib.figure.Figure:
        raise NotImplementedError("This env does not use Matplotlib")
    # Helper methods needed by predicates

    # def get_object_centric_state_info(self) -> Dict[str, Any]:
    #     """Parse State into Object Centric State."""
    #     mujoco_model = self._gym_env.model  # type: ignore
    #     mujoco_data = self._gym_env.data  # type: ignore
    #     mujoco_model_names = self._gym_env.robot_env.model_names  # type: ignore
    #     state_info = {}
    #     for site in _TRACKED_SITES:
    #         state_info[site] = get_site_xpos(mujoco_model, mujoco_data,
    #                                          site).copy()
    #         # Include rotation for gripper.
    #         if site == "EEF":
    #             xmat = get_site_xmat(mujoco_model, mujoco_data, site).copy()
    #             quat = mat2quat(xmat)
    #             state_info[site] = np.concatenate([state_info[site], quat])
    #     for joint in _TRACKED_SITE_TO_JOINT.values():
    #         state_info[joint] = get_joint_qpos(mujoco_model, mujoco_data,
    #                                            joint).copy()
    #     for body in _TRACKED_BODIES:
    #         body_id = mujoco_model_names.body_name2id[body]
    #         state_info[body] = mujoco_data.xpos[body_id].copy()
    #     return state_info

    @classmethod
    def object_name_to_object(cls, obj_name: str) -> Object:
        """Made public for perceiver."""
        if obj_name in cls.obj_name_to_type:
            return Object(obj_name, cls.obj_name_to_type[obj_name])
        else:
            return Object(obj_name, cls.object_type)

    @classmethod
    def state_info_to_state(cls, state_info: Dict[str, Any], contact_set: set[Tuple[Object, Object]] = None) -> State:
        state_dict = {}
        for key, val in state_info.items():            
            if key.endswith("_pos_quat_angle"):
                obj_name = key[:-15]
                obj = cls.object_name_to_object(obj_name)
                state_dict[obj] = {
                    "x": val[0],
                    "y": val[1],
                    "z": val[2],
                    "qx": val[3],
                    "qy": val[4],
                    "qz": val[5],
                    "qw": val[6],
                    "angle": val[7]
                }
            elif key.endswith("_pos_quat"):
                obj_name = key[:-9]
                obj = cls.object_name_to_object(obj_name)
                state_dict[obj] = {
                    "x": val[0],
                    "y": val[1],
                    "z": val[2],
                    "qx": val[3],
                    "qy": val[4],
                    "qz": val[5],
                    "qw": val[6]
                }
            elif key.endswith("_angle"):
                obj_name = key[:-6]
                obj = cls.object_name_to_object(obj_name)
                state_dict[obj] = {
                    "angle": val #currently only support 1 door, double door doesn't work
                }
            elif key.endswith("_quat"):
                obj_name = key[:-5]  # Remove _pos
                pos_val = state_info[key[:-5] + "_pos"]
                obj = cls.object_name_to_object(obj_name)
                state_dict[obj] = { # robosuite is xyzw
                    "x": pos_val[0],
                    "y": pos_val[1],
                    "z": pos_val[2],
                    "qx": val[0],
                    "qy": val[1],
                    "qz": val[2],
                    "qw": val[3],
                }

        state = utils.create_state_from_dict(state_dict)
        state.simulator_state = {}
        state.items_in_contact = contact_set # when defaults, it means Not populated, when empty means no contact
        return state

    @classmethod
    def _ReadyGrabHandle_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        """Check if gripper is ready to grip handle."""
        gripper, handle = objects
        # Check if gripper is open
        if not state.get(gripper, "angle") > cls.gripper_open_thresh:
            print ("ReadyGrabHandle Not True: Gripper not open")
            return False
        # Check if position of gripper is close to handle
        gripper_pos = np.array([state.get(gripper, "x"), state.get(gripper, "y"), state.get(gripper, "z")])
        handle_pos = np.array([state.get(handle, "x"), state.get(handle, "y"), state.get(handle, "z")])
        if np.linalg.norm(gripper_pos - handle_pos) > cls.close_distance_thresh:
            print ("ReadyGrabHandle Not True: Gripper not close enough to handle")
            return False
        # Check if orientation of gripper is close to handle
        return True

    @classmethod
    def _GripperOpen_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        """Made public for use in ground-truth options."""
        obj = objects[0]
        if obj.is_instance(cls.gripper_type):
            return state.get(obj, "angle") > cls.gripper_open_thresh
        return False

    @classmethod
    def _GripperClosed_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        """Made public for use in ground-truth options."""
        obj = objects[0]
        if obj.is_instance(cls.gripper_type):
            return state.get(obj, "angle") <= cls.gripper_closed_thresh
        return False

    @classmethod
    def _HingeOpen_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        """Made public for use in ground-truth options."""
        obj = objects[0]
        if obj.is_instance(cls.hinge_type):
            return state.get(obj, "angle") > cls.hinge_open_thresh
        return False

    @classmethod
    def _HingeClosed_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        """Made public for use in ground-truth options."""
        obj = objects[0]
        if obj.is_instance(cls.hinge_type):
            return state.get(obj, "angle") <= cls.hinge_open_thresh
        return False

    @classmethod
    def _InContact_holds(cls, 
                         state: State, 
                         objects: Sequence[Object]) -> bool:
        """Check if two objects are in contact using robosuite's contact checking."""
        obj1, obj2 = objects
        return (obj1, obj2) in state.items_in_contact or (obj2, obj1) in state.items_in_contact

    def _add_debug_visualization(self):
        """Add debug visualization markers at important locations."""
        # Get the viewer from the simulation
        viewer = self._env.viewer
        if viewer is None:
            return

        # Clear existing visualizations
        viewer.user_scn.ngeom = 0
        geom_count = 0

        # Add visualization for each object's important sites/geoms
        for obj_name, obj in self.objects.items():
            # Get object position and orientation
            obj_pos = sim.data.body_xpos[self.obj_body_id[obj_name]]

            # Create a sphere at object position
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[geom_count],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.02, 0, 0],  # Small sphere
                pos=obj_pos,
                mat=np.eye(3).flatten(),
                rgba=[1, 0, 0, 0.5]  # Semi-transparent red
            )
            geom_count += 1

            # Add more visualizations for specific object types
            if obj_name in ["microwave", "cabinet", "drawer"]:
                # Add handle visualization
                handle_site_id = sim.model.site_name2id(f"{obj_name}_handle")
                if handle_site_id >= 0:
                    handle_pos = sim.data.site_xpos[handle_site_id]
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[geom_count],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[0.015, 0, 0],
                        pos=handle_pos,
                        mat=np.eye(3).flatten(),
                        rgba=[0, 1, 0, 0.5]  # Semi-transparent green
                    )
                    geom_count += 1

        # Update the number of visualization geoms
        viewer.user_scn.ngeom = geom_count
        viewer.sync()
