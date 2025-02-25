"""A Kitchen environment wrapping robosuite kitchen."""
import copy
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, cast

import numpy as np
from gym.spaces import Box
import robosuite
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers import VisualizationWrapper
import robocasa.macros as macros

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
    # Types (similar to original kitchen)
    # object_type = Type("object", ["x", "y", "z"])
    rich_object_type = Type("rich_object_type", ["x", "y", "z", "qx", "qy", "qz", "qw"])
    hinge_door_type = Type("hinge_door_type", ["angle"])
    # on_off_type = Type("on_off", ["x", "y", "z", "angle"], parent=object_type)
    # hinge_door_type = Type("hinge_door", ["x", "y", "z", "angle"],
    #                        parent=on_off_type)
    # knob_type = Type("knob", ["x", "y", "z", "angle"], parent=on_off_type)
    # switch_type = Type("switch", ["x", "y", "z", "angle"], parent=on_off_type)
    # surface_type = Type("surface", ["x", "y", "z"], parent=object_type)
    # kettle_type = Type("kettle", ["x", "y", "z"], parent=object_type)

    tasks = OrderedDict(
        [
            ("PnPCounterToCab", "pick and place from counter to cabinet"),
            ("PnPCounterToSink", "pick and place from counter to sink"),
            ("PnPMicrowaveToCounter", "pick and place from microwave to counter"),
            ("PnPStoveToCounter", "pick and place from stove to counter"),
            ("OpenSingleDoor", "open cabinet or microwave door"),
            ("CloseDrawer", "close drawer"),
            ("TurnOnMicrowave", "turn on microwave"),
            ("TurnOnSinkFaucet", "turn on sink faucet"),
            ("TurnOnStove", "turn on stove"),
            ("ArrangeVegetables", "arrange vegetables on a cutting board"),
            ("MicrowaveThawing", "place frozen food in microwave for thawing"),
            ("RestockPantry", "restock cans in pantry"),
            ("PreSoakPan", "prepare pan for washing"),
            ("PrepareCoffee", "make coffee"),
        ]
    )
    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        if self._using_gui:
            pass
            # assert not CFG.make_test_videos or CFG.make_failure_videos, \
            #     "Turn off --use_gui to make videos in robo kitchen env"

        robot_type = "PandaOmron"
        # Create robosuite environment
        controller_config = load_composite_controller_config(robot=robot_type)

        # Create argument configuration
        tasks_names = ['Lift', 'Stack', 'NutAssembly', 'NutAssemblySingle', 'NutAssemblySquare', 'NutAssemblyRound', 
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
        print(colored(f"Initializing environment...", "yellow"))

        self._pred_name_to_pred = self.create_predicates()


        # Set the selected task - default to OpenSingleDoor
        self.task_selected = CFG.robo_kitchen_task
        if self.task_selected not in self.tasks:
            print(colored(f"Warning: Task {self.task_selected} not found in available tasks. Defaulting to OpenSingleDoor", "yellow"))
            self.task_selected = "OpenSingleDoor"
        print(colored(f"Selected task: {self.task_selected}", "green"))

        self.objects_of_interest = self.get_objects_of_interest()

        config = {
            "env_name": self.task_selected,
            "robots": robot_type,
            "controller_configs": controller_config,
            "layout_ids": 0, # this is the layout of the kitchen
            "style_ids": 0, # this is the style of the kitchen
            "translucent_robot": True,
        }

        self._env_raw = robosuite.make(
            **config,
            has_renderer=self._using_gui,
            has_offscreen_renderer=False,
            render_camera="robot0_frontview",
            ignore_done=True,
            use_camera_obs=False,
            control_freq=20,
            renderer="mjviewer" 
        )

        # Wrap this with visualization wrapper
        self._env = VisualizationWrapper(self._env_raw)
        self.ep_meta = self._env.get_ep_meta()
        # robosuite.robots.robot.print_action_info()
        for robot in self._env.robots:
            robot.print_action_info()
        
    def get_objects_of_interest(self) -> List[Object]:
        """Get the object of interest for the task."""
        if self.task_selected == "OpenSingleDoor":
            return [self.object_name_to_object("handle"), 
                    self.object_name_to_object("door")]
        # by default, there are robot and gripper objects
        else:
            raise ValueError(f"Task {self.task_selected} not supported")

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        """Create an ordered list of tasks for training from demos."""
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

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        # each task has a success condition, we can translate to predicate
        tasks = [] 
   
        for task_idx in range(CFG.num_test_tasks):
            seed = utils.get_task_seed("test", task_idx)
            init_obs = self._reset_initial_state_from_seed(seed, "test")
            # Simplified goal for initial implementation
            goal_description = self.task_selected
            task = EnvironmentTask(init_obs, goal_description)
            tasks.append(task)
        return tasks
    

    def _reset_initial_state_from_seed(self, seed: int,
                                       train_or_test: str) -> Observation:
        
        if CFG.robo_kitchen_randomize_init_state:
            pass # do randomization
        
        return {
            "state_info": self._env.reset(),
            "obs_images": []
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
            # Predicate("AtPreTurnOff", [cls.rich_object_type, cls.rich_object_type],
            #           cls._AtPreTurnOff_holds),
            # Predicate("AtPreTurnOn", [cls.rich_object_type, cls.rich_object_type],
            #           cls._AtPreTurnOn_holds),
            # Predicate("AtPrePushOnTop", [cls.rich_object_type, cls.rich_object_type],
            #           cls._AtPrePushOnTop_holds),
            # Predicate("AtPrePullKettle", [cls.rich_object_type, cls.rich_object_type],
            #           cls._AtPrePullKettle_holds),
            # Predicate("OnTop", [cls.rich_object_type, cls.rich_object_type],
            #           cls._OnTop_holds),
            # Predicate("NotOnTop", [cls.rich_object_type, cls.rich_object_type],
            #           cls._NotOnTop_holds),
            # Predicate("TurnedOn", [cls.rich_object_type], cls.On_holds),
            # Predicate("TurnedOff", [cls.rich_object_type], cls.Off_holds),
            Predicate("Open", [cls.hinge_door_type], cls.Open_holds),
            Predicate("Closed", [cls.hinge_door_type], cls.Closed_holds),
            Predicate("InContact", [cls.rich_object_type, cls.rich_object_type],
                     cls._InContact_holds),
            # Predicate("BurnerAhead", [cls.rich_object_type, cls.rich_object_type],
            #           cls._BurnerAhead_holds),
            # Predicate("BurnerBehind", [cls.rich_object_type, cls.rich_object_type],
            #           cls._BurnerBehind_holds),
            # Predicate("KettleBoiling",
            #           [cls.rich_object_type, cls.rich_object_type, cls.rich_object_type],
            #           cls._KettleBoiling_holds),
            # Predicate("KnobAndBurnerLinked", [cls.rich_object_type, cls.rich_object_type],
            #           cls._KnobAndBurnerLinkedHolds),
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
        obs = {
            "state_info": obs,
            "obs_images": []
        }
        
        self._current_observation = obs
        return self._copy_observation(self._current_observation)

    def reset(self, train_or_test: str, task_idx: int) -> Observation:
        """Reset environment to initial state."""
        #TODO: Need to get the task as predicates
        # self._current_task = self.get_task(train_or_test, task_idx)
        # seed = utils.get_task_seed(train_or_test, task_idx)
        # Add warning that task-specific reset not implemented
        warnings.warn("reset task not implemented for robo_kitchen")

        # Reset robosuite env
        obs = self._env.reset()
        obs = {
            "state_info": obs,
            "obs_images": []
        }
        
        self._current_observation = obs
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
        # Copying from kitchen.py but simplified for initial implementation
        # OnTop = self._pred_name_to_pred["OnTop"]
        # TurnedOn = self._pred_name_to_pred["TurnedOn"]
        # KettleBoiling = self._pred_name_to_pred["KettleBoiling"]
        # KnobAndBurnerLinked = self._pred_name_to_pred["KnobAndBurnerLinked"]
        goal_preds = {self._pred_name_to_pred["InContact"]}
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
            self.rich_object_type, self.hinge_door_type
        }

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
        if obj_name.endswith("_angle"):
            return Object(obj_name, cls.hinge_door_type)
        else:
            return Object(obj_name, cls.rich_object_type)
    
    @classmethod
    def state_info_to_state(cls, state_info: Dict[str, Any], contact_set: set[Tuple[Object, Object]] = None) -> State:
        state_dict = {}
        for key, val in state_info.items():
            if key.endswith("_angle"):
                obj_name = key
                obj = cls.object_name_to_object(obj_name)
                state_dict[obj] = {
                    "angle": val #currently only support 1 door, double door doesn't work
                }
            # elif key.endswith("_rot"): # This is deprecated since we want to use same representation as online
            #     obj_name = key[:-4]  # Remove _pos
            #     pos_val = state_info[key[:-4] + "_pos"]
            #     obj = cls.object_name_to_object(obj_name)
            #     state_dict[obj] = {
            #             "x": pos_val[0],
            #             "y": pos_val[1],
            #             "z": pos_val[2],
            #             "R00": val[0,0],
            #             "R01": val[0,1],
            #             "R02": val[0,2],
            #             "R10": val[1,0],
            #             "R11": val[1,1],
            #             "R12": val[1,2],
            #             "R20": val[2,0],
            #             "R21": val[2,1],
            #             "R22": val[2,2],
            #         }
            elif key.endswith("_quat"):
                obj_name = key[:-5]  # Remove _pos
                pos_val = state_info[key[:-5] + "_pos"]
                obj = cls.object_name_to_object(obj_name)
                state_dict[obj] = {
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
    def Open_holds(cls,
                   state: State,
                   objects: Sequence[Object],
                   thresh_pad: float = 0.0) -> bool:
        """Made public for use in ground-truth options."""
        obj = objects[0]
        if obj.is_instance(cls.hinge_door_type):
            return state.data[obj][0] > cls.hinge_open_thresh
        return False

    @classmethod
    def Closed_holds(cls,
                     state: State,
                     objects: Sequence[Object],
                     thresh_pad: float = 0.0) -> bool:
        """Made public for use in ground-truth options."""
        # Can't do not Open_holds() because of thresh_pad logic.
        obj = objects[0]
        if obj.is_instance(cls.hinge_door_type):
           return state.data[obj][0] > cls.hinge_open_thresh
            
        return False



    @classmethod
    def _InContact_holds(cls, 
                         state: State, 
                         objects: Sequence[Object]) -> bool:
        """Check if two objects are in contact using robosuite's contact checking."""
        obj1, obj2 = objects
        return (obj1, obj2) in state.items_in_contact or (obj2, obj1) in state.items_in_contact
    # @classmethod
    # def _AtPreTurn_holds(cls, state: State, objects: Sequence[Object],
    #                      on_or_off: str) -> bool:
    #     """Helper for _AtPreTurnOn_holds() and _AtPreTurnOff_holds()."""
    #     gripper, obj = objects
    #     obj_xyz = np.array(
    #         [state.get(obj, "x"),
    #          state.get(obj, "y"),
    #          state.get(obj, "z")])
    #     # On refers to Open and Off to Close
    #     dpos = cls.get_pre_push_delta_pos(obj, on_or_off)
    #     gripper_xyz = np.array([
    #         state.get(gripper, "x"),
    #         state.get(gripper, "y"),
    #         state.get(gripper, "z")
    #     ])
    #     return np.allclose(obj_xyz + dpos,
    #                        gripper_xyz,
    #                        atol=cls.at_pre_turn_atol)

    # @classmethod
    # def _AtPreTurnOn_holds(cls, state: State,
    #                        objects: Sequence[Object]) -> bool:
    #     return cls._AtPreTurn_holds(state, objects, "on")

    # @classmethod
    # def _AtPreTurnOff_holds(cls, state: State,
    #                         objects: Sequence[Object]) -> bool:
    #     return cls._AtPreTurn_holds(state, objects, "off")

    # @classmethod
    # def _AtPrePushOnTop_holds(cls, state: State,
    #                           objects: Sequence[Object]) -> bool:
    #     # The main thing that's different from _AtPreTurnOn_holds is that the
    #     # x position has a much higher range of allowed values, since it can
    #     # be anywhere behind the object.
    #     gripper, obj = objects
    #     obj_xyz = np.array(
    #         [state.get(obj, "x"),
    #          state.get(obj, "y"),
    #          state.get(obj, "z")])
    #     dpos = cls.get_pre_push_delta_pos(obj, "on")
    #     target_x, target_y, target_z = obj_xyz + dpos
    #     gripper_x, gripper_y, gripper_z = [
    #         state.get(gripper, "x"),
    #         state.get(gripper, "y"),
    #         state.get(gripper, "z")
    #     ]
    #     if not np.allclose([target_y, target_z], [gripper_y, gripper_z],
    #                        atol=cls.at_pre_pushontop_yz_atol):
    #         return False
    #     return np.isclose(target_x,
    #                       gripper_x,
    #                       atol=cls.at_pre_pushontop_x_atol)

    # @classmethod
    # def _AtPrePullKettle_holds(cls, state: State,
    #                            objects: Sequence[Object]) -> bool:
    #     gripper, obj = objects
    #     obj_xyz = np.array(
    #         [state.get(obj, "x"),
    #          state.get(obj, "y"),
    #          state.get(obj, "z")])
    #     dpos = cls.get_pre_push_delta_pos(obj, "off")
    #     target_x, target_y, target_z = obj_xyz + dpos
    #     gripper_x, gripper_y, gripper_z = [
    #         state.get(gripper, "x"),
    #         state.get(gripper, "y"),
    #         state.get(gripper, "z")
    #     ]
    #     if not np.allclose([target_y, target_z], [gripper_y, gripper_z],
    #                        atol=cls.at_pre_pullontop_yz_atol):
    #         return False
    #     return np.isclose(target_x,
    #                       gripper_x,
    #                       atol=cls.at_pre_pushontop_x_atol)

    # @classmethod
    # def _OnTop_holds(cls, state: State, objects: Sequence[Object]) -> bool:
    #     obj1, obj2 = objects
    #     obj1_xy = [state.get(obj1, "x"), state.get(obj1, "y")]
    #     obj2_xy = [
    #         state.get(obj2, "x"),
    #         state.get(obj2, "y"),
    #     ]
    #     return np.allclose(obj1_xy,
    #                        obj2_xy, atol=cls.ontop_atol) and state.get(
    #                            obj1, "z") > state.get(obj2, "z")

    # @classmethod
    # def _NotOnTop_holds(cls, state: State, objects: Sequence[Object]) -> bool:
    #     return not cls._OnTop_holds(state, objects)

    # @classmethod
    # def On_holds(cls,
    #              state: State,
    #              objects: Sequence[Object],
    #              thresh_pad: float = -0.06) -> bool:
    #     """Made public for use in ground-truth options."""
    #     obj = objects[0]
    #     if obj.is_instance(cls.knob_type):
    #         return state.get(obj, "angle") < cls.on_angle_thresh - thresh_pad
    #     if obj.is_instance(cls.switch_type):
    #         return state.get(obj, "x") < cls.light_on_thresh - thresh_pad
    #     return False

    # @classmethod
    # def Off_holds(cls,
    #               state: State,
    #               objects: Sequence[Object],
    #               thresh_pad: float = 0.0) -> bool:
    #     """Made public for use in ground-truth options."""
    #     # Can't do not On_holds() because of thresh_pad logic.
    #     obj = objects[0]
    #     if obj.is_instance(cls.knob_type):
    #         return state.get(obj, "angle") >= cls.on_angle_thresh + thresh_pad
    #     if obj.is_instance(cls.switch_type):
    #         return state.get(obj, "x") >= cls.light_on_thresh + thresh_pad
    #     return False

     # @classmethod
    # def _BurnerAhead_holds(cls, state: State,
    #                        objects: Sequence[Object]) -> bool:
    #     """Static predicate useful for deciding between pushing or pulling the
    #     kettle."""
    #     burner1, burner2 = objects
    #     if burner1 == burner2:
    #         return False
    #     return state.get(burner1, "y") > state.get(burner2, "y")

    # @classmethod
    # def _BurnerBehind_holds(cls, state: State,
    #                         objects: Sequence[Object]) -> bool:
    #     """Static predicate useful for deciding between pushing or pulling the
    #     kettle."""
    #     burner1, burner2 = objects
    #     if burner1 == burner2:
    #         return False
    #     return not cls._BurnerAhead_holds(state, objects)

    # @classmethod
    # def _KettleBoiling_holds(cls, state: State,
    #                          objects: Sequence[Object]) -> bool:
    #     """Predicate that's necessary for goal specification."""
    #     kettle, burner, knob = objects
    #     return cls.On_holds(state, [knob]) and cls._OnTop_holds(
    #         state, [kettle, burner]) and cls._KnobAndBurnerLinkedHolds(
    #             state, [knob, burner])

    # @classmethod
    # def _KnobAndBurnerLinkedHolds(cls, state: State,
    #                               objects: Sequence[Object]) -> bool:
    #     """Predicate that's necessary for goal specification."""
    #     del state  # unused
    #     knob, burner = objects
    #     # NOTE: we assume the knobs and burners are
    #     # all named "knob1", "burner1", .... And that "knob1" corresponds
    #     # to "burner1"
    #     return knob.name[-1] == burner.name[-1]

