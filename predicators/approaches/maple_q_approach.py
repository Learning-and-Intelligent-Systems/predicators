"""A parameterized action reinforcement learning approach inspired by MAPLE,
(https://ut-austin-rpl.github.io/maple/) but where only a Q-function is
learned.

Base samplers and applicable actions are used to perform the argmax.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Set

import dill as pkl
import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.approaches.online_nsrt_learning_approach import \
    OnlineNSRTLearningApproach
from predicators.explorers import BaseExplorer, create_explorer
from predicators.ml_models import MapleQFunction
from predicators.settings import CFG
from predicators.structs import Action, GroundAtom, InteractionRequest, \
    LowLevelTrajectory, ParameterizedOption, Predicate, State, Task, Type, \
    _GroundNSRT, _Option


class MapleQApproach(OnlineNSRTLearningApproach):
    """A parameterized action RL approach inspired by MAPLE."""

    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks)

        # The current implementation assumes that NSRTs are not changing.
        assert CFG.strips_learner == "oracle"
        # The base sampler should also be unchanging and from the oracle.
        assert CFG.sampler_learner == "oracle"

        # Log all transition data.
        self._interaction_goals: List[Set[GroundAtom]] = []
        self._last_seen_segment_traj_idx = -1

        # Store the Q function. Note that this implicitly
        # contains a replay buffer.
        self._q_function = MapleQFunction(
            seed=CFG.seed,
            hid_sizes=CFG.mlp_regressor_hid_sizes,
            max_train_iters=CFG.mlp_regressor_max_itr,
            clip_gradients=CFG.mlp_regressor_clip_gradients,
            clip_value=CFG.mlp_regressor_gradient_clip_value,
            learning_rate=CFG.learning_rate,
            weight_decay=CFG.weight_decay,
            use_torch_gpu=CFG.use_torch_gpu,
            train_print_every=CFG.pytorch_train_print_every,
            n_iter_no_change=CFG.active_sampler_learning_n_iter_no_change,
            num_lookahead_samples=CFG.
            active_sampler_learning_num_lookahead_samples)

    @classmethod
    def get_name(cls) -> str:
        return "maple_q"
    
    def print_light_q_values(self):
        state = [0.5      , 1.5      , 2.5      , 1.5      , 1.0       , 0.       ,
       0.75, 2.5      , 2.5      ]
        
        light_actions = [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        1.0, -0.25], [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        1.0, 0.], [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        1.0, 0.25], [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        1.0, 0.5], [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        1.0, 0.7], [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        1.0, 0.75], [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        1.0, 0.8], [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        1.0, 1.0]]

        print("Q VALUESSS")
        q_values=[]
        for option in light_actions:
            
            x = np.concatenate([
            state,
            [1],
            option
            ])
            y = self._q_function.predict(x)[0]
            print(y)
            q_values.append(y)

        return q_values
    
    def _solve(self, task: Task, timeout: int, train_or_test) -> Callable[[State], Action]:
        def _option_policy(state: State) -> _Option:
            return self._q_function.get_option(
                state,
                task.goal,
                num_samples_per_ground_nsrt=CFG.
                active_sampler_learning_num_samples, train_or_test=train_or_test)

        return utils.option_policy_to_policy(
            _option_policy, max_option_steps=CFG.max_num_steps_option_rollout)

    def _create_explorer(self) -> BaseExplorer:
        """Create a new explorer at the beginning of each interaction cycle."""
        # Geometrically increase the length of exploration.
        b = CFG.active_sampler_learning_explore_length_base
        max_steps = b**(1 + self._online_learning_cycle)
        preds = self._get_current_predicates()
        assert CFG.explorer == "maple_q"
        explorer = create_explorer(CFG.explorer,
                                   preds,
                                   self._initial_options,
                                   self._types,
                                   self._action_space,
                                   self._train_tasks,
                                   self._get_current_nsrts(),
                                   self._option_model,
                                   max_steps_before_termination=max_steps,
                                   maple_q_function=self._q_function)
        return explorer

    def load(self, online_learning_cycle: Optional[int]) -> None:
        super().load(online_learning_cycle)
        save_path = utils.get_approach_load_path_str()
        with open(f"{save_path}_{online_learning_cycle}.DATA", "rb") as f:
            save_dict = pkl.load(f)
        self._q_function = save_dict["q_function"]
        self._last_seen_segment_traj_idx = save_dict[
            "last_seen_segment_traj_idx"]
        self._interaction_goals = save_dict["interaction_goals"]
        self._online_learning_cycle = CFG.skip_until_cycle + 1

    #an nsrt is a high level action. 
    #ground nsrt is a specific high level action for specific location and stuff
    def _learn_nsrts(self, trajectories: List[LowLevelTrajectory],
                     online_learning_cycle: Optional[int],
                     annotations: Optional[List[Any]]) -> None:
        # Start by learning NSRTs in the usual way.
        super()._learn_nsrts(trajectories, online_learning_cycle, annotations)
        # Check the assumption that operators and options are 1:1.
        # This is just an implementation convenience.
        # assert len({nsrt.option for nsrt in self._nsrts}) == len(self._nsrts)
        # for nsrt in self._nsrts:
        #     assert nsrt.option_vars == nsrt.parameters
        # On the first cycle, we need to register the ground NSRTs, goals, and
        # objects in the Q function so that it can define its inputs.
        if not online_learning_cycle:
            all_ground_nsrts: Set[_GroundNSRT] = set()
            if CFG.sesame_grounder == "naive":
                for nsrt in self._nsrts:
                    all_objects = {
                        o
                        for t in self._train_tasks for o in t.init
                    }
                    all_ground_nsrts.update(
                        utils.all_ground_nsrts(nsrt, all_objects))
            elif CFG.sesame_grounder == "fd_translator":  # pragma: no cover
                all_objects = set()
                for t in self._train_tasks:
                    curr_task_objects = set(t.init)
                    curr_task_types = {o.type for o in t.init}
                    curr_init_atoms = utils.abstract(
                        t.init, self._get_current_predicates())
                    all_ground_nsrts.update(
                        utils.all_ground_nsrts_fd_translator(
                            self._nsrts, curr_task_objects,
                            self._get_current_predicates(), curr_task_types,
                            curr_init_atoms, t.goal))
                    all_objects.update(curr_task_objects)
            else:  # pragma: no cover
                raise ValueError(
                    f"Unrecognized sesame_grounder: {CFG.sesame_grounder}")
            #eventually change the goal to good state
            goals = [t.goal for t in self._train_tasks]
            #initing the input vector
            self._q_function.set_grounding(all_objects, goals,
                                           all_ground_nsrts)
        # Update the data using the updated self._segmented_trajs.
        self._update_maple_data()
        # Re-learn Q function.
        self._q_function.train_q_function()
        # Save the things we need other than the NSRTs, which were already
        # saved in the above call to self._learn_nsrts()
        save_path = utils.get_approach_save_path_str()
        with open(f"{save_path}_{online_learning_cycle}.DATA", "wb") as f:
            pkl.dump(
                {
                    "q_function": self._q_function,
                    "last_seen_segment_traj_idx":
                    self._last_seen_segment_traj_idx,
                    "interaction_goals": self._interaction_goals,
                }, f)

    def _update_maple_data(self) -> None:
        start_idx = self._last_seen_segment_traj_idx + 1
        new_trajs = self._segmented_trajs[start_idx:]

        goal_offset = CFG.max_initial_demos
        # import ipdb; ipdb.set_trace()
        # assert len(self._segmented_trajs) == goal_offset + len(self._interaction_goals)
                
        new_traj_goals = self._interaction_goals[goal_offset + start_idx:]

        for traj_i, segmented_traj in enumerate(new_trajs):
            self._last_seen_segment_traj_idx += 1
            for seg_i, segment in enumerate(segmented_traj):
                s = segment.states[0]
                goal = new_traj_goals[traj_i]
                o = segment.get_option()
                ns = segment.states[-1]
                #eventually improve this reward
                reward = 1.0 if goal.issubset(segment.final_atoms) else 0.0
                terminal = reward > 0 or seg_i == len(segmented_traj) - 1
                vectorized_action=self._q_function._vectorize_option(o)
                vectorized_state=self._q_function._vectorize_state(s)

                # if vectorized_action[-1]<0.85 and vectorized_action[-1]>0.65:
                #     print("reward of turning the light and terminal?", reward, terminal)
                #     if not terminal:
                #         print(vectorized_action, vectorized_state, o, s, segment.final_atoms, segment)
                # if reward == 1.0:
                #     print(s,o)
                # if reward !=0 and reward!=1:
                #     print(reward)
                self._q_function.add_datum_to_replay_buffer(
                    (s, goal, o, ns, reward, terminal))

    def get_interaction_requests(self) -> List[InteractionRequest]:
        # Save the goals for each interaction request so we can later associate
        # states, actions, and goals.
        requests = super().get_interaction_requests()
        for request in requests:
            goal = self._train_tasks[request.train_task_idx].goal
            self._interaction_goals.append(goal)
        return requests
