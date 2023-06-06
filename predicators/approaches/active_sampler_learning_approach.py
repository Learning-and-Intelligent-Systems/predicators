"""An approach that performs active sampler learning.

Example commands
----------------

Bumpy cover easy:
    python predicators/main.py --approach active_sampler_learning --env bumpy_cover \
        --seed 0 \
        --strips_learner oracle \
        --bilevel_plan_without_sim True \
        --offline_data_bilevel_plan_without_sim False \
        --explorer random_nsrts \
        --max_initial_demos 1 \
        --num_train_tasks 1000 \
        --num_test_tasks 10 \
        --max_num_steps_interaction_request 100 \
        --bumpy_cover_num_bumps 2 \
        --bumpy_cover_spaces_per_bump 1 \
        --bumpy_cover_thr_percent 1.0

TODO: do we actually need this new class? Probably yes because we need to try to solve the
training tasks first and then explore. But maybe that's just an explorer?
"""

import logging
from typing import Callable, Dict, List, Optional, Sequence, Set

import dill as pkl
import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.approaches.online_nsrt_learning_approach import OnlineNSRTLearningApproach
from predicators.explorers import create_explorer
from predicators.ml_models import BinaryClassifierEnsemble, \
    KNeighborsClassifier, LearnedPredicateClassifier, MLPBinaryClassifier
from predicators.settings import CFG
from predicators.structs import Dataset, GroundAtom, GroundAtomsHoldQuery, \
    GroundAtomsHoldResponse, InteractionRequest, InteractionResult, \
    LowLevelTrajectory, ParameterizedOption, Predicate, Query, State, Task, \
    Type


class ActiveSamplerLearningApproach(OnlineNSRTLearningApproach):
    """Performs active sampler learning.
    
    Run with --strips_learner oracle so that only samplers are learned.
    """

    @classmethod
    def get_name(cls) -> str:
        return "active_sampler_learning"
