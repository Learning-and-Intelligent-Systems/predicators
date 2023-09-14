"""Interface for spot placing skill."""

import logging
import time
from typing import Tuple

from bosdyn.api import geometry_pb2, manipulation_api_pb2
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.sdk import Robot

from predicators.spot_utils.perception.perception_structs import \
    RGBDImageWithContext
from predicators.spot_utils.utils import stow_arm
