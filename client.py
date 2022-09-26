""" Client to call the Panda server """
import json
import os
from dataclasses import dataclass
from typing import ClassVar, Dict, Optional, List

import requests
from lisdf.planner_output.command import JointSpacePath
from loguru import logger

from lisdf.planner_output.plan import LISDFPlan


USE_LOCALHOST = False

_MODULE_PATH = os.path.dirname(os.path.abspath(__file__))
_TOKEN_JSON = os.path.join(_MODULE_PATH, "token.json")


def get_jwt_token(force_refresh: bool = True) -> str:
    """
    Calls endpoint to get JWT if it doesn't exist.
    Otherwise, load from disk.
    """
    auth_endpoint = f"{PandaClient.endpoint()}/auth"
    if force_refresh or not os.path.exists(_TOKEN_JSON):
        response = requests.get(auth_endpoint)
        token = response.json()
        with open(_TOKEN_JSON, "w") as f:
            json.dump(response.json(), f)
        print(f"Wrote new token to {_TOKEN_JSON}")
    else:
        with open(_TOKEN_JSON, "r") as f:
            token = json.load(f)
        print(f"Loaded token from {_TOKEN_JSON}")

    token = token["access_token"]
    return token


@dataclass
class PandaClient:
    """Panda Client for communicating with our custom Panda Server"""

    token: Optional[str]

    # ted is on the Stata Center Wi-Fi network, so IP may change occasionally
    ted_ip: ClassVar[str] = "http://128.31.36.7"
    port: ClassVar[int] = 1234
    use_localhost: ClassVar[bool] = False

    @classmethod
    def endpoint(cls) -> str:
        if cls.use_localhost:
            return f"http://localhost:{cls.port}"
        else:
            return f"{cls.ted_ip}:{cls.port}"

    @property
    def headers(self) -> Dict:
        """HTTP headers for requests"""
        return {"Authorization": f"Bearer {self.token}"}

    def get_joint_positions(self) -> Dict[str, float]:
        """
        Get joint positions of the Panda arm.
        Returns
        -------
        {
            "panda_joint1": float,
            ...
            "panda_joint7": float
        }
        """
        response = requests.get(
            f"{self.endpoint()}/get_joint_positions", headers=self.headers
        )
        return response.json()

    def get_ee_pose(self) -> Dict[str, List[float]]:
        """
        Get end effector pose of the Panda arm.
        Position is in (x, y, z) format.
        Orientation is in quaternion (x, y, z, w) format.
        Returns
        -------
        {
            "position": [x, y, z],
            "orientation": [x, y, z, w]
        }
        """
        response = requests.get(f"{self.endpoint()}/get_ee_pose", headers=self.headers)
        json_dict = response.json()

        p = json_dict["position"]
        p_list = [p["x"], p["y"], p["z"]]

        o = json_dict["orientation"]
        o_list = [o["x"], o["y"], o["z"], o["w"]]

        return {
            "position": p_list,
            "orientation": o_list
        }

    def go_home(self) -> str:
        """
        Tell Panda to go back to its neutral positions.
        Note, that this API motion plans to the home position with
        respective to the workspace (i.e., table and frame).
        """
        response = requests.post(f"{self.endpoint()}/go_home", headers=self.headers)
        return response.text

    def enable_gravity_compensation(self) -> str:
        """
        Enable gravity compensation mode on the Panda.
        Similar to pressing the button when the white LEDs are on.
        """
        response = requests.post(f"{self.endpoint()}/gravity_compensation", headers=self.headers)
        return response.text

    def gripper_close(self) -> str:
        response = requests.post(f"{endpoint}/gripper_close", headers=self.headers)
        return response.text

    def gripper_open(self) -> str:
        response = requests.post(f"{endpoint}/gripper_open", headers=self.headers)
        return response.text

    def gripper_grasp(self, width: float = 0.04, force: float = 40.0) -> str:
        response = requests.post(f"{endpoint}/gripper_grasp", headers=self.headers, json={"width": width, "force": force})
        return response.text


endpoint = PandaClient.endpoint()


def get_headers(token: str):
    return {"Authorization": f"Bearer {token}"}


def get_joint_positions(token: str) -> Dict[str, float]:
    logger.warning("Deprecated: use PandaClient.get_joint_positions instead")
    response = requests.get(
        f"{endpoint}/get_joint_positions", headers=get_headers(token)
    )
    return response.json()


def get_gripper_positions(token: str):
    response = requests.get(
        f"{endpoint}/get_gripper_positions", headers=get_headers(token)
    )
    return response.json()


def gripper_open(token: str):
    logger.warning("Deprecated: use PandaClient.gripper_open instead")
    response = requests.post(f"{endpoint}/gripper_open", headers=get_headers(token))
    return response.text


def gripper_close(token: str):
    logger.warning("Deprecated: use PandaClient.gripper_close instead")
    response = requests.post(f"{endpoint}/gripper_close", headers=get_headers(token))
    return response.text


def go_home(token: str):
    logger.warning("Deprecated: use PandaClient.go_home instead")
    response = requests.post(f"{endpoint}/go_home", headers=get_headers(token))
    return response.text


def move_to_joint_positions(token: str, joint_positions: dict):
    headers = get_headers(token)
    response = requests.post(
        f"{endpoint}/move_to_joint_positions",
        headers=headers,
        json={"target_conf": joint_positions},
    )
    return response.text


def execute_position_path(token: str, position_path: list):
    headers = get_headers(token)
    response = requests.post(
        f"{endpoint}/execute_position_path",
        headers=headers,
        json={"position_path": position_path},
    )
    return response.text


def execute_lisdf_plan(lisdf_plan: LISDFPlan, token: str):
    headers = get_headers(token)
    headers["Content-Type"] = "application/json"
    lisdf_plan_json_dict = json.loads(lisdf_plan.to_json())
    response = requests.post(
        f"{endpoint}/execute_lisdf_plan", headers=headers, json=lisdf_plan_json_dict
    )
    return response.text


def demo():
    token = get_jwt_token()
    client = PandaClient(token)

    print("Joint Positions:", client.get_joint_positions())
    print("Gripper Positions:", get_gripper_positions(token))

    # print("Close Gripper:", gripper_close(token))
    print("Open Gripper:", gripper_open(token))
    # print("Go Home:", client.go_home())

    with open("scripts/pybullet_blocks__oracle__3________task1.json") as f:
        lisdf_plan = LISDFPlan.from_json(f.read())

        first_conf = lisdf_plan.commands[0]
        assert isinstance(first_conf, JointSpacePath)
        first_conf = first_conf.waypoint(0)
        print(first_conf)
        move_to_joint_positions(token, first_conf)
    #
    #
    print("Execute LISDF Plan:", execute_lisdf_plan(lisdf_plan, token))
    print('meme')


if __name__ == "__main__":
    demo()
