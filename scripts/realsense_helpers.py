"""Wrapper around pyrealsense2.

Based on:
https://github.com/IntelRealSense/librealsense/issues/8388
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyrealsense2 as rs  # pylint: disable=import-error

__all__ = ["Device", "find_devices", "start_pipelines", "stop_pipelines"]

# Note: change these to fit your use case. Assuming USB 3.2 connection.
_NAME_TO_STREAM_CONFIGURATIONS: Dict[str, List[Tuple]] = {
    # Mapping of camera name to a list of streams to enable
    # in the cfg.enable_stream format
    "Intel RealSense L515": [
        (rs.stream.depth, 1024, 768, rs.format.z16, 30),
        (rs.stream.color, 1920, 1080, rs.format.bgr8, 30),
    ],
    "Intel RealSense D415": [
        (rs.stream.depth, 1280, 720, rs.format.z16, 30),
        (rs.stream.color, 1280, 720, rs.format.bgr8, 30),
    ],
    "Intel RealSense D435": [
        (rs.stream.depth, 1280, 720, rs.format.z16, 30),
        (rs.stream.color, 1280, 720, rs.format.bgr8, 30),
    ],
}


@dataclass
class Device:
    """A single realsense camera."""
    name: str
    serial_number: str
    # RealSense pipeline for this device
    pipeline: Optional[rs.pipeline] = None
    counter = 0

    @classmethod
    def from_rs_device(cls, dev: rs.device) -> Device:
        """Initialize a Device from a pyrealsense2 device."""
        name = dev.get_info(rs.camera_info.name)
        serial_number = dev.get_info(rs.camera_info.serial_number)
        return cls(name, serial_number)

    @property
    def stream_configurations(self) -> List[Tuple]:
        """Get the stream configurations for this device."""
        if self.name not in _NAME_TO_STREAM_CONFIGURATIONS:
            raise RuntimeError(f"Configuration not specified for {self.name}")

        return _NAME_TO_STREAM_CONFIGURATIONS[self.name]

    def start_pipeline(self, custom_exposure_and_wb: bool = True) -> None:
        """Start RealSense pipeline."""
        if self.pipeline is not None:
            print(f"Pipeline already started for {self}")
            return

        # Setup pipeline and configuration
        ctx = rs.context()
        pipeline = rs.pipeline(ctx)
        cfg = rs.config()
        cfg.enable_device(self.serial_number)

        for stream_configuration in self.stream_configurations:
            cfg.enable_stream(*stream_configuration)

        try:
            profile = pipeline.start(cfg)
        except RuntimeError as e:
            message = str(e)
            if message == "Couldn't resolve requests":
                # Something wrong with stream configurations probably
                raise RuntimeError(
                    f"{message} for {self}. Check stream configuration and "
                    "USB connection.")
            raise e

        # Warmup camera
        for _ in range(30):
            pipeline.wait_for_frames()
        print("Camera warmed up!")

        # Set profile
        _, color_sensor, *_ = profile.get_device().query_sensors()
        print("Retrieved sensors")

        if custom_exposure_and_wb:
            color_sensor.set_option(rs.option.enable_auto_exposure, False)
            # Use 438 with Yang's ring light
            color_sensor.set_option(rs.option.exposure, 2000)
            color_sensor.set_option(rs.option.gain, 50)

            # Disable backlight compensation
            # color_sensor.set_option(rs.option.backlight_compensation, 0)

            color_sensor.set_option(rs.option.enable_auto_white_balance, False)
            color_sensor.set_option(rs.option.white_balance, 3700)
            print("Disabled auto exposure and white balance for color sensor")

        # Set the pipeline on the device
        self.pipeline = pipeline
        print(f"Started pipeline for {self}")

    def stop_pipeline(self) -> None:
        """Stop RealSense pipeline."""
        if not self.pipeline:
            print(
                f"Warning! Device {self} does not have a pipeline initialized")
        else:
            self.pipeline.stop()
            print(f"Stopped pipeline for {self}")

    def capture_images(self) -> Tuple[np.ndarray, np.ndarray]:
        """Capture color and depth images."""
        if self.pipeline is None:
            raise RuntimeError(f"Pipeline for {self} not started!")

        # Get frames and align
        frames = self.pipeline.wait_for_frames()
        align = rs.align(rs.stream.color)
        aligned_frames = align.process(frames)

        # Get color and depth frame
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        # Convert to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        return color_image, depth_image

    def __str__(self) -> str:
        return f"{self.name} ({self.serial_number})"

    def __repr__(self) -> str:
        return str(self)


def find_devices(device_filter: str = "") -> List[Device]:
    """Get devices as detected by RealSense and filter devices that only
    contain the provided device_filter string in their name.

    e.g. to filter for D435 only you can call `find_devices("D435")`
    """
    ctx = rs.context()
    devices = [Device.from_rs_device(dev) for dev in ctx.devices]

    # Filter devices
    if device_filter:
        devices = [d for d in devices if device_filter in d.name.lower()]
        print(f"Found devices (filter={device_filter}): {devices}")
    else:
        print(f"Found devices: {devices}")

    if not devices:
        raise RuntimeError("No devices connected!")
    return devices


def start_pipelines(devices: List[Device], **kwargs) -> None:
    """Enable each device by starting a stream."""
    for device in devices:
        device.start_pipeline(**kwargs)


def stop_pipelines(devices: List[Device]) -> None:
    """Stop all the pipelines."""
    for device in devices:
        device.stop_pipeline()


def hardware_reset_connected_devices() -> None:
    """Reset all connected devices."""
    ctx = rs.context()
    devices = ctx.query_devices()
    for dev in devices:
        dev.hardware_reset()


# if __name__ == "__main__":
#     os.makedirs("blocks_vision_data", exist_ok=True)
#     # hardware_reset_connected_devices()

#     # Detect devices, start pipelines, visualize, and stop pipelines
#     devices = find_devices("d415")
#     assert len(devices) == 1
#     start_pipelines(devices, custom_exposure_and_wb=True)

#     device = devices[0]
#     color, depth = device.capture_images()
#     color_path = os.path.join("blocks_vision_data",
#                               f"color-0-{device.serial_number}.png")
#     depth_path = os.path.join("blocks_vision_data",
#                               f"depth-0-{device.serial_number}.png")
#     cv2.imwrite(color_path, color)
#     cv2.imwrite(depth_path, depth)

#     # try:
#     #     visualize_devices(devices, save_dir="blocks_vision_data")
#     # finally:
#     stop_pipelines(devices)
