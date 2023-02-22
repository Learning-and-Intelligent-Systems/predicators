"""Wrapper around pyrealsense2.

Based on: https://github.com/IntelRealSense/librealsense/issues/8388
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import imageio.v2 as iio
import numpy as np
import pyrealsense2 as rs  # pylint: disable=import-error

# Note: change these to fit your use case. Assuming USB 3.2 connection.
_NAME_TO_STREAM_CONFIGURATIONS: Dict[str, List[Tuple]] = {
    # Mapping of camera name to a list of streams to enable
    # in the cfg.enable_stream format
    "Intel RealSense L515": [
        (rs.stream.depth, 1024, 768, rs.format.z16, 30),
        (rs.stream.color, 1920, 1080, rs.format.rgb8, 30),
    ],
    "Intel RealSense D415": [
        (rs.stream.depth, 1280, 720, rs.format.z16, 30),
        (rs.stream.color, 1280, 720, rs.format.rgb8, 30),
    ],
    "Intel RealSense D435": [
        (rs.stream.depth, 1280, 720, rs.format.z16, 30),
        (rs.stream.color, 1280, 720, rs.format.rgb8, 30),
    ],
}


@dataclass
class _Device:
    """A single realsense camera."""
    name: str
    serial_number: str
    # RealSense pipeline for this device
    pipeline: Optional[rs.pipeline] = None
    counter = 0

    @classmethod
    def from_rs_device(cls, dev: rs.device) -> _Device:
        """Initialize a _Device from a pyrealsense2 device."""
        name = dev.get_info(rs.camera_info.name)
        serial_number = dev.get_info(rs.camera_info.serial_number)
        return cls(name, serial_number)

    @property
    def stream_configurations(self) -> List[Tuple]:
        """Get the stream configurations for this device."""
        if self.name not in _NAME_TO_STREAM_CONFIGURATIONS:
            raise RuntimeError(f"Configuration not specified for {self.name}")

        return _NAME_TO_STREAM_CONFIGURATIONS[self.name]

    def start_pipeline(self,
                       use_auto_exposure: bool = True,
                       custom_exposure: int = 2000,
                       custom_gain: int = 50,
                       use_auto_white_balance: bool = True,
                       use_backlight_compensation: bool = False,
                       custom_white_balance: int = 3700) -> None:
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

        if not use_auto_exposure:
            color_sensor.set_option(rs.option.enable_auto_exposure, False)
            color_sensor.set_option(rs.option.exposure, custom_exposure)
            color_sensor.set_option(rs.option.gain, custom_gain)

        if not use_backlight_compensation:
            color_sensor.set_option(rs.option.backlight_compensation, 0)

        if not use_auto_white_balance:
            color_sensor.set_option(rs.option.enable_auto_white_balance, False)
            color_sensor.set_option(rs.option.white_balance,
                                    custom_white_balance)

        # Set the pipeline on the device
        self.pipeline = pipeline
        print(f"Started pipeline for {self}")

    def stop_pipeline(self) -> None:
        """Stop RealSense pipeline."""
        if not self.pipeline:
            print(f"Warning! {self} does not have a pipeline initialized")
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


def _find_devices(device_filter: str = "") -> List[_Device]:
    """Get devices as detected by RealSense and filter devices that only
    contain the provided device_filter string in their name.

    e.g. to filter for D435 only you can call `_find_devices("D435")`
    """
    ctx = rs.context()
    devices = [_Device.from_rs_device(dev) for dev in ctx.devices]  # pylint: disable=not-an-iterable

    # Filter devices
    if device_filter:
        devices = [
            d for d in devices if device_filter.lower() in d.name.lower()
        ]
        print(f"Found devices (filter={device_filter}): {devices}")
    else:
        print(f"Found devices: {devices}")

    if not devices:
        raise RuntimeError("No devices connected!")
    return devices


def _hardware_reset_connected_devices() -> None:
    """Reset all connected devices.

    Useful if the USB connection is unreliable, or we get bad stream
    from the camera.
    """
    ctx = rs.context()
    devices = ctx.query_devices()
    for dev in devices:
        dev.hardware_reset()


def _main(
    color_path: Path,
    depth_path: Path,
    device_name: str,
    reset_hardware: bool = False,
) -> None:
    """Capture and save a color and depth image from the device."""

    if reset_hardware:
        _hardware_reset_connected_devices()

    devices = _find_devices(device_name)
    assert len(devices) == 1
    device = devices[0]
    device.start_pipeline()

    color, depth = device.capture_images()
    iio.imwrite(color_path, color)
    iio.imwrite(depth_path, depth)

    device.stop_pipeline()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb", type=str, default="color.png")
    parser.add_argument("--depth", type=str, default="depth.png")
    parser.add_argument("--device_name", type=str, default="D415")
    parser.add_argument("--reset_hardware", action="store_true")
    args = parser.parse_args()
    _main(args.rgb, args.depth, args.device_name, args.reset_hardware)
