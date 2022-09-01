"""Tests for IKFast loading and installation module."""
import os
from unittest.mock import call, patch

import pytest

import predicators.pybullet_helpers.ikfast.load
from predicators.pybullet_helpers.ikfast import IKFastInfo
from predicators.pybullet_helpers.ikfast.load import \
    install_ikfast_if_required, install_ikfast_module
from predicators.utils import get_third_party_path

_MODULE_PATH = predicators.pybullet_helpers.ikfast.load.__name__


@pytest.fixture(scope="module", name="ikfast_info")
def _ikfast_info_fixture() -> IKFastInfo:
    return IKFastInfo(
        module_dir="cool_robot",
        module_name="ikfast_cool_robot",
        base_link="cool_link0",
        ee_link="cool_link7",
        free_joints=["cool_link6"],
    )


def test_install_ikfast_module():
    """Test for install_ikfast_module."""
    with patch(f"{_MODULE_PATH}.os") as mock_os:
        mock_os.system.return_value = 0
        install_ikfast_module("/path/to/ikfast")
        mock_os.system.assert_called_once_with(
            "cd /path/to/ikfast; python setup.py")


@pytest.mark.parametrize("exit_code", list(range(1, 5)))
def test_install_ikfast_module_raises_error(exit_code):
    """Test install_ikfast_module raises error if os.system returns non-
    zero."""
    with patch(f"{_MODULE_PATH}.os") as mock_os:
        mock_os.system.return_value = exit_code

        with pytest.raises(RuntimeError):
            install_ikfast_module("/path/to/ikfast")


def test_install_ikfast_if_required_installs_ikfast_module(ikfast_info):
    """Test install_ikfast_if_required installs IKFast if there are no existing
    module files."""
    expected_ikfast_dir = os.path.join(get_third_party_path(), "ikfast",
                                       ikfast_info.module_dir)
    expected_module_path = os.path.join(
        expected_ikfast_dir, "ikfast_cool_arm.cpython-39-x86_64-linux-gnu.so")
    expected_glob_pattern = os.path.join(expected_ikfast_dir,
                                         f"{ikfast_info.module_name}*.so")

    with patch(f"{_MODULE_PATH}.install_ikfast_module"
               ) as mock_install_ikfast_module, patch(
                   f"{_MODULE_PATH}.glob") as mock_glob:
        mock_glob.glob.side_effect = [
            # First call returns no files, so install_ikfast_module is invoked
            [],
            # Now IKFast should have been installed
            [expected_module_path],
        ]

        module_path = install_ikfast_if_required(ikfast_info)
        assert module_path == expected_module_path

        # Check call made to install IKFast
        mock_install_ikfast_module.assert_called_once_with(expected_ikfast_dir)

        # Check glob called twice with expected pattern
        assert mock_glob.glob.call_count == 2
        mock_glob.glob.assert_has_calls(
            [call(expected_glob_pattern),
             call(expected_glob_pattern)])


def test_install_ikfast_if_required_returns_module_path(ikfast_info):
    """Test install_ikfast_if_required returns path to IKFast module if it
    already exists."""
    expected_ikfast_dir = os.path.join(get_third_party_path(), "ikfast",
                                       ikfast_info.module_dir)
    expected_module_path = os.path.join(
        expected_ikfast_dir, "ikfast_cool_arm.cpython-39-x86_64-linux-gnu.so")
    expected_glob_pattern = os.path.join(expected_ikfast_dir,
                                         f"{ikfast_info.module_name}*.so")

    with patch(f"{_MODULE_PATH}.install_ikfast_module"
               ) as mock_install_ikfast_module, patch(
                   f"{_MODULE_PATH}.glob") as mock_glob:
        mock_glob.glob.side_effect = [
            # IKFast already installed
            [expected_module_path],
        ]

        module_path = install_ikfast_if_required(ikfast_info)
        assert module_path == expected_module_path

        # Check no call made to install IKFast
        mock_install_ikfast_module.assert_not_called()

        # Check glob called once with expected pattern
        assert mock_glob.glob.call_count == 1
        mock_glob.glob.assert_called_once_with(expected_glob_pattern)


def test_install_ikfast_if_required_raises_error(ikfast_info):
    """Test install_ikfast_if_required raises error if more than 1 .so file
    found."""
    with patch(f"{_MODULE_PATH}.install_ikfast_module"
               ) as mock_install_ikfast_module, patch(
                   f"{_MODULE_PATH}.glob") as mock_glob:
        mock_glob.glob.side_effect = [
            ["1.so", "2.so", "3.so"],
        ]

        with pytest.raises(ValueError):
            install_ikfast_if_required(ikfast_info)

        # No calls to install IKFast, 1 call to glob
        mock_install_ikfast_module.assert_not_called()
        assert mock_glob.glob.call_count == 1
