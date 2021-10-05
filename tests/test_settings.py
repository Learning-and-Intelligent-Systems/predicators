"""Tests for settings.py.
"""

from predicators.src import utils
from predicators.src.settings import CFG


def test_settings():
    """Tests for settings.py.
    """
    utils.update_config({"env": "not a real environment", "approach": ""})
    assert CFG.num_train_tasks == CFG.num_test_tasks == 0
    utils.update_config({"env": "cover", "approach": ""})
    assert CFG.num_train_tasks > 0
    assert CFG.num_test_tasks > 0
