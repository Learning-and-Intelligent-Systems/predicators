"""Tests for settings.py.
"""

import os
from predicators.src import utils
from predicators.src.settings import CFG, get_save_path


def test_settings():
    """Tests for settings.py.
    """
    utils.update_config({"env": "not a real environment"})
    assert CFG.num_train_tasks == CFG.num_test_tasks == 0
    utils.update_config({"env": "cover"})
    assert CFG.num_train_tasks > 0
    assert CFG.num_test_tasks > 0
    dirname = "_fake_tmp_save_dir"
    old_save_dir = CFG.save_dir
    utils.update_config({"env": "test_env", "approach": "test_approach",
                         "seed": 123, "save_dir": dirname,
                         "excluded_predicates": "test_pred1,test_pred2"})
    save_path = get_save_path()
    assert save_path == dirname + ("/test_env___test_approach___123___"
                                   "test_pred1,test_pred2.saved")
    utils.update_config({"env": "test_env", "approach": "test_approach",
                         "seed": 123, "save_dir": dirname,
                         "excluded_predicates": ""})
    save_path = get_save_path()
    assert save_path == dirname + "/test_env___test_approach___123___.saved"
    os.rmdir(dirname)
    utils.update_config({"save_dir": old_save_dir})
