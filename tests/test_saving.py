import argparse
import os

from reax import saving


def test_loading_yaml(default_logger):
    hparams = {
        "batch_size": 32,
        "learning_rate": 0.001 * 8,
        "optimizer_name": "adam",
    }

    # save tags
    logger = default_logger
    logger.log_hyperparams(argparse.Namespace(some_str="a_str", an_int=1, a_float=2.0))
    logger.log_hyperparams(hparams)
    logger.save()

    # load hparams
    path_expt_dir = logger.experiment.log_dir
    hparams_path = os.path.join(path_expt_dir, "hparams.yaml")
    tags = saving.load_hparams_from_yaml(hparams_path)

    assert tags["batch_size"] == 32
    assert tags["optimizer_name"] == "adam"
