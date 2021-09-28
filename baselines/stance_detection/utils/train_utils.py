# Copyright 2021 ETH Zurich, Media Technology Center
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import gc
import sys
import json
import logging

from argparse import Namespace
from typing import Any, List, Dict, Optional, Union, Tuple

import pandas as pd
from datasets import DatasetBuilder, Dataset

import transformers
from transformers import TrainingArguments
from transformers.trainer_utils import is_main_process
from transformers.trainer_utils import get_last_checkpoint as \
    get_last_checkpoint_hf

from utils.arguments import CHeeSETrainingArguments, CHeeSEArguments

logger = logging.getLogger(__name__)
LOG_FILE_NAME = "run.log"
CONFIG_FILE_NAME = "run_config.json"
LOG_FORMAT = "[%(levelname)s|%(filename)s:%(lineno)d] %(asctime)s,%(msecs)d >> %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

def get_num_classes(dataset: DatasetBuilder, keys: List[str]) -> Dict[str, int]:
    """ Returns the number of classes for each field in `keys` in dataset.

    Parameters
    ----------
    dataset:
        The dataset for which to retreive the num_classes
    keys:
        The dataset fields for which to extract the num_classes

    Returns
    -------
    A mapping from key to the number of classes
    """

    # Make keys a list if it is a string
    if type(keys) == str:
        keys = [keys]

    # Get the num_labels for each key
    attrib = "num_classes"
    num_labels = {}
    for key in keys:
        num_labels[key] = None
        if hasattr(dataset.features[key], attrib):
            num_labels[key] = dataset.features[key].num_classes
        elif hasattr(dataset.features[key], "feature"):
            if hasattr(dataset.features[key].feature, attrib):
                num_labels[key] = dataset.features[key].feature.num_classes
            elif hasattr(dataset.features[key].feature, "feature"):
                if hasattr(dataset.features[key].feature.feature, attrib):
                    num_labels[key] = \
                        dataset.features[key].feature.feature.num_classes
        if num_labels[key] is None:
            num_labels[key] = 2

    return num_labels

def store_mertics(metrics: Dict[str, Any], logger: logging.RootLogger = None,
        info: str = "Validation", file_path: str = "validation_results.txt"):
    """ Stores metrics to file and writes them to logger.
    
    Parameters
    ----------
    metrics:
        Mapping (name to value) for the metrics to store.
    logger:
        A logger. If provided the metrics are in addition of being written to a
        file also printed in the log.
    info:
        Information used in the logging.
    file_name:
        Name of the file to dump the metrics to.
    """
    with open(file_path, "w") as writer:
        if logger:
            logger.info(f"***** {info}  metrics *****")
        for key, value in sorted(metrics.items()):
            writer.write(f"{key} = {value}\n")
            if logger:
                logger.info(f"  {key} = {value}")


def store_predictions(predictions: dict, dataset: Dataset,
        labels_to_predict: List[str], prediction_columns_to_include:
        Optional[Union[List[str], str, Tuple[str], List[Tuple[str]]]],
        output_dir: str, prediction_csv_kwargs: dict = {}):
    """ Stores predictions to a file together with specified columns of the
    initial dataset.
    
    Parameters
    ----------
    predictions:
        Mapping (name to predictions) from the predictied labels to predictions.
    dataset:
        Dataset for which the predictions were made.
    labels_to_predict:
        List of names of the predicted targets.
    prediction_columns_to_include:
        Columns from dataset to additionally include into the predictions csv.
    output_dir:
        Where to write the predictions to.
    prediction_csv_kwargs:
        Arguments to write the file if output_dir is set.
    """
    column_order = []
    if prediction_columns_to_include is not None:
        for col in prediction_columns_to_include:
            if type(col) == str:
                predictions[col] = dataset[col]
                column_order += [col]
            else:
                predictions[col[1]] = dataset[col[0]]
                column_order += [col[1]]
    column_order += labels_to_predict
    df_pred = pd.DataFrame.from_dict(predictions)
    df_pred = df_pred[column_order]
    if output_dir:
        df_pred.to_csv(os.path.join(output_dir,
            "predictions.csv"), **prediction_csv_kwargs)

def get_last_checkpoint(training_args: TrainingArguments) -> str:
    """ Checks whether there is a previous checkpoint.
    
    Parameters
    ----------
    training_args:
        Training arguments.

    Returns
    -------
    The path to the previous checkpoint if there exists one - else None.
    """
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train \
            and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint_hf(training_args.output_dir)
        files = os.listdir(training_args.output_dir)
        if LOG_FILE_NAME in files:
            files.remove(LOG_FILE_NAME)
        if last_checkpoint is None and len(files) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists "
                "and is not empty. Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:

            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. "
                "To avoid this behavior, change the `--output_dir` or add "
                "`--overwrite_output_dir` to train from scratch."
            )
    return last_checkpoint

def get_run_config_from_checkpoint(checkpoint: str) -> str:
    """ Returns the config used for the training of a checkpoint.

    Parameters
    ----------
    checkpoint:
        Path to a checkpoint.

    Returns
    -------
    Path to the config used for the training of the checkpoint if there is one
    - else None.
    """
    path = os.path.abspath(os.path.join(checkpoint, os.pardir, 
        CONFIG_FILE_NAME))
    if os.path.isfile(path):
        return path
    else:
        logger.warning(f"Could not find the run_config file of the \
            checkpoint {checkpoint}.")
        return None

def setup_logging(args: CHeeSEArguments,
        training_args: CHeeSETrainingArguments) -> logging.Logger:
    """ Initiates a logger.

    Parameters
    ----------
    training_args:
        Training arguments that provide certain arguments to the logger setup.

    Returns
    -------
    A logger instance.
    """
    """ Setup logging. """
    if args.log_to_file:
        if not os.path.isdir(training_args.output_dir):
            os.makedirs(training_args.output_dir)

    logging.basicConfig(
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
        handlers=[logging.StreamHandler(sys.stdout)])

    logger.setLevel(logging.getLevelName(args.logging_level))

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, \
          device: {training_args.device}, n_gpu: {training_args.n_gpu}, \
          distributed training: {bool(training_args.local_rank != -1)}, \
          16-bits training: {training_args.fp16}"
    )
    # Set the verbosity of the logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity(
            logging.getLevelName(args.logging_level))
        transformers.utils.logging.disable_default_handler()
        transformers.utils.logging.enable_propagation()
        transformers.utils.logging.enable_explicit_format()

    logger.info("\n\n*** TRAINING / EVALUATION PARAMETERS ***\n%s\n",
        training_args)
    return logger

def save_config(output_dir: str, config_paths: Optional[List[str]] = None,
        cli_args: Optional[Namespace] = None):
    """ Saves the configs to a file.

    Parameters
    ----------
    output_dir:
        Directory in which to save the config file.
    config_paths:
        List of json config files (the later configs override the arguments
        of earlier ones).
    cli_args:
        A Namespace from argparse with cli arguments. Overrides arguments
        from `config_paths`.
    """
    if config_paths is not None:
        json_objects = []
        for config_path in config_paths:
            json_file = open(config_path, "r")
            json_object = json.load(json_file)
            json_file.close()
            json_objects.append(json_object)
        json_object = json_objects[0]
        for i in json_objects[1:]:
            json_object.update(i)
    else:
        json_object = {}

    if cli_args:
        # CLI args overwrite config args
        args_dict = vars(cli_args)
        for key, obj in args_dict.items():
            json_object[key] = obj

    save_file = open(os.path.join(output_dir, CONFIG_FILE_NAME), "w")
    json.dump(json_object, save_file, indent=4)
    save_file.close()