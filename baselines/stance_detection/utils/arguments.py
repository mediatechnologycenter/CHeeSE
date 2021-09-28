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

from dataclasses import dataclass, field
from typing import Optional, List, Union, Tuple, Dict

from transformers import TrainingArguments

@dataclass
class CHeeSEArguments:
    """ General arguments to run the **run.py** python script.

    Parameters
    ----------
    model_name_or_path: str
        The model checkpoint for weights initialization.
    dataset_path: str
        Path or identifier for the dataset to load.
    metric_path: str
        Path or identifier for the metrics to load.
    task: str
        Identifier that determines which dataset to load (in case of several).
    first_sentence_inputs: List[str]
        Dataset fields to use as first sentence input (concatenated).
    second_sentence_inputs: List[str]
        Dataset fields to use as second sentence input (concatenated).
    labels_to_predict: List[str]
        Dataset fields to use as labels.
    log_to_file: bool
        Whether to log to a file or output stream. In case of
        `run_as_test_case == True` always logs to file.
    logging_level: str
        Level of logging ('INFO', 'ERROR', ...).
    """

    model_name_or_path: str = field(default="dbmdz/bert-base-german-cased",
        metadata={ "help": "The model checkpoint for weights initialization."})

    dataset_path: str = field(default="../../dataset.py",
        metadata={"help": "Path or identifier for the dataset to load."})

    metric_path: str = field(default="../../metric.py",
        metadata={"help": "Path or identifier for the metrics to load."})

    task: str = field(default="stance_detection",
        metadata={"help": 'Identifier that determines which dataset to load.'})

    first_sentence_inputs: List[str] = field(default_factory = 
            lambda: ["question", "title"],
        metadata={"help": 'Dataset fields to use as first sentence input \
        (concatenated).'})

    second_sentence_inputs: Optional[List[str]] = field(default=None,
        metadata={"help": 'Dataset fields to use as second sentence input \
        (concatenated).'})

    labels_to_predict: List[str] = field(default_factory =
            lambda: ["stance", "emotions"],
        metadata={"help": 'Dataset fields to use as labels'})

    log_to_file: bool = field(default=False,
        metadata={"help": "Whether to log to a file or output stream. In case \
        of run_as_test_case == True always logs to file."})

    logging_level: str = field(default="INFO",
        metadata={"help": "Level of logging."})

@dataclass
class CHeeSETrainingArguments(TrainingArguments):
    """ Training arguments to run the **run.py** python script.
    For futher arguments check **transformers.TrainingArguments**
    (https://huggingface.co/transformers/main_classes/trainer.html)

    Parameters
    ----------
    output_dir: str
        The output directory where the model predictions and checkpoints will
        be written to
    do_cross_validation: bool
        Whether to do cross-validation
    cross_validation_folds: int
        How many folds to use for cross-validation in case of
        do_cross_validation == True
    predictions_columns_to_include: Optional[Union[List[str], str, Tuple[str],\
            List[Tuple[str]]]]
        Columns from dataset to additionally include into the predictions csv.
        This argument can be provided in different types. Either as:

        - A string of a single column name.
        - A list of column names.
        - A 2-tuple of strings where the first entry is the column name and the 
          second one the name this column will take in the output file.
        - A list of 2-tuples where each entry has as first string the column
          name and as second the name the column will take in the output file.
    prediction_csv_kwargs: Dict
        - Keyword arguments for the pd.to_csv call of the predictions.
    use_garbage_collector_callback: bool
        - Whether the garbage collection should be called whenever possible
          within the training procedure.
    """
    
    output_dir: str = field(default=None,
        metadata={"help": "The output directory where the model predictions \
        and checkpoints will be written to."})

    do_cross_validation: bool = field(default = False, metadata={"help": 
        "Whether to perform a crossvalidation testing."})

    cross_validation_folds: int = field(default=5, metadata={"help": 
        "The number of folds within the cross validation testing."})

    prediction_columns_to_include: Optional[Union[List[str], str, Tuple[str],
        List[Tuple[str]]]] = field(default=None, metadata={"help": "Columns \
        from dataset to additionally include into the predictions csv."})

    prediction_csv_kwargs: Dict = field(default_factory = lambda: {"sep": ","},
        metadata={"help": "Keyword arguments when while writing the \
        predictions csv (can be used for setting delimiter, ...)"})