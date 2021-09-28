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

import os, sys
sys.path.append(os.path.join(os.pardir, os.pardir))

import argparse
import numpy as np
from typing import List 
from datasets import load_dataset, load_metric, concatenate_datasets, Dataset
from sklearn.model_selection import StratifiedKFold, train_test_split

from transformers import set_seed
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
    Trainer)

from utils.train_utils import get_num_classes
from utils.parser import CHeeSEParser
from utils.arguments import CHeeSEArguments, CHeeSETrainingArguments
from utils.train_utils import (get_last_checkpoint, setup_logging, save_config,
    store_mertics, get_run_config_from_checkpoint, store_predictions)

def main(logger, args, training_args, config_paths=None, 
        cli_args=None):

    # GET CHECKPOINT (if exists)
    last_checkpoint = get_last_checkpoint(training_args)
    if last_checkpoint is not None:
        checkpoint_config = get_run_config_from_checkpoint(last_checkpoint)
        hf_parser = CHeeSEParser((CHeeSEArguments))
        checkpoint_args = hf_parser.parse_json_file(checkpoint_config)[0]
        assert(checkpoint_args.model_name_or_path == args.model_name_or_path), \
            "Last checkpoint for training at this path was with another model."

    # SET SEED FOR REPRODUCABIlITY
    set_seed(training_args.seed)

    # LOAD DATA
    logger.info("LOADING THE DATA\n")
    raw_datasets = load_dataset(args.dataset_path, args.task)

    # LOAD METRICS
    logger.info("LOADING THE METRICS\n")

    # Custom wrapper where metrics are annotated with associated label
    labels_to_predict = args.labels_to_predict
    key = next(iter(raw_datasets))
    metric = load_metric(args.metric_path, args.task)

    # LOAD TOKENIZER
    logger.info("LOADING THE TOKENIZER\n")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
        use_fast=True)

    # SET LABEL - key "labels" is expected
    def set_labels(examples):
        return {"labels": [float(examples[label]) if len(labels_to_predict)
            > 1 else examples[label] for label in labels_to_predict]}
    raw_datasets = raw_datasets.map(set_labels, batched=False)
    args.labels_to_predict = ["labels"]

    # TOKENIZE THE INPUT
    def tokenize_function(examples): 
        # First sentence and second sentence parts are concatenated with
        # Sep-tokens in between.
        first_sentence = (" " + tokenizer.sep_token + " ").join([examples[i]
            for i in args.first_sentence_inputs])
        if args.second_sentence_inputs is not None:
            second_sentence = (" " + tokenizer.sep_token + " ").join(
                [examples[i] for i in args.second_sentence_inputs])
        else:
            second_sentence = None

        # Tokenize part
        tokenized = tokenizer(first_sentence, second_sentence,
            padding="max_length", truncation=True)
        return tokenized

    # Tokenize
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=False)

    # DEFINE DATASETS
    logger.info("SPLITTING THE DATA\n")
    if training_args.do_train:
        train_dataset = tokenized_datasets["train"]
    if training_args.do_eval or training_args.do_cross_validation:
        valid_dataset = tokenized_datasets["validation"]
    if training_args.do_predict:
        test_dataset = tokenized_datasets["test"]

    # LOAD MODEL
    def model_init(trial):
        """ Method to load a model. Trial parameter used for providing
        parameters within hyperparameter search. """
        
        logger.info("LOADING THE MODEL\n")
        trial = {} if trial is None else trial.params
        model_kwargs = {} 

        # Reset seed each time this is called - i.e. for hyperparameter search
        set_seed(training_args.seed)
        
        # Number of classes for each label
        num_labels = get_num_classes(tokenized_datasets["train"], 
            labels_to_predict)

        # AutoModelForSequenceClassification can either predict several 
        # binary labels or one multi-class label
        if len(num_labels) == 1:
            num_labels = num_labels[next(iter(num_labels))]
        else:
            assert(all([num_labels[key] == 2 for key in num_labels])), \
                ("Multi-labels have to be binary")
            num_labels = len(num_labels)
        model_kwargs = {"num_labels": num_labels}

        # Load the model
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path, **model_kwargs)

        return model

    # TRAINER + ARGS
    training_args.label_names = ['labels']

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=valid_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        compute_metrics=lambda x: metric.compute(*x)
    )
    
    # SAVE CONFIG FILE
    save_config(training_args.output_dir, config_paths=config_paths,
        cli_args=cli_args)
    
    # do_crossvalidation
    if training_args.do_cross_validation:

        # Store metrics from all runs
        valid_metrics = []
        test_metrics = []

        # Concatenate all data
        dataset = concatenate_datasets((train_dataset,valid_dataset,
            test_dataset))

        skf = StratifiedKFold(n_splits=training_args.cross_validation_folds,
            shuffle=True, random_state=training_args.seed)

        for train_index, test_index in skf.split(dataset,
                dataset[labels_to_predict[0]]):

            # Data split
            train_dataset = Dataset.from_dict(dataset[train_index])
            test_dataset = Dataset.from_dict(dataset[test_index])
            if training_args.do_eval:
                train_dataset, valid_dataset = train_test_split(train_dataset,
                train_size=0.95, random_state=training_args.seed, shuffle=True,
                stratify=train_dataset[labels_to_predict[0]])
                train_dataset = Dataset.from_dict(train_dataset)
                valid_dataset = Dataset.from_dict(valid_dataset)

            # Trainer
            trainer = Trainer(
                model_init=model_init,
                args=training_args,
                train_dataset=train_dataset if training_args.do_train else None,
                eval_dataset=valid_dataset if training_args.do_eval else None,
                tokenizer=tokenizer,
                compute_metrics=lambda x: metric.compute(*x)
            ) 

            # Results from training (metrics/predictions)
            results = train_valid_test(logger=logger,
                training_args=training_args, last_checkpoint=last_checkpoint,
                trainer=trainer, test_dataset=test_dataset, labels_to_predict=
                labels_to_predict, to_file=False)
            valid_metrics.append(results[0])
            test_metrics.append(results[1])
        
        if training_args.do_eval:
            # Mean validation metrics
            valid_metrics = {k: (sum([i[k] for i in valid_metrics]) / 
                len(valid_metrics)) for k in valid_metrics[0]}
            eval_metrics_path = os.path.join(training_args.output_dir,
                "validation_results.txt")
            if trainer.is_world_process_zero():
                store_mertics(valid_metrics, logger=logger,
                    file_path=eval_metrics_path)

        if training_args.do_predict:
            # Mean test metrics
            test_metrics = {k: (sum([i[k] for i in test_metrics]) / 
                len(test_metrics)) for k in test_metrics[0]}
            test_metrics_path = os.path.join(training_args.output_dir,
                        "test_results.txt")
            if trainer.is_world_process_zero():
                store_mertics(test_metrics, logger=logger, info="Test",
                    file_path=test_metrics_path)
        
    else:
        train_valid_test(logger=logger, training_args=training_args,
        last_checkpoint=last_checkpoint, trainer=trainer,
        test_dataset=test_dataset, labels_to_predict=labels_to_predict)


def train_valid_test(logger, training_args, last_checkpoint, trainer,
        test_dataset, labels_to_predict, to_file=True):

    # TRAINING
    if training_args.do_train:
        logger.info("*** Train ***")
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        else:
            checkpoint = None

        # Start training
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        # Saves the tokenizer too for easy upload
        trainer.save_model()  

        # Write train stats
        output_train_file = os.path.join(training_args.output_dir, 
            "train_results.txt")

        if trainer.is_world_process_zero() and to_file:
            with open(output_train_file, "w") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(train_result.metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")
            
            # Need to save the state, since Trainer.save_model saves only the 
            # tokenizer with the model
            trainer.state.save_to_json(os.path.join(training_args.output_dir,
                "trainer_state.json"))

    # EVALUATION
    valid_metrics = None
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        valid_metrics = metrics
        eval_metrics_path = os.path.join(training_args.output_dir,
            "validation_results.txt")
        if trainer.is_world_process_zero() and to_file:
            store_mertics(metrics, logger=logger, file_path=eval_metrics_path)
    
    # PREDICTIONS
    test_metrics = None
    if training_args.do_predict:
        logger.info("*** Predict ***")
        predictions = trainer.predict(test_dataset)
        if len(predictions.predictions.shape) == 2:
            # Argmax for prediction not applied yet
            preds = np.argmax(predictions.predictions, axis=1)
        else:
            # Argmax for prediction already applied
            preds = predictions.predictions
        if hasattr(predictions, "metrics"):
            test_metrics = predictions.metrics
        
        if to_file:
            store_predictions(predictions=preds, dataset=test_dataset,
                labels_to_predict=labels_to_predict,
                prediction_columns_to_include=
                training_args.prediction_columns_to_include,
                output_dir=training_args.output_dir,
                prediction_csv_kwargs=training_args.prediction_csv_kwargs)

            if hasattr(predictions, "metrics"):
                metrics = predictions.metrics
                test_metrics_path = os.path.join(training_args.output_dir,
                    "test_results.txt")
                if trainer.is_world_process_zero() and to_file:
                    store_mertics(metrics, logger=logger, info="Test",
                        file_path=test_metrics_path)

    return (valid_metrics, test_metrics)

if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser(description='Stance Detection')
    parser.add_argument('--config', type=str, default=None, nargs="*",
        help="Cofig file with additional arguments.")
    config_paths = [os.path.abspath(i) for i in
        parser.parse_known_args()[0].config]

    hf_parser = CHeeSEParser((CHeeSEArguments, CHeeSETrainingArguments))
    args, training_args, remaining = \
        hf_parser.parse_args_into_dataclasses_with_default(
            return_remaining_strings=True, json_default_files=config_paths)
    cli_args = hf_parser.parse_known_args(with_default=False)[0]

    # Check that output dir is set
    if training_args.output_dir is None:
        raise ValueError("Output directory is not set. Please set it.")

    # LOGGING
    logger = setup_logging(args, training_args)

    # Check whether all the arguments are consumed
    if "--config" in remaining:
        for _ in range(len(config_paths)):
            remaining.remove(remaining[remaining.index("--config")+1])
        remaining.remove("--config")
    if len(remaining) > 0:
        raise RuntimeError("There are remaining attributes that could not "
            "be attributed: {}".format(remaining))

    main(logger=logger, args=args, training_args=training_args, 
        config_paths=config_paths, cli_args=cli_args)