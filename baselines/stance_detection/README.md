

# Manual

Instead of writing own training/evaluation scripts for bert based models using
the custom huggingface dataset/metric loaders within this project
(`../../dataset.py`, `../../metric.py`), the `bert_baseline.py` script can
be used/extended.

The script can be run within the `baselines/stance_detection` directory with:

```bash
    $ python3 bert_baseline.py
```

Arguments can be provided through one or several `.json` config files - where
latter ones overwrite values of earlier ones. In addition command line arguments
can be provided, which have the priority on all given arguments
(cli/json/default).

The order of priority in which the provided arguments are considered is:

    - cli arguments
    - last config file
    - ...
    - first config file
    - dataclass defaults

To provide a single config file the script can be run as following:

```bash
    $ python3 bert_baseline.py --config <path_to_config_file>
```

To provide multiple config files the script can be run as following:

```bash
    $ python3 bert_baseline.py --config <path_to_config_file1> <path_to_config_file2> ...
```

Any argument that can be defined in json config file, i.e. is present in one of
the argument dataclasses, can also be provided as a command
line arguments as following:

```bash
    $ python3 bert_baseline.py --config <path_to_config_file> --num_train_epochs 10
```

All arguments to the `bert_baseline.py` script can be provided among others as a .json
script. The order in which they are specified does not matter. In addition, it
is possible to provide comments wihtin the json by adding entries whose name
starts with a underline (e.g. "_CHeeSEArguments"). This can help keeping
the arguments organized. For most arguments the possible choices can be checked
out in baselines/stance_detection/utils/arguments. However, for argument
dataclasses that inherit from Huggingface's arguments dataclasses
(such as TrainingArguments), the remaining possibilities can be directly checked
out within, e.g. [Huggingface's TrainingArguments](https://huggingface.co/transformers/main_classes/trainer.html#transformers.TrainingArguments/)

The config to reproduce the baseline of the paper can be found in at
`baselines/stance_detection/bert_baseline_config.json`.

An example of a config file looks as following:

```json
    {
        "_CHeeSEArguments": "CHeeSEArguments",

        "model_name_or_path": "dbmdz/bert-base-german-cased",
        "dataset_path": "dataset/CHeeSE/dataset.py",
        "metric_path": "metrics/metric.py",
        "task": "stance",
        "first_sentence_inputs": ["question"],
        "second_sentence_inputs": ["title", "headline", "paragraphs"],
        "labels_to_predict": ["stance"],


        "_CHeeSETrainingArguments": "CHeeSETrainingArguments",

        "do_train": true,
        "do_eval": false,
        "do_predict": true,
        "do_cross_validation": true,
        
        "output_dir": "outputs/Fever/CHeeSE",
        "logging_dir": "outputs/Fever/CHeeSE/log",
        "save_total_limit": 2,
        "log_to_file": true,
        "logging_strategy": "steps",
        "logging_steps": 50,
        "cross_validation_folds": 5,

        "num_train_epochs": 4,
        "per_device_train_batch_size": 5,
        "per_device_eval_batch_size": 5,
        "fp16": false,
        "learning_rate": 3e-05,

        "prediction_csv_kwargs": {"index": false}
    }
```

## Running/reproducing experiments

**1. Running a new experiment**

A new experiment can be created by providing at least all arguments without
default either by command line or with a/several json file(s).

``` bash
    $ python3 bert_baseline.py --config config.json --output_dir outputs/example
```

**2. Reproducing an experiment**

When running the script, a new run_config.json with all the none-default
arguments (i.e. merged from all input configs, both json and cli) will be
written to the output directory, where it can be double-checked or used to
reproduce an experiment (by providing it again to the experiment `bert_baseline.py` 
script). 

```bash
    $ python3 bert_baseline.py --config outputs/example/run_config.json
```


The experiment can also be reproduced by simply providing all the initial 
arguments to the script again.