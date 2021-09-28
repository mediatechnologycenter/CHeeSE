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

""" CHeeSE benchmark metrics for stance detection. """

import re
import datasets
import numpy as np
from typing import Any, Dict, List, Union
from sklearn.metrics import f1_score

STANCE_LABELS = ['Kein Bezug', 'Diskutierend', 'Ja, dafür', 'Nein, dagegen']
INT_TO_STR = [re.sub(r'[^\w\s]', '', i).replace(" ","-").lower() 
    for i in STANCE_LABELS]

_CITATION = """\
{
    @article{
        "reference" = "TODO",
        "author" = "Laura Mascarell, Tatyana Ruzsics, Christian Schneebeli, Philippe Schlattner, Luca Campanella, Severin Klingler, Cristina Kadar",
        "title" = "Towards Stance Detection in German News Articles",
        "journal" = FEVER,
        "year" = 2021
    }
}
"""

_DESCRIPTION = """\
Metrics for the baselines of Swiss Stance and Emotion Dataset (CHeeSE).
"""

_KWARGS_DESCRIPTION = """
Parameters
----------
predictions: np.array
    Arrays of predictions to score.
references: np.array
    List of references to score.

Returns
-------
metrics: Dict[str, float]
Depending on the dataset/task a subset, one or several of:
    - "accuracy": Accuracy
    - "f1": F1 score

Examples
--------

>>> import numpy as np
>>> from datasets import load_metric
>>> metric = load_metric('metric.py')  
>>> labels = np.array([0, 1, 1, 2, 3])
>>> predictions = np.array([0, 1, 0, 3, 2])
>>> results = metric.compute(predictions=predictions, labels=labels)
>>> print(results)
{'accuracy': 0.4, 'f1_macro': 0.3333333333333333,
'f1_micro': 0.4000000000000001, 'f1_weighted': 0.4,
'f1_unweighted_kein-bezug': 0.6666666666666666,
'f1_unweighted_diskutierend': 0.6666666666666666,
'f1_unweighted_ja-dafür': 0.0, 'f1_unweighted_nein-dagegen': 0.0}
"""

def simple_accuracy(preds: np.array, labels: np.array) -> float:
    """ Standard simple accuracy.

    Parameters
    ----------
    preds:
        Numpy array of predictions.
    labels:
        Numpy array of labels.

    Returns
    -------
    The accuracy of the predictions.
    """
    if len(preds.shape) > 1:
        preds = preds.reshape(-1)
    if len(labels.shape) > 1:
        labels = labels.reshape(-1)
    return (preds == labels).mean()
    
def f1_scores(preds: np.array, labels: np.array) \
        -> Dict[str, float]:
    """ F1 scores - from sklearn.metrics.f1_score for more infos.

    Parameters
    ----------
    preds:
        Numpy array of predictions.
    labels:
        Numpy array of labels.

    Returns
    -------
    A number of different f1 scores (differently averaged).
    """
    f1 = {}
    for average in ['macro', 'micro', 'weighted', None]:
        f1["f1_" + average if average else "f1_unweighted"] = f1_score(
            y_true=labels, y_pred=preds, labels=[0,1,2,3], average=average, 
            zero_division=1)
    return f1

@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION)
class CHeeSE(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            # # Is not enforced - depends on whether argmax provided on classes
            features=datasets.Features(
                {
                    "predictions": datasets.Value("int64"),
                    "labels": datasets.Value("int64"),
                }
            ),
            codebase_urls=[],
            reference_urls=[],
            format="numpy",
        )

    def get_all_metrics(self, predictions: np.array, labels: np.array) \
            -> Dict[str, float]:
        """ A set of different metrics that can be computed.

        Parameters
        ----------
        preds:
            Numpy array of predictions.
        labels:
            Numpy array of labels.

        Returns
        -------
        A number of metrics (accuracy, f1) (multiple f1's that are differently
        averaged).
        """
        metrics = {}
        metrics["accuracy"] = simple_accuracy(predictions, labels)
        metrics.update(f1_scores(predictions, labels))
        metrics = self.split_list_metrics(metrics)
        return metrics

    def split_list_metrics(self, metrics_to_split: Dict[str, Union[int, List]]):
        """ Splits up list entries (metrics where score is computed per class).

        Parameters
        ----------
        metrics_to_split:
            Dictionnary of metrics.

        Returns
        -------
        Dictionnary of metrics, where list metrics are split up.
        """
        metrics = dict(metrics_to_split)
        keys_to_delete = []
        entries_to_add = {}
        for key, value in metrics.items():
            if type(value) == list or type(value) == np.ndarray or \
                    type(value) == np.array:
                keys_to_delete.append(key)
                for ind, str_rep in enumerate(INT_TO_STR):
                    entries_to_add[key + "_" + str_rep] = value[ind]

        metrics.update(entries_to_add)
        for key in keys_to_delete:
            del metrics[key]
        return metrics
        
    @datasets.utils.file_utils.add_start_docstrings(_KWARGS_DESCRIPTION)
    def compute(self, predictions: np.array, labels: np.array) \
            -> Dict[str, float]:
        """ Overwriting the default compute method. """

        # Crossentropy - argmax over predictions if not done before
        if (labels.dtype == np.long or labels.dtype == np.int):
            if len(predictions.shape) == 2:
                predictions = np.argmax(predictions, axis=1)

        # If labels are provided with unnecessary second dimension remove
        if len(labels.shape) == 3:
            if labels.shape[1] == 1:
                labels = labels[:,0]

        # Return metrics
        return self.get_all_metrics(predictions=predictions, labels=labels)
