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

import re
import pprint
import numpy as np
import fasttext

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

from sacremoses import MosesTokenizer
from sklearn.model_selection import StratifiedKFold
from datasets import load_dataset, load_metric, concatenate_datasets, Dataset

SEED = 42
FOLDS = 5

SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
REMOVE_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('german'))
TOKENIZER = MosesTokenizer(lang='de')

# Prepare text, i.e. clean it
def prepare_text(text):
    # Lowercase and remove line breaks within text
    text = text.replace('\n', ' ').lower()
    # Replace by space within text
    text = SPACE_RE.sub(' ',text)
    # Remove from text
    text = REMOVE_RE.sub('',text)
    # Tokenize and delete stopwords from text
    text = ' '.join([w for w in TOKENIZER.tokenize(text) if not w in STOPWORDS])
    return text

# Prepare input
def prepare_input(sample, prepare=True):
    def get_input(samp):
        return (samp['question'] + "<SEP>" +
                samp['title'] + "<SEP>" +
                samp['snippet'] + "<SEP>" +
                samp['paragraphs'])
    return {"input": prepare_text(get_input(sample)) if prepare else
        get_input(sample)}

# Dataset
path = '../../dataset.py'
dataset = load_dataset(path, 'stance_detection')
train_dataset = dataset['train']
valid_dataset = dataset['validation']
test_dataset = dataset['test']

# Metric
metric=load_metric('../../metric.py')  

# Concatenate all data
dataset = concatenate_datasets((train_dataset,valid_dataset,test_dataset))
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
scores = None

# Prepare input and clean
dataset = dataset.map(prepare_input, batched=False)

for train_index, test_index in skf.split(dataset, dataset["stance"]):
    print("\nNext split...\n")
    # Data split
    train_dataset = Dataset.from_dict(dataset[train_index])
    test_dataset = Dataset.from_dict(dataset[test_index])
    
    # Build data
    X_train = train_dataset['input']
    X_test = test_dataset['input']
    y_train = train_dataset['stance']
    y_test = test_dataset['stance']
    
    with open("train.txt", "w") as file:
        for i in range(len(X_train)):
            file.write(("__label__" + dataset.features['stance']._int2str[
                y_train[i]].replace(" ", "-") + " " + X_train[i] + "\n"))
            
    with open("test.txt", "w") as file:
        for i in range(len(X_test)):
            file.write(("__label__" + dataset.features['stance']._int2str[
                y_test[i]].replace(" ", "-") + " " + X_test[i] + "\n"))
    
    # Classifier
    model = fasttext.train_supervised(input="train.txt", epoch=50, lr=1.0,
        seed=SEED)
    y_test_predicted_labels_mybag = [dataset.features['stance']._str2int[
        model.predict(i)[0][0][9:].replace("-", " ")] for i in X_test]
    
    # Results
    result = metric.compute(predictions=np.array(y_test_predicted_labels_mybag), 
        labels=np.array(y_test))

    pprint.pprint(result)
    
    if scores is None:
        scores = result
    else:
        for i in scores:
            scores[i] += result[i]

            
for i in scores:
    scores[i] = scores[i] / FOLDS

print("\nFinal scores: ")
pprint.pprint(scores)