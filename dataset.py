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

""" CHeeSE: Swiss Stance and Emotion Dataset. """

import os
import datasets
import pandas as pd
from typing import List, Dict
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

logger = datasets.logging.get_logger(__name__)

_CITATION = """\
{       
    @inproceedings{mascarell-etal-2021-stance,
        title = "Stance Detection in {G}erman News Articles",
        author = "Mascarell, Laura  and
            Ruzsics, Tatyana  and
            Schneebeli, Christian  and
            Schlattner, Philippe  and
            Campanella, Luca  and
            Klingler, Severin  and
            Kadar, Cristina",
        booktitle = "Proceedings of the Fourth Workshop on Fact Extraction and VERification (FEVER)",
        month = nov,
        year = "2021",
        address = "Dominican Republic",
        publisher = "Association for Computational Linguistics",
        url = "https://aclanthology.org/2021.fever-1.8",
        pages = "66--77",
        abstract = "The widespread use of the Internet and the rapid dissemination of information poses the challenge of identifying the veracity of its content. Stance detection, which is the task of predicting the position of a text in regard to a specific target (e.g. claim or debate question), has been used to determine the veracity of information in tasks such as rumor classification and fake news detection. While most of the work and available datasets for stance detection address short texts snippets extracted from textual dialogues, social media platforms, or news headlines with a strong focus on the English language, there is a lack of resources targeting long texts in other languages. Our contribution in this paper is twofold. First, we present a German dataset of debate questions and news articles that is manually annotated for stance and emotion detection. Second, we leverage the dataset to tackle the supervised task of classifying the stance of a news article with regards to a debate question and provide baseline models as a reference for future work on stance detection in German news articles.",
    }
}
"""

_DESCRIPTION = """\
Swiss Stance and Emotion Dataset (CHeeSE) is an emotion and stance detection
dataset, consisting of hand-labeled (news article, question) pairs where 
each pair includes the following labels:
- Stance of the article (A) towards the question (Q)
- Global emotions of the article (A)
- Emotions of each article's (A) paragraph 
"""

HOMEPAGE = "https://mtc.ethz.ch/research/natural-language-processing/emotion-stance.html"
AMBIGUOUS_LABEL = 'Unklar'
STANCE_LABELS = ['Kein Bezug', 'Diskutierend', 'Ja, dafür', 'Nein, dagegen']
EMOTION_LABELS = ['Freude', 'Traurigkeit', 'Keine', 'Antizipation',
                  'Ärger', 'Vertrauen', 'Ekel', 'Angst', 'Überraschung']
BASE_PATH = "../../data/"
SEED = 2021

@dataclass
class CHeeSEConfig(datasets.BuilderConfig):
    """BuilderConfig for CHeeSE Dataset.
    
    Parameters
    ----------
    task: str
        The predictions task {"stance"}
    """
    task: str = "stance_detection"
    _id_counter: int = 0
    train_size = 0.75
    valid_size = 0.05


class CHeeSE(datasets.GeneratorBasedBuilder):
    """ CHeeSE: Swiss Stance and Emotion Dataset. 
    
    Examples
    --------
    To load the dataset use the **datasets.load_dataset** script. Check the
    CHeeSEConfig class attribute's `names` within this class for more task
    options (second argument to the `load_dataset` method). 

    >>> from datasets import load_dataset
    >>> path = 'dataset.py'
    >>> dataset = load_dataset(path, 'stance_detection')
    >>> print(dataset)
    DatasetDict({
        train: Dataset({
            features: ['article_id', 'general_area_of_interest', 'target_topic',
                       'selection_stage', 'selection_rank', 'source',
                       'question_id', 'question', 'title', 'snippet',
                       'paragraphs', 'stance'],
            num_rows: 2368
        })
        validation: Dataset({
            features: ['article_id', 'general_area_of_interest', 'target_topic',
                       'selection_stage', 'selection_rank', 'source',
                       'question_id', 'question', 'title', 'snippet',
                       'paragraphs', 'stance'],
            num_rows: 158
        })
        test: Dataset({
            features: ['article_id', 'general_area_of_interest', 'target_topic',
                       'selection_stage', 'selection_rank', 'source',
                       'question_id', 'question', 'title', 'snippet',
                       'paragraphs', 'stance'],
            num_rows: 632
        })
    })
    """

    STANCE = CHeeSEConfig(
            name="stance_detection",
            version=datasets.Version("1.0.0", ""),
            description="Stance prediction from unique file with all data.",
            task="stance_detection"
        )
    
    BUILDER_CONFIG_CLASS = CHeeSEConfig

    BUILDER_CONFIGS = [ 
            STANCE
        ]

    def _info(self):
        """ Sets the own info based on self.config.name. """

        if "stance_detection" == self.config.task:
            return datasets.DatasetInfo(
                description=_DESCRIPTION,
                features=datasets.Features(
                    {
                        "article_id": datasets.Value("string"),                                     
                        "general_area_of_interest": datasets.Value("string"),   
                        "target_topic": datasets.Value("string"),             
                        "selection_stage": datasets.Value("int32"),
                        "selection_rank": datasets.Value("string"),               
                        "source": datasets.Value("string"),
                        "question_id": datasets.Value("int32"),                         
                        "question": datasets.Value("string"),                     
                        "title": datasets.Value("string"), 
                        "snippet": datasets.Value("string"),                    
                        "paragraphs": datasets.Value("string"),
                        "stance": datasets.ClassLabel(len(STANCE_LABELS),       
                            STANCE_LABELS),
                    }
                ),
                homepage=HOMEPAGE,
                citation=_CITATION,
            )

    def _split_generators(self, dl_manager):
        """ Creates the different dataset split for train/validation/test. """

        file_path = os.path.join(BASE_PATH, "cheese.json")
        df = pd.read_json(file_path, lines=True,  orient="records")
        
        if self.config.name == "stance_detection":
            col = 'article_stance'
            df = pd.concat([df.explode(col).drop([col], axis=1),
                            df.explode(col)[col].apply(pd.Series)], axis=1)
            df = df[df.stance != "Unklar"]

            # First split train_valid/test
            train_valid, test = train_test_split(df, train_size=
                self.config.train_size+self.config.valid_size,
                random_state=SEED, shuffle=True, stratify=df['stance'])
                
            # Split train_valid
            train, valid = train_test_split(train_valid, train_size=
                1-self.config.valid_size/(self.config.train_size +
                self.config.valid_size), random_state=SEED, shuffle=True, 
                stratify=train_valid['stance'])
    
            dfs = { "train": train,
                    "valid": valid,
                    "test": test}
        
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN,
                gen_kwargs={"df": dfs["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION,
                gen_kwargs={"df": dfs["valid"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, 
                gen_kwargs={"df": dfs["test"]}),
        ]

    def _generate_examples(self, df):
        """ Generates examples from json files.

        Parameters
        ----------
        df:
            Dataframe with data.

        Yields
        ------
        id: int
            The id of the example.
        content: dict
            A mapping from names to values of the example.
            
        """
        for _, row in df.iterrows():
            
            sample = dict()
            if self.config.task == "stance_detection":
                sample = dict(row)
                sample['paragraphs'] = "\n".join(
                    [i['text'] for i in row['paragraphs']])
                del sample['article_emotion']
                
            yield self.config._id_counter, sample
            self.config._id_counter += 1
