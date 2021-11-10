# CHeeSE

Dataset and baselines presented in the paper "Towards Stance Detection in
German News Articles". The dataset consists of debate questions and news
articles that are matched and annotated for stance detection.
It contains ~2000 new articles in German from the Swiss news papers "NZZ", "NZZ am
Sonntag" and "Blick" that are matched with 91 debate questions. On average each
article gets matched with ~1.9 questions resulting in ~3800 stance-annotated 
article-question pairs. Moreover, all articles and each of their paragraphs are
annotated with emotions, thus, enabling emotion detection at an article or
paragraph level. We believe, this hand-annotated dataset enables research in
many interesting areas, such as multi-task learning, transfer learning, emotion
flow in news articles, etc. The reprository also contains the code to reproduce
the stance detection baselines of the paper, namely the one using
fastText and the deep learning approach with a pretrained transformer (BERT)
using Huggingface.


## Dataset Structure


All the data is contained within the file `data/cheese.json`.
The data is structured as following:

```json
    {
        "title": "...",
        "snippet": "...",
        "paragraphs": [
            {
                "text": "...", 
                "paragraph_emotion": ["Angst", "Traurigkeit"],
            },
            {
                "text": "...", 
                "paragraph_emotion": ["Antizipation"],
            }
        ],
        "article_emotion": ["Angst", "Überraschung"],
        "article_stance": [
            {    
                "question_id": 10,
                "question": "...",
                "stance": "Nein, dagegen",
                "selection_stage": 1,
                "selection_rank": "gold",
                "general_area_of_interest": "...",
                "target_topic": "...",
            },
            {    
                "question_id": 53,
                "question": "...",
                "stance": "Kein Bezug",
                "selection_stage": 2,
                "selection_rank": "silver",
                "general_area_of_interest": "...",
                "target_topic": "...",
            }
        ],
        "article_id": "...",
        "source": "NZZ",
        "date": "...",
    }
    {
        ...
    }
```

The possible choices for **stance** are:

- "Kein Bezug" (unrelated)
- "Diskutierend" (discussing)
- "Ja, dafür" (in favor)
- "Nein, dagegen" (against)
- "Unklar" (unclear)

The possible choices for article/paragraph **emotion** are:

- "Angst" (fear)
- "Antizipation" (anticipation)
- "Ärger" (anger)
- "Ekel" (disgust)
- "Freude" (joy)
- "Keine" (none)
- "Traurigkeit" (sadness)
- "Überraschung" (surprise)
- "Vertrauen" (trust)
- "Unklar" (unclear)

## Baselines


The paper presents two baselines for stance detection:
- Fasttext classifier
- BERT classifier

**Data**

First, downlaod the data from https://projects.mtc.ethz.ch/cheese-data and place
the `cheese.json` file into CHeeSE/data.

**Setup**

We recommend using a python environment to install all the python packages
required for this project. Once setup, the packages can be installed with:

```bash
    pip3 install -r requirements.txt
```

System settings of the machine used for the results in the paper:
- CPU: Intel(R) Core(TM) i7-8700 @ 3.20GHz
- GPU: GeForce RTX 2070
- Python Version: Python 3.6
- CUDA Version: 11.2

**Fasttext Baseline**

The Fasttext baseline can be reproduced by running the script
`fasttext_baseline.py` within the baselines/stance_detection directory with the
following command:

```bash
    cd baselines/stance_detection
    python3 fasttext_baseline.py
```

**Bert Baseline**

The Bert baseline can be reproduced by running the script
`bert_baseline.py` within the baselines/stance_detection directory with the
following command and using the provided config file:

```bash
    cd baselines/stance_detection
    python3 bert_baseline.py --config bert_baseline_config.json
```

## Reference

The dataset and baseline models are presented in:

https://aclanthology.org/2021.fever-1.8.pdf

When using the CHeeSE data set for research purpose, please cite:

```
    @inproceedings{mascarell-etal-2021-stance,
        title = "Stance Detection in {G}erman News Articles",
        author = "Mascarell Laura, Ruzsics Tatyana, Schneebeli Christian, Schlattner Philippe, Campanella Luca, Klingler Severin, Kadar Cristina",
        booktitle = "Proceedings of the Fourth Workshop on Fact Extraction and VERification (FEVER)",
        month = nov,
        year = "2021",
        address = "Dominican Republic",
        publisher = "Association for Computational Linguistics",
        url = "https://aclanthology.org/2021.fever-1.8",
        pages = "66--77"
    }
```


## Acknowledgements

This project is supported by Ringier, TX Group, NZZ, SRG, VSM, viscom, and the ETH Zurich Foundation.
