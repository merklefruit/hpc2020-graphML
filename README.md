# Oracle High Performance Computing Contest - Fall 2020

_Author: **Nicolas Racchi**, BSc Automation Engineering student at [Polimi](https://www.polimi.it)_

> This repository contains all the content for my submission to the **Graph ML** contest.

## Installing and running my submission

The easiest way to run my submission is by using the `run.py` script I prepared, which will run the HinSAGE with DGI + Decision tree classifier models and output final metrics and results.

NOTE: The dataset is pulled from the original [repository](https://github.com/AlbertoParravicini/high-performance-graph-analytics-2020/tree/main/track-ml/) of the contest, so you don't have to install anything manually.

```bash
# 1 Clone the repo
git clone https://github.com/nicolas-racchi/hpc2020-graphML

# 2 cd in the project folder
cd hpc2020-graphML

# 3 (Optional): activate your virtualenv & install dependencies
virtualenv venv && source venv/bin/activate
pip install -r requirements.txt

# 4 run the script
python run.py
```

## Written Report

Please refer to my written report for all the implementation specific details.
