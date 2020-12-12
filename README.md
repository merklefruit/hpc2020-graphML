# Oracle High Performance Computing Contest - Fall 2020

_Author: **Nicolas Racchi**, BSc Automation Engineering student at [Polimi](https://www.polimi.it)_

> This repository contains all the content for my submission to the **Graph ML** contest.

## Installing and running my submission

The easiest way to run my submission is by using the `run.py` script I prepared, which will run the final submission and output the testing cases predictions.

NOTE: The dataset is pulled from the original [repository](https://github.com/AlbertoParravicini/high-performance-graph-analytics-2020/tree/main/track-ml/) of the contest, so you don't have to install anything manually. Refer to (utils/load_data) for more info.

```bash
# 1. Clone the repo
git clone https://github.com/nicolas-racchi/hpc2020-graphML

# 2. cd in the project folder
cd hpc2020-graphML

# 3. (Optional): activate your virtualenv
virtualenv venv && source venv/bin/activate

# 4. install dependencies
pip install -r requirements.txt

# 5. run the main script
python run.py
```

## Solution overview

The final solution is composed of:

1. An embedding model: 2 HinSAGE layers that produce node embeddings (32-dimensional tensors), trained with Deep Graph Infomax in a semi-supervised way.
2. A classifier that takes as input the embeddings, that predicts whether each node could be fraudolent or not (aka Binary classifier predicting 1 if the node should have ExtendedCaseID and 0 if it shouldn't).
3. A classifier that takes as input the embeddings of the aforementioned fraudolent nodes and outputs their ExtendedCaseID.

The reason for the second classifier is to remove class imbalance with regard to the ExtendedCaseID=0 class.

> Please refer to my written report for all the implementation specific details.

## External Libraries

The main libraries leveraged in this solutions are:

- Stellargraph (with Tensorflow backend)
- Pandas
- Numpy
- Sklearn

You can find the complete list of the dependencies as well as their version in `requirements.txt`.

Please note that some libraries were only used in the development phase and are not relevant to the final solution. The ones mentioned above are the ones that have the most important roles in the final submission.
