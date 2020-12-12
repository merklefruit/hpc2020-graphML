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

The final pipeline is composed of:

1. Loading data
2. Preprocessing data
3. An embedding model: 2 HinSAGE layers that produce node embeddings (32-dimensional tensors), trained with Deep Graph Infomax in a semi-supervised way. This must be repeated for each node type (5 times).
4. A Decision Tree classifier that takes as input the node embeddings and outputs the predicted Case ID.

> Please refer to my [written report](linktoreport) for all the implementation specific details.

## External Libraries

The main libraries leveraged in this solutions are:

- Stellargraph (with Tensorflow backend)
- Pandas
- Numpy
- Sklearn

You can find the complete list of the dependencies as well as their version in `requirements.txt`.

Please note that some libraries were only used in the development phase and are not relevant to the final solution but might be only used in some notebooks. The ones mentioned above are the ones that I used the most.
