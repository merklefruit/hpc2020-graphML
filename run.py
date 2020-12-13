import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils import load_data, smart_preprocess, subsample
from models import sg_DeepWalk, createEmbeddings, DGIPipeline

'''
Main script -
  When you run the script for the first time, the dataset is downloaded from the upstream repo
  of the contest, and it runs the final submission script (full pipeline).
  To see other methods instead, uncomment the necessary lines.

  After the first run, the dataset is pulled from the local folder (data/raw/..) and the processed 
  data is pulled from the local folder (data/processed/..) in order to save time.
'''
def main():
  print("Running the scripts sequentially \n")

  #? Load data - either remotely or locally if already downloaded
  v_data, e_data, core_targets, ext_targets, core_testing = load_data(from_jup=False)

  #? Subsample helper function (n edges)
  #Note: this is commented since I'm training the embedding model on the whole graph.
  #n = 10000
  #v_data, e_data, core_targets, ext_targets = subsample(v_data, e_data, core_targets, ext_targets, n)

  #? Preprocess helper function
  v_sets, e_sets = smart_preprocess(v_data, e_data, core_targets, ext_targets, core_testing)

  '''
  Old methods:
  - DeepWalk
  - Deep Graph Infomax embeddings (only used to evaluate embeddings with unoptimal parameters)
  - Full pipeline: the actual submission script.
  '''

  #? DEEP WALK (demo method for reference)
  #node_ids, embeddings, core_t, ext_t = sg_DeepWalk(v_sets, e_sets, v_data, e_data)


  #? Deep Graph Infomax embeddings (deprecated, old solution)
  #full_graph_embeddings = createEmbeddings(v_sets, e_sets, core_targets, ext_targets, v_data, e_data)


  #? Run the entire pipeline: embedding + classification (current solution)
  DGIPipeline(v_sets, e_sets, v_data, e_data, core_targets, ext_targets, core_testing)

  return 1

if __name__ == "__main__":
    main()