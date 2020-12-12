import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils.load_data import load_data
from utils.preprocess import preprocess_data
from utils.tools import subsample

from models import sg_DeepWalk, createEmbeddings

def main():
  print("Running the scripts sequentially \n")

  # Load data
  v_data, e_data, core_targets, ext_targets, core_testing = load_data()

  # Subsample data to n edges
  n = 10000
  v_sample, e_sample, core_sample, ext_sample = subsample(v_data, e_data, core_targets, ext_targets, n)

  # Preprocess data
  v_sets, e_sets = preprocess_data(v_sample, e_sample, core_targets, ext_targets, core_testing)
  
  '''
  #? DEEP WALK (demo method for reference)
  node_ids, embeddings, core_t, ext_t = sg_DeepWalk(v_sets, e_sets, v_sample, e_sample)
  '''

  '''
  #? Deep Graph Infomax embeddings (deprecated, old solution)
  full_graph_embeddings = createEmbeddings(v_sets, e_sets, core_sample, ext_sample, v_sample, e_sample)
  '''

  #? Run the entire pipeline: embedding + classification (current solution)
  

  return v_sets, e_sets, core_sample, ext_sample, v_sample, e_sample

if __name__ == "__main__":
    main()