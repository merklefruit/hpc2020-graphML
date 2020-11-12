from utils.load_data import load_data
from utils.preprocess import preprocess_data
from utils.tools import subsample
from utils.visualization import get_TSNE

from model.sg_setup_1 import sg_DeepWalk

import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
  print("Running the scripts sequentially \n")

  # Load data
  v_data, e_data, core_targets, ext_targets = load_data()

  # Subsample data to n edges
  n = 10000
  v_sample, e_sample, core_sample, ext_sample = subsample(v_data, e_data, core_targets, ext_targets, n)

  # Preprocess data
  v_sets, e_sets = preprocess_data(v_sample, e_sample)

  # Save data to checkpoint
  # Path('./checkpoint/v_sets.json', exist_ok=True).touch()
  # json.dump(v_sets, open('./checkpoint/v_sets.json', 'w'))
  # Path('./checkpoint/e_sets.json', exist_ok=True).touch()
  # json.dump(e_sets, open('./checkpoint/e_sets.json', 'w'))

  #! DEEP WALK
  node_ids, embeddings, core_t, ext_t = sg_DeepWalk(v_sets, e_sets, v_sample, e_sample)

  # Visualize embeddings with TSNE
  embs_2d = get_TSNE(embeddings)

  # Draw the embedding points, coloring them by the target label (CaseID)
  alpha = 0.6
  label_map = {l: i for i, l in enumerate(np.unique(ext_t), start=1000) if pd.notna(l)}
  label_map[0] = 1
  node_colours = [label_map[target] if pd.notna(target) else 0 for target in ext_t]

  plt.figure(figsize=(15, 15))
  plt.axes().set(aspect="equal")
  plt.scatter(
      embs_2d[:, 0],
      embs_2d[:, 1],
      c=node_colours,
      cmap="jet",
      alpha=alpha,
  )
  plt.title("TSNE visualization of node embeddings w.r.t. Extended Case ID")
  plt.show()

  return 1

if __name__ == "__main__":
    main()