from utils.load_data import load_data
from utils.preprocess import preprocess_data
from utils.tools import subsample

import json
from pathlib import Path

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
  Path('./checkpoint/v_sets.json', exist_ok=True).touch()
  json.dump(v_sets, open('./checkpoint/v_sets.json', 'w'))
  Path('./checkpoint/e_sets.json', exist_ok=True).touch()
  json.dump(e_sets, open('./checkpoint/e_sets.json', 'w'))


if __name__ == "__main__":
    main()