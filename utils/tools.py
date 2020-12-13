import pandas as pd
from collections import defaultdict
import time
import os
import utils

def subsample(v_data, e_data, core_target, ext_target, n=10000):
  t0 = time.time()

  e_sample = e_data.sample(n)
  v_list = list(pd.Categorical(list(e_sample.to_id)+list(e_sample.from_id)).categories)
  v_sample = v_data[v_data.index.isin(v_list)]
  core_sample = core_target[core_target.index.isin(v_list)]
  ext_sample = ext_target[ext_target.index.isin(v_list)]

  t1 = time.time()
  print(f"SUBSAMPLING: {(t1-t0):.2f} s")

  return v_sample, e_sample, core_sample, ext_sample

def load_preprocessed_data():
  #TODO add from_jup flag to change dirs (for robustness)

  v_sets = defaultdict()
  e_sets = defaultdict()

  path = "./data/processed"
  v_types = ['Account', 'Customer', 'Derived Entity', 'External Entity', 'Address']
  e_types = ['money transfer', 'is similar', 'has account', 'has address']

  for v_type in v_types:
      v_sets[v_type] = pd.read_csv(f"{path}/{v_type}.csv", low_memory=False, sep=',', index_col='node_id')
  for e_type in e_types:
      e_sets[e_type] = pd.read_csv(f"{path}/{e_type}.csv".replace(" ", ""), low_memory=False, sep=',', index_col='edge_id')

  return v_sets, e_sets


def smart_preprocess(v_data, e_data, core_targets, ext_targets, core_testing):
  path = "./data/processed"
  if os.path.exists(path):
      print("Loading preprocessed data from local directory")
      # load data from processed csvs
      v_sets, e_sets = load_preprocessed_data()

  else:
      print("Processed data not found. Preprocessing started.")
      v_sets, e_sets = utils.preprocess_data(v_data, e_data, core_targets, ext_targets, core_testing)

      # save data to data/processed
      os.makedirs(path, exist_ok=True)

      for v_type in v_sets:
          v_sets[v_type].to_csv(f"{path}/{v_type}.csv")
      for e_type in e_sets:
          e_sets[e_type].to_csv(f"{path}/{e_type}.csv".replace(" ", ""))

  return v_sets, e_sets


def load_for_jupyter():
  v_data, e_data, core_targets, ext_targets, core_testing = utils.load_data(from_jup=True)
  #v_sample, e_sample, core_sample, ext_sample = subsample(v_data, e_data, core_targets, ext_targets, n)
  v_sets, e_sets = utils.preprocess_data(v_data, e_data, core_targets, ext_targets, core_testing)
  return v_data, e_data, v_sets, e_sets, core_targets, ext_targets, core_testing

def load_for_jupyter_raw():
  v_data, e_data, core_targets, ext_targets, core_testing = utils.load_data(from_jup=True)
  return v_data, e_data, core_targets, ext_targets, core_testing