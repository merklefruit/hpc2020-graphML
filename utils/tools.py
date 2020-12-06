import pandas as pd
import time
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

def load_for_jupyter():
  v_data, e_data, core_targets, ext_targets, core_testing = utils.load_data()
  #v_sample, e_sample, core_sample, ext_sample = subsample(v_data, e_data, core_targets, ext_targets, n)
  v_sets, e_sets = utils.preprocess_data(v_data, e_data)
  return v_data, e_data, v_sets, e_sets, core_targets, ext_targets, core_testing

def load_for_jupyter_raw():
  v_data, e_data, core_targets, ext_targets, core_testing = utils.load_data()
  return v_data, e_data, core_targets, ext_targets, core_testing