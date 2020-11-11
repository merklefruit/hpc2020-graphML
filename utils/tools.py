import pandas as pd

def subsample(v_data, e_data, core_target, ext_target, n=10000):
  e_sample = e_data.sample(n=10000)
  v_list = list(pd.Categorical(list(e_sample.to_id)+list(e_sample.from_id)).categories)
  v_sample = v_data[v_data.index.isin(v_list)]
  core_sample = core_target[core_target.index.isin(v_list)]
  ext_sample = ext_target[ext_target.index.isin(v_list)]
  return v_sample, e_sample, core_sample, ext_sample

