import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
pd.set_option('mode.chained_assignment', None)

# preprocessing of heterogeneous nodes and edges
def preprocess_data(v_sample, e_sample):
  t0 = time.time()

  #? 1: missing core/ext case ID
  # Solution: set to 0 if NaN

  v_sample.CoreCaseGraphID = v_sample.CoreCaseGraphID.fillna(0)
  v_sample.ExtendedCaseGraphID = v_sample.ExtendedCaseGraphID.fillna(0)

  #? 2: Create dataframes for each node type (Account, Customer, Derived entity, External entity, Address)
  # and each edge type (has account, has address, is similar, money transfer)

  v_sets = defaultdict()
  for v_type in list(pd.Categorical(v_sample.Label).categories):
      v_sets[v_type] = v_sample[v_sample.Label == v_type]
      v_sets[v_type] = v_sets[v_type].drop(['Label', 'testingFlag']+list(v_sets[v_type].columns[v_sets[v_type].isnull().all()]), axis=1)

  e_sets = defaultdict()
  for e_type in list(pd.Categorical(e_sample.Label).categories):
      e_sets[e_type] = e_sample[e_sample.Label == e_type]
      e_sets[e_type] = e_sets[e_type].drop(['Label']+list(e_sets[e_type].columns[e_sets[e_type].isnull().all()]), axis=1)
      e_sets[e_type] = e_sets[e_type].rename(columns={'from_id':'source', 'to_id':'target'})

  #? 3: Logical conversion of categorical features

  #Revenue Size Flag: low, mid_low, medium, mid_high, high -> 1,2,3,4,5
  conversion = {'low':1, 'mid_low':2, 'medium':3, 'mid_high':4, 'high':5}
  for i in v_sets:
    if 'Revenue Size Flag' in list(v_sets[i].columns):
      v_sets[i]['Revenue Size Flag']=v_sets[i]['Revenue Size Flag'].map(conversion)
    
  #Income Size Flag: low, medium, high -> 1,2,3
  conversion = {'low':1, 'medium':2, 'high':3}
  for i in v_sets:
    if 'Income Size Flag' in list(v_sets[i].columns):
      v_sets[i]['Income Size Flag']=v_sets[i]['Income Size Flag'].map(conversion)

  #Similarity Strength: weak, medium, strong -> 1,2,3
  conversion = {'weak':1, 'medium':2, 'strong':3}
  for i in e_sets:
    if 'Similarity Strength' in list(e_sets[i].columns):
      e_sets[i]['Similarity Strength']= e_sets[i]['Similarity Strength'].map(conversion)
      e_sets[i] = e_sets[i].rename(columns={'Similarity Strength':'weight'})

  #Amount Flag: small, medium, large -> 1,50,500 -> treated as weights
  conversion = {'small':1, 'medium':50, 'large':500}
  for i in e_sets:
    if 'Amount Flag' in list(e_sets[i].columns):
      e_sets[i]['Amount Flag']=e_sets[i]['Amount Flag'].map(conversion)
      e_sets[i] = e_sets[i].rename(columns={'Amount Flag':'weight'})

  #? 4: One-hot encoding for categorical features

  # get_dummies for one-hot encoding
  for i in v_sets:
      if 'Person or Organisation' in list(v_sets[i].columns):
          v_sets[i] = pd.get_dummies(v_sets[i], columns=['Person or Organisation'])

  #? 5: String features

  # Attempt 1: remove them
  for i in v_sets:
      if 'Account ID String' in list(v_sets[i].columns):
        v_sets[i] = v_sets[i].drop('Account ID String', axis=1)
      if 'Address' in list(v_sets[i].columns):
        v_sets[i] = v_sets[i].drop('Address', axis=1)
      if 'Name' in list(v_sets[i].columns):
        v_sets[i] = v_sets[i].drop('Name', axis=1)


  #? 6: Additional Features

  # Adding 'Fraudolent' flag:
  for set in v_sets:
    v_sets[set]['Fraudolent'] = np.where(
    np.logical_or(v_sets[set]['CoreCaseGraphID'] != 0.0, v_sets[set]['ExtendedCaseGraphID'] != 0.0), '1', '0')

  t1 = time.time()
  print(f"PREPROCESSING: {(t1-t0):.2f} s")

  return v_sets, e_sets