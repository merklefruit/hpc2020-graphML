import pandas as pd
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
pd.set_option('mode.chained_assignment', None)

# preprocessing of heterogeneous nodes and edges
def preprocess_data(v_sample, e_sample, core_targets, ext_targets, core_testing):
  t0 = time.time()
  print("PREPROCESSING DATA STARTED")

  v_data = v_sample

  #? 0: Replace CoreCaseID, ExtCaseID and testingFlag with CSV data (notebook: "Correct CVS data")
  v_data_new = v_data.drop(['CoreCaseGraphID', 'ExtendedCaseGraphID', 'testingFlag'], axis=1)

  core_targets_new = core_targets.rename(columns={'CaseID': 'CoreCaseGraphID'})
  core_targets_new.index.name = 'node_id'

  ext_targets_new = ext_targets.rename(columns={'CaseID': 'ExtendedCaseGraphID'})
  ext_targets_new.index.name = 'node_id'

  core_testing_new = core_testing.rename(columns={'CaseID': 'CoreCaseGraphID'})
  core_testing_new.index.name = 'node_id'

  v_data_new = pd.merge(v_data_new, core_targets_new, left_index=True, right_index=True, how='left')
  v_data_new = pd.merge(v_data_new, ext_targets_new, left_index=True, right_index=True, how='left')

  # For some reason some nodes have 2 different core case IDs... filtering to just take the first
  for ind, row in v_data.iterrows():
      v_data_new['CoreCaseGraphID'][ind] = core_testing.loc[core_testing.index == ind].CaseID.values[0] if len(core_testing.loc[core_testing.index == ind].CaseID.values) != 0 else row['CoreCaseGraphID']

  tsf = pd.DataFrame(v_data_new.index)
  tsf = tsf.set_index('node_id')
  tsf['testingFlag'] = np.NaN
  for ind, row in tsf.iterrows():
      tsf['testingFlag'][ind] = 0 if len(core_targets.loc[core_targets.index == ind]) != 0 or len(ext_targets.loc[ext_targets.index == ind]) != 0 else row.testingFlag
  for ind, row in tsf.iterrows():
      tsf['testingFlag'][ind] = 1 if len(core_testing.loc[core_testing.index == ind]) != 0 else row.testingFlag
  v_data_new = pd.merge(v_data_new, tsf, left_index=True, right_index=True, how='left')

  v_data_new = v_data_new[~v_data_new.index.duplicated(keep='first')]

  v_sample = v_data_new


  #? 0.1: Add Extra Features: Node Degree (notebook: Node Degree feature)
  source_data = e_sample.groupby('from_id').count().to_id
  source_data = pd.DataFrame(source_data)
  source_data = source_data.rename(columns={'to_id': 'source_degree'})
  source_data = source_data.rename_axis('node_id')

  target_data = e_sample.groupby('to_id').count().from_id
  target_data = pd.DataFrame(target_data)
  target_data = target_data.rename(columns={'from_id': 'target_degree'})
  target_data = target_data.rename_axis('node_id')

  v_sample = pd.merge(v_sample, source_data, left_index=True, right_index=True, how='left')
  v_sample = pd.merge(v_sample, target_data, left_index=True, right_index=True, how='left')

  v_sample['source_degree'] = v_sample['source_degree'].fillna(0)
  v_sample['target_degree'] = v_sample['target_degree'].fillna(0)


  #? 1: missing core/ext case ID
  # Solution: set to 0 if NaN

  v_sample.CoreCaseGraphID = v_sample.CoreCaseGraphID.fillna(0)
  v_sample.ExtendedCaseGraphID = v_sample.ExtendedCaseGraphID.fillna(0)


  #? 2: Create dataframes for each node type (Account, Customer, Derived entity, External entity, Address)
  # and each edge type (has account, has address, is similar, money transfer)

  v_sets = defaultdict()
  for v_type in list(pd.Categorical(v_sample.Label).categories):
      v_sets[v_type] = v_sample[v_sample.Label == v_type]
      v_sets[v_type] = v_sets[v_type].drop(['Label']+list(v_sets[v_type].columns[v_sets[v_type].isnull().all()]), axis=1)
      v_sets[v_type].testingFlag = v_sets[v_type].testingFlag.fillna(-1)

  e_sets = defaultdict()
  for e_type in list(pd.Categorical(e_sample.Label).categories):
      e_sets[e_type] = e_sample[e_sample.Label == e_type]
      e_sets[e_type] = e_sets[e_type].drop(['Label']+list(e_sets[e_type].columns[e_sets[e_type].isnull().all()]), axis=1)
      e_sets[e_type] = e_sets[e_type].rename(columns={'from_id':'source', 'to_id':'target'})


  #? 3: Logical conversion of categorical features

  # Revenue Size Flag: low, mid_low, medium, mid_high, high -> 0,1
  conversion = {'low':0.1, 'mid_low':0.3, 'medium':0.6, 'mid_high':0.8, 'high':1}
  for i in v_sets:
    if 'Revenue Size Flag' in list(v_sets[i].columns):
      v_sets[i]['Revenue Size Flag']=v_sets[i]['Revenue Size Flag'].map(conversion)
    
  # Income Size Flag: low, medium, high -> 0,1
  conversion = {'low':0.1, 'medium':0.5, 'high':1}
  for i in v_sets:
    if 'Income Size Flag' in list(v_sets[i].columns):
      v_sets[i]['Income Size Flag']=v_sets[i]['Income Size Flag'].map(conversion)

  # Similarity Strength: weak, medium, strong -> 0,1
  conversion = {'weak':0.1, 'medium':0.5, 'strong':1}
  for i in e_sets:
    if 'Similarity Strength' in list(e_sets[i].columns):
      e_sets[i]['Similarity Strength']= e_sets[i]['Similarity Strength'].map(conversion)
      e_sets[i] = e_sets[i].rename(columns={'Similarity Strength':'weight'})

  # Amount Flag: small, medium, large -> 0,1 -> treated as weights
  conversion = {'small':0.1, 'medium':0.5, 'large':1}
  for i in e_sets:
    if 'Amount Flag' in list(e_sets[i].columns):
      e_sets[i]['Amount Flag']=e_sets[i]['Amount Flag'].map(conversion)
      e_sets[i] = e_sets[i].rename(columns={'Amount Flag':'weight'})


  #? 4: One-hot encoding

  # One-hot encoding of Person or Organisation
  for i in v_sets:
      if 'Person or Organisation' in list(v_sets[i].columns):
          v_sets[i] = pd.get_dummies(v_sets[i], columns=['Person or Organisation'])

  # one-hot encoding of CoreCaseGraphID
  for i in v_sets:
      if 'CoreCaseGraphID' in list(v_sets[i].columns):
          v_sets[i] = pd.get_dummies(v_sets[i], columns=['CoreCaseGraphID'])


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

  '''
  Note: isReadable flag: 
    it basically tells me if the name has been protected by encryption and is thus unreadable.
    I finally decided not to use this feature.
    (view "notebooks/isReadable field" for more details)
  '''

  # # Adding 'Fraudolent' flag:
  # for set in v_sets:
  #   v_sets[set]['Fraudolent'] = np.where(
  #   np.logical_or(v_sets[set]['CoreCaseGraphID'] != 0.0, v_sets[set]['ExtendedCaseGraphID'] != 0.0), '1', '0')

  t1 = time.time()
  print(f"PREPROCESSING DATA COMPLETED: {(t1-t0):.2f} s")

  return v_sets, e_sets