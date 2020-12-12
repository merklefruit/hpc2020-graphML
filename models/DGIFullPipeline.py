import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score

import stellargraph as sg
from stellargraph.mapper import CorruptedGenerator, HinSAGENodeGenerator
from stellargraph.layer import DeepGraphInfomax, HinSAGE

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Model, optimizers, losses, metrics

'''
Runs the entire pipeline:
  - Takes preprocessed data as input
  - Outputs predictions on the test_set nodes.
'''
def DGIPipeline(v_sets, e_sets, v_data, e_data, core_targets, ext_targets, core_testing):
  print("HINSAGE DGI FULL PIPELINE STARTED")
  tin = time.time()

  #? Sort based on testingFlag
  # data_splits[i].iloc[INDEX].values[0]
  # where INDEX:
  # [0] testingFlag=NaN
  # [1] testingFlag=0
  # [2] testingFlag=1
  data_splits = dict()
  for i in v_sets:
      v_sets[i] = v_sets[i].sort_values('testingFlag')
      data_splits[i] = v_sets[i].testingFlag.value_counts().to_frame()
      v_sets[i] = v_sets[i].drop('testingFlag', axis=1)
      
  #? Removing ExtendedCaseGraphID
  for i in v_sets:
      v_sets[i] = v_sets[i].drop('ExtendedCaseGraphID', axis=1)
  
  #? Create the graph object
  G = sg.StellarDiGraph(v_sets, e_sets)


  '''
  Iterate through the algotithm for every node type.
  This is because HinSAGE can predict on one node type at a time, even though
  it uses all the graph to compute the embeddings.
  '''
  # Parameters
  batch_size = 200
  epochs = 10
  num_samples = [8, 4]
  dropout = 0.4
  hinsage_layer_sizes = [32, 32]
  verbose = 1
  visualize = False

  def run_for_node_type(v_type):
      print(f"Starting pipeline for ")
      nan_tflag = data_splits[v_type].iloc[0].values[0]
      train_tflag = data_splits[v_type].iloc[1].values[0]
      test_tflag = data_splits[v_type].iloc[2].values[0]

      train_cv_set = v_sets[v_type][nan_tflag:nan_tflag+train_tflag]
      train_cv_ids = train_cv_set.index.values.tolist()
      train_cv_labels = v_data.loc[[int(node_id) for node_id in train_cv_ids]].ExtendedCaseGraphID

      test_set = v_sets[v_type][-test_tflag:]
      test_ids = test_set.index.values.tolist()

      generator = HinSAGENodeGenerator(
          G, 
          batch_size, 
          num_samples,
          head_node_type=v_type
      )

      hinsage = HinSAGE(
          layer_sizes=hinsage_layer_sizes,
          activations=['relu', 'relu'],
          generator=generator, 
          bias=True,
          normalize="l2",
          dropout=dropout
      )

      def run_deep_graph_infomax(base_model, generator, epochs):
          print(f"Starting training for {v_type} type: ")
          t0 = time.time()
          corrupted_generator = CorruptedGenerator(generator)
          gen = corrupted_generator.flow(G.nodes(node_type=v_type))
          infomax = DeepGraphInfomax(base_model, corrupted_generator)

          x_in, x_out = infomax.in_out_tensors()

          # Train with DGI
          model = Model(inputs=x_in, outputs=x_out)
          model.compile(loss=tf.nn.sigmoid_cross_entropy_with_logits, optimizer=Adam(lr=1e-3))
          es = EarlyStopping(monitor="loss", min_delta=0, patience=10)
          history = model.fit(gen, epochs=epochs, verbose=verbose, callbacks=[es])
          #sg.utils.plot_history(history)

          x_emb_in, x_emb_out = base_model.in_out_tensors()
          if generator.num_batch_dims() == 2:
              x_emb_out = tf.squeeze(x_emb_out, axis=0)

          t1 = time.time()
          print(f'Time required: {t1-t0:.2f} s ({(t1-t0)/60:.1f} min)')
          
          return x_emb_in, x_emb_out, model

      #? Train HinSAGE model:
      x_emb_in, x_emb_out, _model = run_deep_graph_infomax(hinsage, generator, epochs=epochs)

      emb_model = Model(inputs=x_emb_in, outputs=x_emb_out)

      train_cv_embs = emb_model.predict(
          generator.flow(train_cv_set.index.values)
      )

      #? Optional: Plot embeddings of training and CV set of current node type
      if (visualize == True):
          train_cv_embs_2d = pd.DataFrame(
              TSNE(n_components=2).fit_transform(train_cv_embs), 
              index=train_cv_set.index.values
          )
          label_map = {l: i*10 for i, l in enumerate(np.unique(train_cv_labels), start=10) if pd.notna(l)}
          node_colours = [label_map[target] if pd.notna(target) else 0 for target in train_cv_labels]
          
          alpha = 0.7
          fig, ax = plt.subplots(figsize=(15, 15))
          ax.scatter(
              train_cv_embs_2d[0],
              train_cv_embs_2d[1],
              c=node_colours,
              cmap="jet",
              alpha=alpha,
          )
          ax.set(aspect="equal")
          plt.title(f"TSNE of HinSAGE {v_type} embeddings with DGI- coloring on ExtendedCaseGraphID")
          plt.show()

          return 1

      #? Split training and cross valuation set using 80% 20% simple ordered split
      n_embs = train_cv_embs.shape[0]
      train_size = int(n_embs*0.80)
      cv_size = int(n_embs*0.20)

      train_set = train_cv_embs[:train_size]
      train_labels = np.ravel(pd.DataFrame(train_cv_labels.values[:train_size]).fillna(0))

      cv_set = train_cv_embs[-cv_size:]
      cv_labels = np.ravel(pd.DataFrame(train_cv_labels.values[-cv_size:]).fillna(0))

      #? CLASSIFY
      print(f"Running Classifier for {v_type} type")
      classifier = DecisionTreeClassifier()
      classifier.fit(
          X=train_set,
          y=train_labels,
      )
      cv_pred = classifier.predict(cv_set)
      f1_avg = f1_score(cv_labels, cv_pred, average='weighted')
      acc = (cv_pred == cv_labels).mean()
      print(f"{v_type} CV Metrics: f1: {f1_avg:.6f} - acc: {acc:.6f}")

      #? Now Run on test set
      test_embs = emb_model.predict(
          generator.flow(test_set.index.values)
      )
      test_pred = classifier.predict(test_embs)
      
      output = pd.DataFrame(test_ids)
      output = output.rename(columns={0: 'node_id'})
      output['ExtendedCaseGraphID'] = test_pred
      output = output.set_index('node_id')
      output.to_csv(f"./output/{v_type}_predictions.csv")

      return output
  
  #? Run for each node type
  full_predictions = pd.DataFrame()
  for v_type in v_sets:
      predictions = run_for_node_type(v_type)
      full_predictions.append(predictions)

  tout = time.time()
  print(f"HINSAGE DGI FULL PIPELINE COMPLETED: {(tin-tout)/60:.0f} min")
  return 1