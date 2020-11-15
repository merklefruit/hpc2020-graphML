import time
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression

import stellargraph as sg
from stellargraph.mapper import CorruptedGenerator, HinSAGENodeGenerator
from stellargraph.layer import DeepGraphInfomax, HinSAGE, Dense

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Model, optimizers, losses, metrics

def deepGraphInfomax(v_sets, e_sets, core_targets, ext_targets, v_sample, e_sample):
  print("DeepGraphInfomax Starting")

  t0 = time.time()

  verbose = 1

  # Initialize stellargraph object
  G = sg.StellarDiGraph(v_sets, e_sets)

  '''
  HinSAGENodeGenerator(G, batch_size, num_samples, head_node_type=None, schema=None, seed=None, name=None)

  G = graph (stellargraph object)
  batch_size = size of batch to return
  num_samples = the number of samples per layer (hop) to take
  head_node_type = the node type that will be given to the generator using the flow method. 
                  The model will expect this type.
                  If not given, it defaults to a single node type.
                  Note: HinSAGE does aggregation on multiple node types 
                  but then predicts on one type.
  '''

  def create_embeddings(
    node_type, num_samples, hinsage_layer_sizes, 
    epochs, patience, batch_size, dropout, activations):

    # Check if num_samples and layer_size are compatible
    assert len(hinsage_layer_sizes) == len(num_samples)

    generator = HinSAGENodeGenerator(
      G, 
      batch_size, 
      num_samples=num_samples,
      head_node_type=node_type
    )

    # HinSAGE layers
    hinsage = HinSAGE(
      layer_sizes=hinsage_layer_sizes,
      activations=activations,
      generator=generator, 
      bias=True,
      normalize="l2",
      dropout=dropout
    )

    def run_deep_graph_infomax(base_model, generator, epochs, node_type):
      corrupted_generator = CorruptedGenerator(generator)
      gen = corrupted_generator.flow(G.nodes(node_type=node_type))
      infomax = DeepGraphInfomax(base_model, corrupted_generator)

      x_in, x_out = infomax.in_out_tensors()

      print("Starting Training")
      ttrain = time.time()
      # Train
      model = Model(inputs=x_in, outputs=x_out)
      model.compile(loss=tf.nn.sigmoid_cross_entropy_with_logits, optimizer=Adam(lr=1e-3))
      es = EarlyStopping(monitor="loss", min_delta=0, patience=patience)
      
      history = model.fit(gen, epochs=epochs, verbose=verbose, callbacks=[es])
      # sg.utils.plot_history(history)
      
      ttrain1 = time.time()
      print(f"Training complete in {(ttrain1-ttrain):.2f} s ({(ttrain1-ttrain)/60:.2f} min)")

      x_emb_in, x_emb_out = base_model.in_out_tensors()
      # for full batch models, squeeze out the batch dim (which is 1)
      if generator.num_batch_dims() == 2:
          x_emb_out = tf.squeeze(x_emb_out, axis=0)

      return x_emb_in, x_emb_out

    # Run Deep Graph Infomax
    x_emb_in, x_emb_out = run_deep_graph_infomax(
      hinsage, generator, epochs=epochs, node_type=node_type)
    
    emb_model = Model(inputs=x_emb_in, outputs=x_emb_out)
    all_embeddings = emb_model.predict(
      generator.flow(G.nodes(node_type=node_type))
    )

    # TSNE visualization of embeddings
    ttsne = time.time()
    print("Creating TSNE")
    embeddings_2d = pd.DataFrame(TSNE(n_components=2).fit_transform(all_embeddings), index=G.nodes(node_type=node_type))

    # draw the points
    node_ids = G.nodes(node_type=node_type).tolist()
    ext_targets = v_sample.loc[[int(node_id) for node_id in node_ids]].ExtendedCaseGraphID 

    label_map = {l: i*10 for i, l in enumerate(np.unique(ext_targets), start=10) if pd.notna(l)}
    node_colours = [label_map[target] if pd.notna(target) else 0 for target in ext_targets]
    
    ttsne1 = time.time()
    print(f"TSNE completed in {(ttsne1-ttsne):.2f} s ({(ttsne1-ttsne)/60:.2f} min)")

    alpha = 0.7
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.scatter(
        embeddings_2d[0],
        embeddings_2d[1],
        c=node_colours,
        cmap="jet",
        alpha=alpha,
    )
    ax.set(aspect="equal")
    plt.title(f'TSNE visualization of HinSAGE "{node_type}" embeddings with Deep Graph Infomax')
    plt.savefig(f"./embeddings/HinSAGE_DGI_embeddings_{node_type}.pdf")

    return all_embeddings, embeddings_2d
  
  # Repeat DGI HinSAGE algorithm for every node type
  # (each node type requires a training phase)

  account_embeddings, account_2d = create_embeddings(
   node_type = "Account",
   epochs = 75,
   patience = 25,
   batch_size = 250,
   dropout = 0.4,
   num_samples = [8, 4], 
   hinsage_layer_sizes = [32, 32],
   activations = ['relu', 'softmax']
  )

  customer_embeddings, customer_2d = create_embeddings(
    node_type = "Customer",
    epochs = 100,
    patience = 50,
    batch_size = 400,
    dropout = 0.4,
    num_samples = [12],
    hinsage_layer_sizes = [72],
    activations = ['relu']
  )

  derEntity_embeddings, derEntity_2d = create_embeddings(
    node_type = "Derived Entity",
    epochs = 100,
    patience = 50,
    batch_size = 1200,
    dropout = 0.25,
    num_samples = [12],
    hinsage_layer_sizes = [72],
    activations = ['relu']
  )

  # Address and External Entity don't have any outgoing nodes and can't be used for this.
  # Another technique specific for External Entities and Addresses might be a good fit.

  # Put all the embeddings in the same map
  # TODO

  # arrays
  full_graph_embeddings = [
    account_embeddings,
    customer_embeddings,
    derEntity_embeddings
  ]

  # dataframes
  full_graph_2d_frames = [
    account_2d,
    customer_2d,
    derEntity_2d
  ]
  full_graph_2d = pd.concat(full_graph_2d_frames)

  # draw all the embeddings together
  node_ids_full = np.concatenate((
      G.nodes(node_type='Account'), 
      G.nodes(node_type='Customer'), 
      G.nodes(node_type='Derived Entity')
  )).tolist()

  ext_targets_full = v_sample.loc[[int(node_id) for node_id in node_ids_full]].ExtendedCaseGraphID 

  label_map_full = {l: i*10 for i, l in enumerate(np.unique(ext_targets_full), start=10) if pd.notna(l)}
  node_colours_full = [label_map_full[target] if pd.notna(target) else 0 for target in ext_targets_full]
  
  alpha = 0.7
  fig, ax = plt.subplots(figsize=(15, 15))
  ax.scatter(
      full_graph_2d[0],
      full_graph_2d[1],
      c=node_colours_full,
      cmap="jet",
      alpha=alpha,
  )
  ax.set(aspect="equal")
  plt.title(f'TSNE visualization of HinSAGE Full Graph embeddings with Deep Graph Infomax')
  plt.savefig("./embeddings/HinSAGE_DGI_embeddings_FullGraph.pdf")

  # Train a classifier for prediction
  # TODO



  t1 = time.time()
  print(f"HinSAGE DGI completed in {(t1-t0):.2f} s ({(t1-t0)/60:.2f} min)")

  return 1