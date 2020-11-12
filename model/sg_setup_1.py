import time
import stellargraph as sg
from gensim.models import Word2Vec


def sg_DeepWalk(v_sets, e_sets, v_sample, e_sample):
  G = sg.StellarDiGraph(v_sets, e_sets)

  #### Graph embedding with NODE2VEC and WORD2VEC

  print("Running DeepWalk")

  rw = sg.data.BiasedRandomWalk(G)
  t0 = time.time()
  walks = rw.run(
      nodes=list(G.nodes()),  # root nodes
      length=10,  # maximum length of a random walk
      n=10,  # number of random walks per root node
      p=0.6,  # Defines (unormalised) probability, 1/p, of returning to source node
      q=1.7,  # Defines (unormalised) probability, 1/q, for moving away from source node
  )
  t1 = time.time()
  print("Number of random walks: {} in {:.2f} s".format(len(walks), (t1-t0)))


  str_walks = [[str(n) for n in walk] for walk in walks]
  model = Word2Vec(str_walks, size=128, window=5, min_count=0, sg=1, workers=8, iter=5)
  # size: length of embedding vector

  # The embedding vectors can be retrieved from model.wv using the node ID.
  # model.wv["19231"].shape 
  
  # Retrieve node embeddings 
  node_ids = model.wv.index2word  # list of node IDs
  node_embeddings = (model.wv.vectors)  # numpy.ndarray of size number of nodes times embeddings dimensionality

  # Retrieve corresponding targets

  # from training csv
  # core_targets = core_target_sample.loc[[int(node_id) for node_id in node_ids if int(node_id) in list(core_target_sample.index)]].CaseID
  # ext_targets = ext_target_sample.loc[[int(node_id) for node_id in node_ids if int(node_id) in list(ext_target_sample.index)]].CaseID

  # from vertices' data
  core_targets = v_sample.loc[[int(node_id) for node_id in node_ids]].CoreCaseGraphID
  ext_targets = v_sample.loc[[int(node_id) for node_id in node_ids]].ExtendedCaseGraphID

  t2 = time.time()
  print(f"Deepwalk complete: {(t2-t0):.2f} s")

  return node_ids, node_embeddings, core_targets, ext_targets
