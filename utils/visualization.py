import time
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Transform the embeddings to 2d space for visualization
def get_TSNE(embeddings, components=2):
  t0 = time.time()

  trans = TSNE(n_components=components)
  node_embeddings_2d = trans.fit_transform(embeddings)
  
  t1 = time.time()
  print("dimensionality reduction (TSNE) done in {:.2f} s".format(t1-t0))

  return node_embeddings_2d

def get_PCA(embeddings, components=2):
  t0 = time.time()

  trans = PCA(n_components=components)
  node_embeddings_2d = trans.fit_transform(embeddings)
  
  t1 = time.time()
  print("dimensionality reduction (PCA) done in {:.2f} s".format(t1-t0))

  return node_embeddings_2d
