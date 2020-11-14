from pathlib import Path
import pandas as pd
import os
import time

# Dataset remote location (github)
graph_edges = "https://raw.githubusercontent.com/AlbertoParravicini/high-performance-graph-analytics-2020/main/track-ml/data/polimi.case.graphs.edges.csv"
graph_nodes = "https://raw.githubusercontent.com/AlbertoParravicini/high-performance-graph-analytics-2020/main/track-ml/data/polimi.case.graphs.vertices.csv"
training_core_nodes = "https://raw.githubusercontent.com/AlbertoParravicini/high-performance-graph-analytics-2020/main/track-ml/data/training.core.vertices.csv"
training_extended_nodes = "https://raw.githubusercontent.com/AlbertoParravicini/high-performance-graph-analytics-2020/main/track-ml/data/training.extended.vertices.csv"

# Dataset local location (if already downloaded)
data_path = Path("./data/raw")
nodes_path = Path("/nodes.csv")
edges_path = Path("/edges.csv")
training_core_path = Path("/training_core_nodes.csv")
training_extended_path = Path("/training_extended_nodes.csv")

# Loads data either from remote or local locations
# based on if you have already downloaded it before.
def load_data():
  
  def load_local_data():
    print("Dataset already downloaded. Loading it from file system")

    t0 = time.time()
    nodes.read_csv(f"{data_path}{nodes_path}")
    edges.read_csv(f"{data_path}{edges_path}")
    core_target.read_csv(f"{data_path}{training_core_path}")
    ext_target.read_csv(f"{data_path}{training_extended_path}")
    t1 = time.time()

    print(f"LOADING DATA: {(t1-t0):.2f} s")

    return nodes, edges, core_target, ext_target
  
  def load_remote_data():
    print("Dataset not found in file system. Downloading them from GitHub...")
    
    t0 = time.time()
    nodes = pd.read_csv(graph_nodes, low_memory=False, sep=',', index_col='node_id')
    edges = pd.read_csv(graph_edges, low_memory=False, sep=',', index_col='edge_id')
    core_target = pd.read_csv(training_core_nodes, sep='\t', index_col='NodeID')   
    ext_target = pd.read_csv(training_extended_nodes, sep='\t', index_col='NodeID')
    t1 = time.time()

    print(f"LOADING DATA: {(t1-t0):.2f} s")

    nodes.to_csv(f"{data_path}{nodes_path}")
    edges.to_csv(f"{data_path}{edges_path}")
    core_target.to_csv(f"{data_path}{training_core_path}")
    ext_target.to_csv(f"{data_path}{training_extended_path}")

    return nodes, edges, core_target, ext_target
  

  if os.path.exists(data_path) & os.path.exists(f"{data_path}{nodes_path}"):
    nodes, edges, core_target, ext_target = load_local_data()
  else:
    try:
      os.makedirs(data_path)
    except OSError:
      try:
        os.rmdir(data_path)
        load_data()
      except OSError:
        print("Error creating data directory")
    nodes, edges, core_target, ext_target = load_remote_data()

  return nodes, edges, core_target, ext_target
