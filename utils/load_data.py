import pandas as pd

# Dataset remote location (github)
graph_edges = "https://raw.githubusercontent.com/AlbertoParravicini/high-performance-graph-analytics-2020/main/track-ml/data/polimi.case.graphs.edges.csv"
graph_nodes = "https://raw.githubusercontent.com/AlbertoParravicini/high-performance-graph-analytics-2020/main/track-ml/data/polimi.case.graphs.vertices.csv"
training_core_nodes = "https://raw.githubusercontent.com/AlbertoParravicini/high-performance-graph-analytics-2020/main/track-ml/data/training.core.vertices.csv"
training_extended_nodes = "https://raw.githubusercontent.com/AlbertoParravicini/high-performance-graph-analytics-2020/main/track-ml/data/training.extended.vertices.csv"

def load_data():
  nodes = pd.read_csv(graph_nodes, low_memory=False, sep=',', index_col='node_id')
  edges = pd.read_csv(graph_edges, low_memory=False, sep=',', index_col='edge_id')

  core_target = pd.read_csv(training_core_nodes, sep='\t', index_col='NodeID')   
  ext_target = pd.read_csv(training_extended_nodes, sep='\t', index_col='NodeID')
  
  return nodes, edges, core_target, ext_target
