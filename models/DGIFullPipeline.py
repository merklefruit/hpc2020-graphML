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


def DGIPipeline():
  return None