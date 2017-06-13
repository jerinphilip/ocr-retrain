

import tensorflow as tf
from tensorflow.python.ops import ctc_ops as ctc
import numpy as np

param_learning = {
        "rate": 0.001,
        "momentum": 0.9,
        "epochs": 120,
        "batch_size": 128
}


graph = tf.Graph()
with graph.as_default():
    x = tf.placeholder(tf.float32, [None, None, 32]) 
