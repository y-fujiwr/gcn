from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import pickle
import os
import pandas as pd

from utils import *
from models import GCN, MLP

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)
INPUT_DIM = 201

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'small', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 1000, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 256, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.2, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_integer('layers', 4, 'Number of layers to train.')
flags.DEFINE_string('model_name', 'default', "Model name string.")
flags.DEFINE_integer('outputdim', 20, 'Number of dimension of output.')
FLAGS.model_name = "{},{},{},{},{},{},{},{}".format(
    FLAGS.model_name,
    FLAGS.dataset,
    FLAGS.model,
    FLAGS.learning_rate,
    FLAGS.hidden1,
    FLAGS.dropout,
    FLAGS.weight_decay,
    FLAGS.layers
)

# Load data
adj, features, testdata, labels, positions = load_test_data("data/{}/test".format(FLAGS.dataset))
# Some preprocessing
features = preprocess_features(features)
if FLAGS.model == 'gcn':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':
    support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
elif FLAGS.model == 'dense':
    support = [preprocess_adj(adj)]  # Not used
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))



# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, testdata.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}
# Create model
model = model_func(placeholders, input_dim=INPUT_DIM, output_dim=FLAGS.outputdim, logging=True)

feed_dict = construct_test_feed_dict(features, support, placeholders)
predicts = model.predict(model_load_dir=os.path.join("model", FLAGS.model_name), feed_dict=feed_dict)
with open(os.path.join(*["log",FLAGS.model_name+".txt"]),"w") as w:
    for pair, filename, G in positions:
        predict = predicts[pair[0]:pair[1]].argmax(axis=1)
        result = pd.Series(predict).value_counts(normalize=True)
        result_label = result.index.values[0]
        answer = labels[pair[0]:pair[1]].argmax(axis=1)[0]
    
        w.write(filename + "\n")
        w.write("Answer:{}\nResult:".format(answer))
        w.write(str(result)+"\n")
        w.write(str(answer == result_label)+"\n")

        x = (predict == labels[pair[0]:pair[1]].argmax(axis=1))
        print(filename)
        print(predict)
        print(result)
        print()
        for i in range(len(predict)):
            if x[i] == False:
                G.node(str(i),color="red")
        #G.render("tree/" + filename)
    


