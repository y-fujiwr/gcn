from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import pickle
import os
import sys

from utils import *
from models import GCN, MLP
import pandas as pd

import requests

def line_notify(msg):
	line_token = 'htThV5RCliNCarubopToCi91nOpwuuw8iLbz5extQc9'
	line_api = 'https://notify-api.line.me/api/notify'
	payload = {'message': msg}
	headers = {'Authorization': 'Bearer ' + line_token}
	requests.post(line_api, data = payload, headers = headers)

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'small', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 2000, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 256, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.2, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_integer('layers', 4, 'Number of layers to train.')
flags.DEFINE_string('model_name', 'default', "Model name string.")
flags.DEFINE_integer('class_num', 20, 'Number of dimension of output.')
flags.DEFINE_integer('input_dim', 201, 'Dimension of input vectors. Java:84, C:201')
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
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data("data/{}".format(FLAGS.dataset), FLAGS.class_num, FLAGS.input_dim)
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
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}
# Create model
model = model_func(placeholders, input_dim=features[2][1], logging=True)

# Initialize session
sess = tf.Session()


# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)


# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []
try:
    os.makedirs("model",True)
    os.chmod("model",0o777)
    os.makedirs(os.path.join(*["model", FLAGS.model_name]), True)
    os.chmod(os.path.join(*["model", FLAGS.model_name]), 0o777)
except FileExistsError:
    pass
try:
    os.makedirs("log/{}".format(FLAGS.dataset), True)
    os.chmod("log/{}".format(FLAGS.dataset), 0o777)
except FileExistsError:
    pass
saver = tf.train.Saver()
# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy, model.outputs], feed_dict=feed_dict)
    # Validation
    cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders)
    cost_val.append(cost)

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))
    """
    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        print("Early stopping...")
        break
    """

print("Optimization Finished!")
saver.save(sess, os.path.join(*["model", FLAGS.model_name, FLAGS.model_name]))

# Testing
test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

with open(os.path.join(*["log",FLAGS.dataset,"result.csv"]),"a") as r :
    r.write("{},{},{},{}\n".format(str(sys.argv[1:]).replace(",","_"),test_cost,test_acc,test_duration))

line_notify("{}'s learning finished!\naccuracy:{}".format(FLAGS.model_name,test_acc))
if test_acc >=0.8:
    line_notify(str(pd.Series(np.sum(labels,axis=0))))

