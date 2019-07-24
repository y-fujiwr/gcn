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
from hyperopt import hp, tpe, Trials, fmin

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
flags.DEFINE_string('dataset', 'balance_node20', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 50, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 256, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.2, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_integer('layers', 4, 'Number of layers to train.')
flags.DEFINE_string('model_name', 'default', "Model name string.")
flags.DEFINE_integer('class_num', 20, 'Number of dimension of output.')
flags.DEFINE_integer('input_dim', 201, 'Dimension of input vectors. Java:84, C:201')

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

def train_model():
    # Create model
    model = model_func(placeholders, input_dim=features[2][1], logging=True)
    model_name = "{},{},{},{},{},{},{},{},{}".format(
        FLAGS.model_name,
        FLAGS.dataset,
        FLAGS.model,
        FLAGS.learning_rate,
        FLAGS.hidden1,
        FLAGS.dropout,
        FLAGS.weight_decay,
        FLAGS.layers,
        FLAGS.epochs
    )

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
        os.makedirs(os.path.join(*["model", model_name]), True)
        os.chmod(os.path.join(*["model", model_name]), 0o777)
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
        """
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
              "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
             "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))
        """

    #print("Optimization Finished!")
    #saver.save(sess, os.path.join(*["model", model_name, model_name]))

    # Testing
    """
    test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders)
    print("Test set results:", "cost=", "{:.5f}".format(test_cost),
        "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

    with open(os.path.join(*["log",FLAGS.dataset,"result.csv"]),"a") as r :
        r.write("{},{},{},{}\n".format(str(sys.argv[1:]).replace(",","_"),test_cost,test_acc,test_duration))
    """

    line_notify("{}'s learning finished!\naccuracy:{}".format(model_name,acc))

    return acc

def objective(args):
    FLAGS.learning_rate = args["learning_rate"]
    FLAGS.hidden1 = int(args["hidden1"])
    FLAGS.dropout = args["dropout"]
    FLAGS.weight_decay = args["weight_decay"]
    FLAGS.layers = int(args["num_layers"])
    FLAGS.epochs = int(args["epochs"])
    accuracy = train_model()
    return -1 * accuracy

hyperopt_parameters = {
    "learning_rate" : hp.loguniform("learning_rate", -5, -1),
    "hidden1" : hp.quniform("hidden1", 10, 500, q=1),
    "dropout" : hp.uniform("dropout", 0, 1),
    "weight_decay" : hp.loguniform("weight_decay", -7, -1),
    "num_layers" : hp.quniform("layers", 1, 8, q=1),
    "epochs" : hp.quniform("epochs", 100, 4000, q=100)
}

max_evals = 400
trials = Trials()

best = fmin(objective,hyperopt_parameters,algo=tpe.suggest,max_evals=max_evals,trials=trials,verbose=1)
print(str(best))
line_notify("{}".format(str(best)))
