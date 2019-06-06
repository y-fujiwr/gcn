from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import pickle
import os
import pandas as pd
import shutil
import h5py

from utils import *
from models import GCN, MLP

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)
INPUT_DIM = 84
#C:201
#java:84

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
flags.DEFINE_integer('class_num', 20, 'Number of dimension of output.')
flags.DEFINE_string('mode', 'test', 'Mode of predict (val or test).')
flags.DEFINE_string('input', None, 'Test dataset string')
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
if FLAGS.input == None:
    FLAGS.input = FLAGS.dataset
# Load data
if FLAGS.mode == 'val':
    adj, features, testdata, labels, positions = load_test_data("data/{}/train".format(FLAGS.dataset), FLAGS.class_num)
else:
    adj, features, testdata, labels, positions = load_test_data("data/{}/test".format(FLAGS.input), FLAGS.class_num)
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
model = model_func(placeholders, input_dim=INPUT_DIM, output_dim=FLAGS.class_num, logging=True)

feed_dict = construct_test_feed_dict(features, support, placeholders)
predicts = model.predict(model_load_dir=os.path.join("model", FLAGS.model_name), feed_dict=feed_dict)
df = []
log_name = os.path.join(*["log",FLAGS.dataset,FLAGS.model_name+".txt"])
if FLAGS.mode == "test":
    log_name = os.path.join(*["log",FLAGS.dataset,FLAGS.model_name+",test.txt"])
with open(log_name,"w") as w:
    for pair, filename, G in positions:
        predict = predicts[pair[0]:pair[1]].argmax(axis=1)
        result = pd.Series(predict).value_counts(normalize=True)
        result_label = result.index.values[0]
        answer = labels[pair[0]:pair[1]].argmax(axis=1)[0]
    
        w.write(filename + "\n")
        w.write("Answer:{}\nResult:".format(answer))
        w.write(str(result)+"\n")
        w.write(str(answer == result_label)+"\n")
        
        df.append([filename,answer,result_label])

        x = (predict == labels[pair[0]:pair[1]].argmax(axis=1))
        for i in range(len(predict)):
            if x[i] == False:
                G.node(str(i),str(predict[i]),color="red")
            else:
                G.node(str(i),str(predict[i]),color="blue")
        #G.render("tree/" + filename)
    result_table = pd.DataFrame(df,columns=["filename","label","predict"])
    fp = result_table[result_table["label"] != result_table["predict"]]
    w.write("\nRecall:{}".format( ( len(result_table) - len(fp) ) / len(result_table) ) )
with open("log/{}/recall.csv".format(FLAGS.dataset),"a") as w:
    w.write("{}\n".format( ( len(result_table) - len(fp) ) / len(result_table) ) )
print(fp["label"].value_counts().index.values)
print(pd.Series(np.sum(labels,axis=0)))

if FLAGS.mode == "test":
    exit()

target = fp["label"].value_counts().index.values
i = 1
while True:
    additional_dataset_dir = "data/{}/train/copy{}".format(FLAGS.dataset,i)
    if os.path.exists(additional_dataset_dir):
        i+=1
    else:
        os.makedirs(additional_dataset_dir,True)
        os.chmod(additional_dataset_dir,0o777)
        for t in target:
            shutil.copytree("data/{}/train/{}".format(FLAGS.dataset,t),additional_dataset_dir+"/{}".format(t))
        break
        