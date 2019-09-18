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
topK = 5

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
flags.DEFINE_string('mode', 'test', 'Mode of predict (val or test).')
flags.DEFINE_string('input', None, 'Test dataset string')
flags.DEFINE_integer('input_dim',201,'Dimension of input vectors. Java:85, C:201')
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
    adj, features, testdata, labels, positions = load_test_data("data/{}/train".format(FLAGS.dataset), FLAGS.class_num, FLAGS.input_dim)
else:
    adj, features, testdata, labels, positions = load_test_data("data/{}/test".format(FLAGS.input), FLAGS.class_num, FLAGS.input_dim)
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
model = model_func(placeholders, input_dim=FLAGS.input_dim, output_dim=FLAGS.class_num, logging=True)

feed_dict = construct_test_feed_dict(features, support, placeholders)
predicts = model.predict(model_load_dir=os.path.join("model", FLAGS.model_name), feed_dict=feed_dict)
df_top1 = []
df_top3 = []
df_top5 = []
df_top10 = []
df = []
recalls = []
log_name = os.path.join(*["log",FLAGS.dataset,FLAGS.model_name+".txt"])
if FLAGS.mode == "test":
    log_name = os.path.join(*["log",FLAGS.dataset,FLAGS.model_name+",test.txt"])
os.makedirs("log/{}".format(FLAGS.dataset),exist_ok=True)
os.chmod("log/{}".format(FLAGS.dataset),0o777)
with open(log_name,"w") as w:
    recall_log = open("log/{}/recall_{}.csv".format(FLAGS.dataset,FLAGS.mode),"a")
    for pair, filename, G in positions:
        predict = predicts[pair[0]:pair[1]].argmax(axis=1)
        sum_all_predict = np.sum(predicts[pair[0]:pair[1]], axis=0)
        result = pd.Series(predict).value_counts(normalize=True)
        result_label = result.index.values[0]
        result_top1 = np.argsort(sum_all_predict)[::-1][:1]
        result_top3 = np.argsort(sum_all_predict)[::-1][:3]
        result_top5 = np.argsort(sum_all_predict)[::-1][:5]
        result_top10 = np.argsort(sum_all_predict)[::-1][:10]

        answer = labels[pair[0]:pair[1]].argmax(axis=1)[0]

        w.write(filename + "\n")
        w.write("Answer:{}\nResult:".format(answer))
        def normalize(x):
            return x/len(predict)
        w.write(str(pd.Series(sum_all_predict).sort_values(ascending=False).map(normalize))+"\n")
        w.write(str(answer == result_label)+"\n")
        
        df_top1.append([filename,answer,result_top1])
        df_top3.append([filename,answer,result_top3])
        df_top5.append([filename,answer,result_top5])
        df_top10.append([filename,answer,result_top10])

        x = (predict == labels[pair[0]:pair[1]].argmax(axis=1))
        for i in range(len(predict)):
            if x[i] == False:
                G.node(str(i),str(predict[i]),color="red")
            else:
                G.node(str(i),str(predict[i]),color="blue")
        #G.render("tree/" + filename)
    result_table1 = pd.DataFrame(df_top1,columns=["filename","label","predict"])
    result_table3 = pd.DataFrame(df_top3,columns=["filename","label","predict"])
    result_table5 = pd.DataFrame(df_top5,columns=["filename","label","predict"])
    result_table10 = pd.DataFrame(df_top10,columns=["filename","label","predict"])
    df.append(result_table1)
    df.append(result_table3)
    df.append(result_table5)
    df.append(result_table10)
    
    for result_table in df:
        fp = pd.DataFrame(columns=["filename","label","predict"])
        for _, item in result_table.iterrows():
            if item["label"] not in item["predict"]:
                fp = fp.append(item)
        recall = ( len(result_table) - len(fp) ) / len(result_table)
        recalls.append(recall)
        w.write("\nRecall:{}".format(recall) )
        recall_log.write("{},".format(recall) )
    recall_log.write("\n")

#print(pd.Series(np.sum(labels,axis=0)))
print(list(fp["label"].value_counts().index.values))
if FLAGS.mode == "test":
    exit()
if recalls[0] <=0.2:
    exit()
target = []
if os.path.exists('data/{}/addition.pkl'.format(FLAGS.dataset)):
    with open('data/{}/addition.pkl'.format(FLAGS.dataset),'rb') as f:
        target = pickle.load(f)
        target.append(list(fp["label"].value_counts().index.values))
else:
    target.append(list(fp["label"].value_counts().index.values))
with open('data/{}/addition.pkl'.format(FLAGS.dataset),'wb') as f:
    pickle.dump(target, f)
"""
i = 1
while True:
    additional_dataset_dir = "data/{}/train/copy{}".format(FLAGS.dataset,i)
    if os.path.exists(additional_dataset_dir):
        i+=1
    else:
        break
os.makedirs(additional_dataset_dir,True)
os.chmod(additional_dataset_dir,0o777)
for t in target:
    shutil.copytree("data/{}/train/{}".format(FLAGS.dataset,t),additional_dataset_dir+"/{}".format(t))
"""
        