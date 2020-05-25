from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import pickle
import os
import pandas as pd
import shutil
import h5py
from itertools import product
from pathlib import Path
import textdistance
from distance import levenshtein

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
flags.DEFINE_integer('epochs', 2500, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.2, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_integer('layers', 4, 'Number of layers to train.')
flags.DEFINE_string('model_name', 'default', "Model name string.")
flags.DEFINE_integer('class_num', 20, 'Number of dimension of output.')
flags.DEFINE_string('mode', 'test', 'Mode of predict (val or test).')
flags.DEFINE_string('input', None, 'Test dataset string')
flags.DEFINE_string('language', "java", 'Programming Language. java, c')
flags.DEFINE_string('learning_type','raw','Select learning mode (reinforcement, node, method).')
flags.DEFINE_boolean('normalize', False, 'Select whether normalizing identifier or not.')
flags.DEFINE_integer('lsi', None, 'Set dimension of LSI (None means BoW)')
flags.DEFINE_string('model_file_name', None, 'Test dataset string')

if FLAGS.model_file_name == None:
    FLAGS.model_name = "{},{},{},{},{},{},{},{},{}".format(
        FLAGS.model_name,
        FLAGS.dataset,
        FLAGS.model,
        FLAGS.learning_rate,
        FLAGS.hidden1,
        FLAGS.dropout,
        FLAGS.weight_decay,
        FLAGS.layers,
        FLAGS.learning_type
    )
else:
    FLAGS.model_name = FLAGS.model_file_name

if FLAGS.input == None:
    FLAGS.input = FLAGS.dataset

# Calculate input vector dimension
vector_dimension = 0
if FLAGS.language == "java":
    vector_dimension = 85
elif FLAGS.language == "c":
    vector_dimension = 201
dictionary_filepath = f"data/{FLAGS.dataset}/dictionary.txt"
if os.path.exists(dictionary_filepath) and FLAGS.normalize == False:
    vector_dimension += len(open(dictionary_filepath).readlines()) + 1

# Load data
if FLAGS.mode == 'val':
    adj, features, testdata, labels, positions = load_test_data("data/{}/train".format(FLAGS.dataset), FLAGS.class_num, vector_dimension)
else:
    adj, features, testdata, labels, positions = load_test_data("data/{}/{}".format(FLAGS.input,FLAGS.mode), FLAGS.class_num, vector_dimension)
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

print("The number of node")
print(labels.sum(axis=0))

# Create model
model = model_func(placeholders, input_dim=features[2][1], output_dim=FLAGS.class_num, logging=True)

feed_dict = construct_test_feed_dict(features, support, placeholders)
predicts = model.predict(model_load_dir=os.path.join("model", FLAGS.model_name), feed_dict=feed_dict)
df_top1 = []
df_top3 = []
df_top5 = []
df_top10 = []
df_all=[]
df = []
recalls = []
log_name = os.path.join(*["log",FLAGS.dataset,f"{FLAGS.model_name}_{FLAGS.mode}.txt"])
os.makedirs("log/{}".format(FLAGS.dataset),exist_ok=True)
os.chmod("log/{}".format(FLAGS.dataset),0o777)
with open(log_name,"w") as w:
    recall_log = open("log/{}/recall_{}_{}.csv".format(FLAGS.dataset,FLAGS.mode,FLAGS.learning_type),"a")
    recall_log.write(f"{FLAGS.model_name},")
    for pair, filename,line_num, G in positions:
        predict = predicts[pair[0]:pair[1]].argmax(axis=1)
        sum_all_predict = np.sum(predicts[pair[0]:pair[1]], axis=0)
        result = pd.Series(predict).value_counts(normalize=True)
        result_label = result.index.values[0]

        # can calculate levenshtein similarity score
        input_ast_string = open(filename,"r").readlines()[line_num].split(" ")
        output_ast_string = open(list(Path(f"data/{FLAGS.dataset}/test/{result_label}").glob("**/*.txt"))[0]).readlines()[0].split(" ")
        sim_levenshtein = 1# - textdistance.hamming.distance(output_ast_string,input_ast_string)/max(len(input_ast_string),len(output_ast_string))
        
        #ranking setup
        result_top1 = np.argsort(sum_all_predict)[::-1][:1]
        result_top3 = np.argsort(sum_all_predict)[::-1][:3]
        result_top5 = np.argsort(sum_all_predict)[::-1][:5]
        result_top10 = np.argsort(sum_all_predict)[::-1][:10]
        answer = labels[pair[0]:pair[1]].argmax(axis=1)[0]

        w.write(filename + "\n")
        w.write(f"Answer:{answer}\nResult:")
        def normalize(x):
            return x/len(predict)
        rank = result#pd.Series(sum_all_predict).sort_values(ascending=False).map(normalize)
        w.write(str(rank)+"\n")
        w.write(str(answer == result_label)+"\n")
        df_top1.append([filename,line_num,answer,result.index.values[:1],rank.iloc[0],sim_levenshtein])
        df_top3.append([filename,line_num,answer,result.index.values[:3],rank,sim_levenshtein])
        df_top5.append([filename,line_num,answer,result.index.values[:5],rank,sim_levenshtein])
        df_top10.append([filename,line_num,answer,result.index.values[:10],rank,sim_levenshtein])
        df_all.append([filename,line_num,answer,result.index.values,rank,sim_levenshtein])

        #paint red or blue at AST
        x = (predict == labels[pair[0]:pair[1]].argmax(axis=1))
        for i in range(len(predict)):
            if x[i] == False:
                G.node(str(i),str(predict[i]),color="red")
            else:
                G.node(str(i),str(predict[i]),color="blue")
        #G.render("tree/" + filename)
    result_table1 = pd.DataFrame(df_top1,columns=["filename","line","label","predict","predict_score","sim_levenshtein"])
    result_table_all = pd.DataFrame(df_all,columns=["filename","line","label","predict","predict_score","sim_levenshtein"])
    result_table3 = pd.DataFrame(df_top3,columns=["filename","line","label","predict","predict_score","sim_levenshtein"])
    result_table5 = pd.DataFrame(df_top5,columns=["filename","line","label","predict","predict_score","sim_levenshtein"])
    result_table10 = pd.DataFrame(df_top10,columns=["filename","line","label","predict","predict_score","sim_levenshtein"])
    #res = result_table1["predict_score"].corr(result_table1["sim_levenshtein"])
    #print(res)

    df.append(result_table1)
    if FLAGS.mode != "val":
        df.append(result_table3)
        df.append(result_table5)
        df.append(result_table10)
        df.append(result_table_all)
        
    #result_table_all.to_csv("log/{}/{}_{}_result_table.csv".format(FLAGS.dataset,FLAGS.model_name,FLAGS.mode))

    #check false positive and true positive
    for result_table in df:
        fp = pd.DataFrame(columns=["filename","label","predict","predict_score","sim_levenshtein"])
        tp = pd.DataFrame(columns=["filename","label","predict","predict_score","sim_levenshtein"])
        for _, item in result_table.iterrows():
            if item["label"] not in item["predict"]:
                fp = fp.append(item)
            else:
                tp = tp.append(item)
        recall = ( len(result_table) - len(fp) ) / len(result_table)
        recalls.append(recall)
        w.write("\nRecall:{}".format(recall) )
        recall_log.write("{},".format(recall) )

    #calculate score_S in journal
    """
    counter1 = 0
    counter2 = 0
    num = 0
    t = 0
    for _, item1 in result_table1.iterrows():
        counter1 += 1
        for _, item2 in result_table1.iterrows():
            counter2 += 1
            if counter1 >= counter2:
                continue
            if item1["filename"] == item2["filename"]:
                num += 1
                if item1["predict"] == item2["predict"]:
                    t += 1
        counter2 = 0
    print(t/num)
    print(t)
    print(num)
    """
    recall_log.write("\n")

#print(pd.Series(np.sum(labels,axis=0)))
print(list(fp["label"].value_counts().index.values))
print(list(tp["label"].value_counts().index.values))
print(list(range(FLAGS.class_num)))
add = list(set(list(range(FLAGS.class_num)))-set(list(tp["label"].value_counts().index.values)))
print(add)
if FLAGS.mode != "val":
    exit()
if recalls[0] <=0.01:
    exit()
target = []
if os.path.exists('data/{}/addition.pkl'.format(FLAGS.dataset)):
    with open('data/{}/addition.pkl'.format(FLAGS.dataset),'rb') as f:
        target = pickle.load(f)
        target.append(add)
else:
    target.append(add)
with open('data/{}/addition.pkl'.format(FLAGS.dataset),'wb') as f:
    pickle.dump(target, f)
