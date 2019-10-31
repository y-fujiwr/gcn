import logging.config
import sys,os
import numpy as np
import re
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import trange,tqdm
from scipy.sparse import lil_matrix, csr_matrix
from collections import defaultdict
from pathlib import Path
from graphviz import Digraph
import pandas as pd
from gensim.models.word2vec import Word2Vec
from rulenames import ruleNames_C
import random
import pickle
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

def get_input_features(ast_string, label, start_num, class_num, NODE_TYPE_NUM):
    ast_stream = ast_string.split(" ")[:-1]
    node_array = []
    parent_node_array = []
    pointer = -1
    terminal_flag = False
    #w2v = Word2Vec.load("model/node_w2v_128").wv
    for node in ast_stream:
        if terminal_flag:
            node_array = []
            parent_node_array = []
            terminal_flag = False
        if(node[0]=="("):
            if(len(node) > 1):
                temp = [0] * NODE_TYPE_NUM
                temp[int(node[1:])] = 1
                node_array.append(temp)
                """
                if node[1:] in w2v:
                    node_array.append(w2v[node[1:]].tolist())
                else:
                    node_array.append([0]*128)
                """
                parent_node_array.append(pointer)
                pointer = len(parent_node_array) - 1 
        else:
            pointer = parent_node_array[pointer]
            if pointer == -1:
                terminal_flag = True
    if(pointer != -1):
        print("parse warning!!")
        #return None
    stack = [-1]
    leaf = len(parent_node_array) - 1
    for x in range(len(parent_node_array)-1, 0, -1):
        if x - parent_node_array[x] != 1:
            parent_node_array[leaf] = parent_node_array[x]
            if x != leaf:
                parent_node_array[x] = -2
            if stack[-1] != parent_node_array[x]:
                stack.append(parent_node_array[leaf])
            leaf = x - 1
        else:
            if stack[-1] == parent_node_array[x]:
                parent_node_array[leaf] = stack.pop()
                if x != leaf:
                    parent_node_array[x] = -2
                leaf = x - 1
            else:
                if x != leaf:
                    parent_node_array[x] = -2
    parent_node_array[leaf] = 0

    s_parent_na = []
    s_na = []
    for i in range(len(parent_node_array)):
        x = parent_node_array[i]
        if x != -2:
            s_parent_na.append(x - len([n for n in parent_node_array[:x] if n == -2]))
            s_na.append(node_array[i])

    num_1 = s_parent_na.count(1)
    if num_1 >= 3:
        for i in range(num_1-2):
            s_parent_na.insert(3,2)
            for j in range(4,len(s_parent_na)):
                s_parent_na[j] += 1
            for k in range(len(s_parent_na)-1,0,-1):
                if s_parent_na[k] <= i+2:
                    s_parent_na[k] -= 1
                if s_parent_na[k] == i+1:
                    break
            temp = [0] * NODE_TYPE_NUM
            temp[50] = 1
            s_na.insert(3,temp)
            #s_na.insert(3,w2v["50"].tolist())

    graph = defaultdict(list)
    #node_array = lil_matrix(np.array(s_na, dtype=np.float32)).tocsr()
    #adj_matrix = lil_matrix((len(s_na),len(s_na)))
    G = Digraph(format="png")
    G.attr("node", shape ="circle")
    edges = []
    for i in range(1,len(s_parent_na)):
        #adj_matrix[i, s_parent_na[i]] = 1
        #adj_matrix[s_parent_na[i], i] = 1
        graph[i+start_num].append(s_parent_na[i]+start_num)
        graph[s_parent_na[i]+start_num].append(i+start_num)
        edges.append((s_parent_na[i],i))
    #adj_matrix = adj_matrix.tocsr()
    for i,j in edges:
        G.edge(str(i),str(j))
    #for i in range(len(s_na)):
        #G.node(str(i), str(s_na[i].index(1)))
    labels = [[0] * class_num] * len(s_na)
    labels[0][label] = 1

    return s_na, labels, graph, G #,adj_matrix

def load_ast_features(target_dir_path, class_num, NODE_TYPE_NUM):
    tanni = 10
    first = 50
    print("load train data...")
    dataname = FLAGS.learning_type + "_dataset.pkl"
    traindatapkl = str(Path(target_dir_path,dataname))
    if os.path.exists(traindatapkl):
        print("load")
        filelist = pickle.load(open(traindatapkl,"rb"))
    else:
        #fileGenerator = list(Path(target_dir_path, "train").glob("**"))
        dirIterator = [d for d in Path(target_dir_path, "train").glob("**") if re.search("\D+\d+", str(d))]
        filelist = []
        if FLAGS.learning_type == "node":
            for d in dirIterator:
                temp = list(Path(d).glob("**/*.txt"))
                while len(temp) < first:
                    temp.extend(temp)
                sampled = random.sample(temp,first)
                filelist.extend(sampled)
            train_labels = []
            graphs = defaultdict(list)
            for f in tqdm(filelist):
                try:
                    for target in open(str(f),"r").readlines():
                        label= int(str(f).split(os.path.sep)[-2])
                        __, label, ___, _ = get_input_features(target, label, 0, class_num, NODE_TYPE_NUM)
                        train_labels.extend(label)
                except IndexError:
                    print(f)
                except TypeError:
                    print(f)
            train_labels = np.array(train_labels, dtype=np.int32)
            num_node = pd.Series(np.nonzero(train_labels)[1]).value_counts()
            node_dict = {}
            for i,n in num_node.iteritems():
                node_dict[i] = int(20000000 / class_num / n)
            filelist = []
            for d in range(class_num):
                temp = list(Path(target_dir_path, "train/{}".format(d)).glob("**/*.txt"))
                while len(temp) < node_dict[d]:
                    temp.extend(temp)
                sampled = random.sample(temp,node_dict[d])
                filelist.extend(sampled)
        else:
            for d in dirIterator:
                temp = list(Path(d).glob("**/*.txt"))
                while len(temp) < 100:
                    temp.extend(temp)
                sampled = random.sample(temp,100)
                filelist.extend(sampled)

    if FLAGS.learning_type == "reinforcement":
        if os.path.exists('data/{}/addition.pkl'.format(FLAGS.dataset)):
            with open('data/{}/addition.pkl'.format(FLAGS.dataset),'rb') as f:
                target = pickle.load(f)
            if len(filelist) < (FLAGS.class_num * 100 + my_len(target) * tanni):
                for d in target[-1]:
                    temp = list(Path(target_dir_path, "train/{}".format(d)).glob("**/*.txt"))
                    while len(temp) < tanni:
                        temp.extend(temp)
                    sampled = random.sample(temp,tanni)
                    filelist.extend(sampled)

    pickle.dump(filelist,open(traindatapkl,"wb"))
    start_num = 0
    train_node_arrays = [] 
    train_labels = []
    graphs = defaultdict(list)
    for f in tqdm(filelist):
        try:
            for target in open(str(f),"r").readlines():
                label= int(str(f).split(os.path.sep)[-2])
                node_array, label, graph, _ = get_input_features(target, label, start_num, class_num, NODE_TYPE_NUM)
                start_num += len(node_array)
                train_node_arrays.extend(node_array)
                train_labels.extend(label)
                graphs.update(graph)
        except IndexError:
            print(f)
        except TypeError:
            print(f)
    train_node_arrays = lil_matrix(np.array(train_node_arrays, dtype=np.float32)).tocsr()
    train_labels = np.array(train_labels, dtype=np.int32)
    
    fileGenerator = list(Path(target_dir_path, "test").glob("**/*.txt"))
    test_node_arrays = [] 
    test_labels = []
    
    print("load validation data...")
    for f in tqdm(fileGenerator):
        for target in open(str(f),"r").readlines():
            try:
                label = int(str(f).split(os.path.sep)[-2])
                node_array, label, graph, _ = get_input_features(target, label, start_num, class_num, NODE_TYPE_NUM)
                start_num += len(node_array)
                test_node_arrays.extend(node_array)
                test_labels.extend(label)
                graphs.update(graph)
            except IndexError:
                print(f)
            except TypeError:
                print(f)                
    
    test_node_arrays = lil_matrix(np.array(test_node_arrays, dtype=np.float32)).tocsr()
    test_labels = np.array(test_labels, dtype=np.int32)
    
    return train_node_arrays, train_labels, test_node_arrays, test_labels, graphs

def load_test_ast_features(target_dir_path, class_num, NODE_TYPE_NUM):
    fileGenerator = Path(target_dir_path).glob("**/*.txt")
    start_num = 0
    test_node_arrays = []
    test_labels = []
    positions = []
    graphs = defaultdict(list)
    print("load test data...")
    if FLAGS.mode == "val":
        dataname = "../reinforcement_dataset.pkl"
        traindatapkl = str(Path(target_dir_path,dataname))
        fileGenerator = pickle.load(open(traindatapkl,"rb"))
    for f in tqdm(fileGenerator):
        try:
            for target in open(str(f),"r").readlines():
                label = int(str(f).split(os.path.sep)[-2])
                node_array, label, graph, G = get_input_features(target, label, start_num, class_num, NODE_TYPE_NUM)
                end_num = start_num + len(node_array)
                positions.append(((start_num, end_num),str(f),G))
                start_num = end_num
                test_node_arrays.extend(node_array)
                test_labels.extend(label)
                graphs.update(graph)
        except IndexError:
            print(f)
    test_node_arrays = lil_matrix(np.array(test_node_arrays, dtype=np.float32)).tocsr()
    test_labels = np.array(test_labels, dtype=np.int32)

    return test_node_arrays, test_labels, graphs, positions
    
def analyze_ast(ast_string,name,NODE_TYPE_NUM):
    ast_stream = ast_string.split(" ")
    node_array = []
    parent_node_array = []
    pointer = -1
    terminal_flag = False
    for node in ast_stream:
        if terminal_flag:
            node_array = []
            parent_node_array = []
            terminal_flag = False
        if(node[0]=="("):
            if(len(node) > 1):
                temp = [0] * NODE_TYPE_NUM
                temp[int(node[1:])] = 1
                node_array.append(temp)
                parent_node_array.append(pointer)
                pointer = len(parent_node_array) - 1 
        else:
            pointer = parent_node_array[pointer]
            if pointer == -1:
                terminal_flag = True
            
    if(pointer != -1):
        print("error")
        return None
    stack = [-1]
    leaf = len(parent_node_array) - 1
    for x in range(len(parent_node_array)-1, 0, -1):
        if x - parent_node_array[x] != 1:
            parent_node_array[leaf] = parent_node_array[x]
            if x != leaf:
                parent_node_array[x] = -2
            if stack[-1] != parent_node_array[x]:
                stack.append(parent_node_array[leaf])
            leaf = x - 1
        else:
            if stack[-1] == parent_node_array[x]:
                parent_node_array[leaf] = stack.pop()
                if x != leaf:
                    parent_node_array[x] = -2
                leaf = x - 1
            else:
                if x != leaf:
                    parent_node_array[x] = -2
    parent_node_array[leaf] = 0
    s_parent_na = []
    s_na = []
    for i in range(len(parent_node_array)):
        x = parent_node_array[i]
        if x != -2:
            s_parent_na.append(x - len([n for n in parent_node_array[:x] if n == -2]))
            s_na.append(node_array[i])

    num_1 = s_parent_na.count(1)
    if num_1 >= 3:
        for i in range(num_1-2):
            s_parent_na.insert(3,2)
            for j in range(4,len(s_parent_na)):
                s_parent_na[j] += 1
            for k in range(len(s_parent_na)-1,0,-1):
                if s_parent_na[k] <= i+2:
                    s_parent_na[k] -= 1
                if s_parent_na[k] == i+1:
                    break
            temp = [0] * NODE_TYPE_NUM
            temp[50] = 1
            s_na.insert(3,temp)

    G = Digraph(format="png")
    G.attr("node", shape ="circle")
    edges = []
    for i in range(1,len(s_parent_na)):
        edges.append((s_parent_na[i],i))

    for i,j in edges:
        G.edge(str(i),str(j))
    for i in range(len(s_na)):
        G.node(str(i), str(s_na[i].index(1)) ,color="blue")
        #G.node(str(i), ruleNames_C[s_na[i].index(1)],color="blue")
    G.render("tree/" + name)
    print(s_parent_na)
    return G

def node_embedding(target_dir_path):
    corpus = []
    fileGenerator = Path(target_dir_path).glob("**/*.txt")
    for f in fileGenerator:
        with open(str(f),"r") as s:
            for ast_string in s.readlines():
                ast_stream = pd.Series(ast_string.replace("(","").split(" "))
                corpus.append(ast_stream[ast_stream != ')'].values.tolist()[0:-1])
    
    w2v = Word2Vec(corpus, size=128, workers=16, sg=1, min_count=0)
    w2v.save("model/node_w2v_128")
def my_len(l):
    count = 0
    if isinstance(l, list):
        for v in l:
            count += my_len(v)
        return count
    else:
        return 1
if __name__ == '__main__':
    args = sys.argv
    #logging_setting_path = '../resources/logging/utiltools_log.conf'
    #logging.config.fileConfig(logging_setting_path)
    #logger = logging.getLogger(__file__)
    flags.DEFINE_string('learning_type','node','Select learning mode (reinforcement, node, method).')
    #target_file = args[1]
    a,labels,c,d,e= load_ast_features("data/maven_dataset/",20,85)
    num_node = pd.Series(np.nonzero(labels)[1]).value_counts()
    print(num_node)
    #node_embedding(target_file)