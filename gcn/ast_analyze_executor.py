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
use_lsi = None
dictionary_lsi = None
vectors_lsi = None
dictionary_node = {}
kind_node = 0
invalid_keys = []

def read_lsi(dim):
    global dictionary_lsi
    global vectors_lsi
    global dictionary_node
    global invalid_keys
    with open(f"data/{FLAGS.dataset}/dictionary_lsi_{FLAGS.lsi}.pickle", "rb") as fi:
        dictionary_lsi = pickle.load(fi)
    vectors_lsi = np.load(f"data/{FLAGS.dataset}/vec_lsi_{FLAGS.lsi}.npy")
    with open(f"data/{FLAGS.dataset}/dictionary.txt") as fi:
        identifier_list = fi.read().splitlines()
    for i in range(len(identifier_list)):
        try:
            dictionary_node[i+dim] = vectors_lsi[dictionary_lsi[identifier_list[i]]]
        except KeyError:
            invalid_keys.append(i+dim)
            continue

def get_input_features(ast_string, label, start_num, class_num, NODE_TYPE_NUM):
    if ast_string == "(8 )":
        print("aaaa")
    ast_stream = ast_string.split(" ")[:-1]
    node_array = []
    parent_node_array = []
    pointer = -1
    terminal_flag = False
    vector_dimension = 0
    if FLAGS.language == "java":
        vector_dimension = 85
    elif FLAGS.language == "c":
        vector_dimension = 201

    #Translate AST stream to node_array and parent_node_array
    for node in ast_stream:
        if terminal_flag:
            node_array = []
            parent_node_array = []
            terminal_flag = False
        if(node[0]=="("):
            if(len(node) > 1):
                if use_lsi != None:
                    if int(node[1:]) > kind_node:
                        if int(node[1:]) not in invalid_keys:
                            temp = [0] * kind_node
                            temp = np.concatenate([temp,dictionary_node[int(node[1:])]])
                        else:
                            temp = [0] * (kind_node + len(vectors_lsi[0]))
                            if FLAGS.language == "java":
                                temp[42] = 1
                            elif FLAGS.language == "c":
                                temp[3] = 1
                    else:
                        temp = [0] * (kind_node + len(vectors_lsi[0]))
                        temp[int(node[1:])] = 1
                else:
                    if FLAGS.normalize:
                        temp = [0] * vector_dimension
                        if int(node[1:]) >= vector_dimension:
                            if FLAGS.language == "java":
                                temp[42] = 1
                            elif FLAGS.language == "c":
                                temp[3] = 1
                        else:
                            temp[int(node[1:])] = 1
                    else:
                        temp = [0] * NODE_TYPE_NUM
                        temp[int(node[1:])] = 1
                node_array.append(temp)
                parent_node_array.append(pointer)
                pointer = len(parent_node_array) - 1 
        else:
            pointer = parent_node_array[pointer]
            if pointer == -1:
                terminal_flag = True
    if(pointer != -1) and FLAGS.language == "java":
        print("parse warning!!")
        #return None

    #Remove redundant nodes
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

    short_parent_node_array = []
    short_node_array = []
    for i in range(len(parent_node_array)):
        x = parent_node_array[i]
        if x != -2:
            short_parent_node_array.append(x - len([n for n in parent_node_array[:x] if n == -2]))
            short_node_array.append(node_array[i])

    num_1 = short_parent_node_array.count(1)
    if num_1 >= 3:
        for i in range(num_1-2):
            short_parent_node_array.insert(3,2)
            for j in range(4,len(short_parent_node_array)):
                short_parent_node_array[j] += 1
            for k in range(len(short_parent_node_array)-1,0,-1):
                if short_parent_node_array[k] <= i+2:
                    short_parent_node_array[k] -= 1
                if short_parent_node_array[k] == i+1:
                    break
            if use_lsi != None:
                temp = [0] * (kind_node + len(vectors_lsi[0]))
            elif FLAGS.normalize:
                temp = [0] * vector_dimension
            else:
                temp = [0] * NODE_TYPE_NUM
            temp[50] = 1
            short_node_array.insert(3,temp)

    #Create a dictionary-type graph
    graph = defaultdict(list)
    """
    #node_array = lil_matrix(np.array(short_node_array, dtype=np.float32)).tocsr()
    #adj_matrix = lil_matrix((len(short_node_array),len(short_node_array)))
    """
    G = Digraph(format="png")
    G.attr("node", shape ="circle")
    edges = []
    for i in range(1,len(short_parent_node_array)):
        """
        #adj_matrix[i, short_parent_node_array[i]] = 1
        #adj_matrix[short_parent_node_array[i], i] = 1
        """
        graph[i+start_num].append(short_parent_node_array[i]+start_num)
        graph[short_parent_node_array[i]+start_num].append(i+start_num)
        edges.append((short_parent_node_array[i],i))
    #adj_matrix = adj_matrix.tocsr()
    for i,j in edges:
        G.edge(str(i),str(j))
    #for i in range(len(short_node_array)):
        #G.node(str(i), str(short_node_array[i].index(1)))
    labels = [[0] * class_num] * len(short_node_array)
    labels[0][label] = 1

    return short_node_array, labels, graph, G #,adj_matrix

def load_ast_features(target_dir_path, class_num, NODE_TYPE_NUM):
    global kind_node
    global use_lsi
    use_lsi = FLAGS.lsi
    if FLAGS.language == "java":
        kind_node = 85
    elif FLAGS.language == "c":
        kind_node = 201
    if use_lsi != None:
        print("use lsi")
        read_lsi(kind_node+1)
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
                    print(f"indexerror:{f}")
                except TypeError:
                    print(f"typeerror:{f}")
            train_labels = np.array(train_labels, dtype=np.int32)
            num_node = pd.Series(np.nonzero(train_labels)[1]).value_counts()
            node_dict = {}
            for i,n in num_node.iteritems():
                node_dict[i] = int(40000000 / class_num / n)
            filelist = []
            for d in range(class_num):
                temp = list(Path(target_dir_path, f"train/{d}").glob("**/*.txt"))
                while len(temp) < node_dict[d]:
                    temp.extend(temp)
                sampled = random.sample(temp,node_dict[d])
                filelist.extend(sampled)
        elif FLAGS.learning_type in ["method","reinforcement"]:
            for d in dirIterator:
                temp = list(Path(d).glob("**/*.txt"))
                while len(temp) < first:
                    temp.extend(temp)
                sampled = random.sample(temp,first)
                filelist.extend(sampled)
        else:
            for d in dirIterator:
                filelist.extend(list(Path(d).glob("**/*.txt")))

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
                if len(graph) != len(node_array):
                    print(len(graph))
                    print(len(node_array))
                    print(label)
                    print(str(f))
        #except IndexError:
            #print(f"indexerror:{f}")
        except TypeError:
            print(f"typeerror:{f}")
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
    global kind_node
    global use_lsi
    use_lsi = FLAGS.lsi
    if FLAGS.language == "java":
        kind_node = 85
    elif FLAGS.language == "c":
        kind_node = 201
    if use_lsi != None:
        print("use lsi")
        read_lsi(kind_node+1)
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
        line_num = -1
        try:
            for target in open(str(f),"r").readlines():
                line_num += 1
                label = int(str(f).split(os.path.sep)[-2])
                node_array, label, graph, G = get_input_features(target, label, start_num, class_num, NODE_TYPE_NUM)
                #print debug
                if len(graph) == 0:
                    print(str(f))
                    exit()
                end_num = start_num + len(node_array)
                positions.append(((start_num, end_num),str(f),line_num,G))
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
    short_parent_node_array = []
    short_node_array = []
    for i in range(len(parent_node_array)):
        x = parent_node_array[i]
        if x != -2:
            short_parent_node_array.append(x - len([n for n in parent_node_array[:x] if n == -2]))
            short_node_array.append(node_array[i])

    num_1 = short_parent_node_array.count(1)
    if num_1 >= 3:
        for i in range(num_1-2):
            short_parent_node_array.insert(3,2)
            for j in range(4,len(short_parent_node_array)):
                short_parent_node_array[j] += 1
            for k in range(len(short_parent_node_array)-1,0,-1):
                if short_parent_node_array[k] <= i+2:
                    short_parent_node_array[k] -= 1
                if short_parent_node_array[k] == i+1:
                    break
            temp = [0] * NODE_TYPE_NUM
            temp[50] = 1
            short_node_array.insert(3,temp)

    G = Digraph(format="png")
    G.attr("node", shape ="circle")
    edges = []
    for i in range(1,len(short_parent_node_array)):
        edges.append((short_parent_node_array[i],i))

    for i,j in edges:
        G.edge(str(i),str(j))
    for i in range(len(short_node_array)):
        G.node(str(i), str(short_node_array[i].index(1)) ,color="blue")
        #G.node(str(i), ruleNames_C[short_node_array[i].index(1)],color="blue")
    G.render("tree/" + name)
    print(short_parent_node_array)
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
    flags.DEFINE_string('dataset', 'small', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
    FLAGS.dataset = "bcb_identifier"
    flags.DEFINE_string('language', 'java', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
    
    dim = 85
    num = 42
    read_lsi(dim+1)
    #ノードベクトル作成
    vector_dimension = 0
    if FLAGS.language == "java":
        vector_dimension = 85
    elif FLAGS.language == "c":
        vector_dimension = 201
    temp = [0] * vector_dimension
    temp = np.concatenate([temp,dictionary_node[86]])

    get_input_features()