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

NODE_TYPE_NUM = 201

def get_input_features(ast_string, label, start_num):
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

    graph = defaultdict(list)
    #node_array = lil_matrix(np.array(s_na, dtype=np.float32)).tocsr()
    #adj_matrix = lil_matrix((len(s_na),len(s_na)))
    for i in range(1,len(s_parent_na)):
        #adj_matrix[i, s_parent_na[i]] = 1
        #adj_matrix[s_parent_na[i], i] = 1
        graph[i+start_num].append(s_parent_na[i]+start_num)
        graph[s_parent_na[i]+start_num].append(i+start_num)
    #adj_matrix = adj_matrix.tocsr()

    class_num = 704
    labels = [[0] * class_num] * len(s_na)
    labels[0][label] = 1

    return s_na, labels, graph #,adj_matrix

def load_ast_features(target_dir_path):
    fileGenerator = Path(target_dir_path, "train").glob("**/*.txt")
    start_num = 0
    train_node_arrays = [] 
    train_labels = []
    graphs = defaultdict(list)
    for f in fileGenerator:
        for target in open(str(f),"r").readlines():
            label= int(str(f).split(os.path.sep)[-2])
            node_array, label, graph = get_input_features(target, label, start_num)
            start_num += len(node_array)
            train_node_arrays.extend(node_array)
            train_labels.extend(label)
            graphs.update(graph)
    train_node_arrays = lil_matrix(np.array(train_node_arrays, dtype=np.float32)).tocsr()
    train_labels = np.array(train_labels, dtype=np.int32)
    
    fileGenerator = Path(target_dir_path, "test").glob("**/*.txt")
    test_node_arrays = [] 
    test_labels = []
    
    for f in fileGenerator:
        for target in open(str(f),"r").readlines():
            label= int(str(f).split(os.path.sep)[-2])
            node_array, label, graph = get_input_features(target, label, start_num)
            start_num += len(node_array)
            test_node_arrays.extend(node_array)
            test_labels.extend(label)
            graphs.update(graph)
    
    test_node_arrays = lil_matrix(np.array(test_node_arrays, dtype=np.float32)).tocsr()
    test_labels = np.array(test_labels, dtype=np.int32)
    
    return train_node_arrays, train_labels, test_node_arrays, test_labels, graphs
    


if __name__ == '__main__':
    args = sys.argv
    #logging_setting_path = '../resources/logging/utiltools_log.conf'
    #logging.config.fileConfig(logging_setting_path)
    #logger = logging.getLogger(__file__)

    target_dir_path = args[1]

    input_features = load_ast_features(target_dir_path)

    print(input_features[0])

    
