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

NODE_TYPE_NUM = 201
#C:201
#java:84
ruleNames = [
		"translationunit", "primaryexpression", "idexpression", "unqualifiedid",
		"qualifiedid", "nestednamespecifier", "lambdaexpression", "lambdaintroducer",
		"lambdacapture", "capturedefault", "capturelist", "capture", "simplecapture",
		"initcapture", "lambdadeclarator", "postfixexpression", "expressionlist",
		"pseudodestructorname", "unaryexpression", "unaryoperator", "newexpression",
		"newplacement", "newtypeid", "newdeclarator", "noptrnewdeclarator", "newinitializer",
		"deleteexpression", "noexceptexpression", "castexpression", "pmexpression",
		"multiplicativeexpression", "additiveexpression", "shiftexpression", "relationalexpression",
		"equalityexpression", "andexpression", "exclusiveorexpression", "inclusiveorexpression",
		"logicalandexpression", "logicalorexpression", "conditionalexpression",
		"assignmentexpression", "assignmentoperator", "expression", "constantexpression",
		"nestedParenthesesBlock", "statement", "labeledstatement", "expressionstatement",
		"compoundstatement", "statementseq", "selectionstatement", "condition",
		"iterationstatement", "forinitstatement", "forrangedeclaration", "forrangeinitializer",
		"jumpstatement", "unknownstatement", "declarationstatement", "declarationseq",
		"declaration", "blockdeclaration", "aliasdeclaration", "simpledeclaration",
		"static_assertdeclaration", "emptydeclaration", "attributedeclaration",
		"declspecifier", "declspecifierseq", "storageclassspecifier", "functionspecifier",
		"typedefname", "typespecifier", "trailingtypespecifier", "typespecifierseq",
		"trailingtypespecifierseq", "simpletypespecifier", "typename", "decltypespecifier",
		"elaboratedtypespecifier", "enumname", "enumspecifier", "enumhead", "opaqueenumdeclaration",
		"enumkey", "enumbase", "enumeratorlist", "enumeratordefinition", "enumerator",
		"namespacename", "originalnamespacename", "namespacedefinition", "namednamespacedefinition",
		"originalnamespacedefinition", "extensionnamespacedefinition", "unnamednamespacedefinition",
		"namespacebody", "namespacealias", "namespacealiasdefinition", "qualifiednamespacespecifier",
		"usingdeclaration", "usingdirective", "asmdefinition", "linkagespecification",
		"attributespecifierseq", "attributespecifier", "alignmentspecifier", "attributelist",
		"attribute", "attributetoken", "attributescopedtoken", "attributenamespace",
		"attributeargumentclause", "balancedtokenseq", "balancedtoken", "initdeclaratorlist",
		"initdeclarator", "declarator", "ptrdeclarator", "noptrdeclarator", "parametersandqualifiers",
		"trailingreturntype", "ptroperator", "cvqualifierseq", "cvqualifier",
		"refqualifier", "declaratorid", "typeid", "abstractdeclarator", "ptrabstractdeclarator",
		"noptrabstractdeclarator", "abstractpackdeclarator", "noptrabstractpackdeclarator",
		"parameterdeclarationclause", "parameterdeclarationlist", "parameterdeclaration",
		"functiondefinition", "functionbody", "initializer", "braceorequalinitializer",
		"initializerclause", "initializerlist", "bracedinitlist", "classname",
		"classspecifier", "classhead", "classheadname", "classvirtspecifier",
		"classkey", "memberspecification", "memberdeclaration", "memberdeclaratorlist",
		"memberdeclarator", "virtspecifierseq", "virtspecifier", "purespecifier",
		"baseclause", "basespecifierlist", "basespecifier", "classordecltype",
		"basetypespecifier", "accessspecifier", "conversionfunctionid", "conversiontypeid",
		"conversiondeclarator", "ctorinitializer", "meminitializerlist", "meminitializer",
		"meminitializerid", "operatorfunctionid", "literaloperatorid", "templatedeclaration",
		"templateparameterlist", "templateparameter", "typeparameter", "simpletemplateid",
		"templateid", "templatename", "templateargumentlist", "templateargument",
		"typenamespecifier", "explicitinstantiation", "explicitspecialization",
		"tryblock", "functiontryblock", "handlerseq", "handler", "exceptiondeclaration",
		"throwexpression", "exceptionspecification", "dynamicexceptionspecification",
		"typeidlist", "noexceptspecification", "rightShift", "rightShiftAssign",
		"operator", "literal", "booleanliteral", "pointerliteral", "userdefinedliteral"
]
def get_input_features(ast_string, label, class_num, node_num):
    ast_stream = ast_string.split(" ")
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
        graph[i].append(s_parent_na[i])
        graph[s_parent_na[i]].append(i)
        edges.append((s_parent_na[i],i))
    #adj_matrix = adj_matrix.tocsr()
    for i,j in edges:
        G.edge(str(i),str(j))
    #for i in range(len(s_na)):
        #G.node(str(i), str(s_na[i].index(1)))
    labels = [[0] * class_num] * node_num
    labels[0][label] = 1

    add_num = node_num - len(s_na)
    for i in range(len(s_na),node_num):
        graph[i]
    s_na.extend([[0] * NODE_TYPE_NUM] * add_num)

    return s_na, labels, graph, G #,adj_matrix

def load_ast_features(target_dir_path, class_num):
    start_num = 0
    train_node_arrays = [] 
    train_labels = []
    train_graphs = []
    node_max = 0
    fileGenerator = Path(target_dir_path).glob("**/*.txt")
    for f in fileGenerator:
        try:
            for target in open(str(f),"r").readlines():
                x = get_node_num(target) 
                if node_max < x:
                    node_max = x 
        except MemoryError:
            print(f)
    print("load train data...")
    fileGenerator = Path(target_dir_path, "train").glob("**/*.txt")
    for f in tqdm(fileGenerator):
        try:
            for target in open(str(f),"r").readlines():
                label= int(str(f).split(os.path.sep)[-2])
                node_array, label, graph, _ = get_input_features(target, label, class_num, node_max)
                start_num += len(node_array)
                train_node_arrays.append(lil_matrix(np.array(node_array, dtype=np.float32)).tocsr())
                train_labels.append(np.array(label, dtype=np.int32))
                train_graphs.append(graph)
        except IndexError:
            print(f)
    
    fileGenerator = Path(target_dir_path, "test").glob("**/*.txt")
    test_node_arrays = [] 
    test_labels = []
    test_graphs = []
    
    print("load validation data...")
    for f in tqdm(fileGenerator):
        try:
            for target in open(str(f),"r").readlines():
                label = int(str(f).split(os.path.sep)[-2])
                node_array, label, graph, _ = get_input_features(target, label, class_num, node_max)
                start_num += len(node_array)
                test_node_arrays.append(lil_matrix(np.array(node_array, dtype=np.float32)).tocsr())
                test_labels.append(np.array(label, dtype=np.int32))
                test_graphs.append(graph)
        except IndexError:
            print(f)
        
    return train_node_arrays, train_labels, train_graphs, test_node_arrays, test_labels, test_graphs

def load_test_ast_features(target_dir_path, class_num):
    fileGenerator = Path(target_dir_path).glob("**/*.txt")
    start_num = 0
    test_node_arrays = []
    test_labels = []
    positions = []
    graphs = []
    node_max = 0
    print("load test data...")
    for f in fileGenerator:
        try:
            for target in open(str(f),"r").readlines():
                x = get_node_num(target) 
                if node_max < x:
                    node_max = x 
        except IndexError:
            print(f)
    fileGenerator = Path(target_dir_path).glob("**/*.txt")
    for f in tqdm(fileGenerator):
        try:
            for target in open(str(f),"r").readlines():
                label = int(str(f).split(os.path.sep)[-2])
                node_array, label, graph, G = get_input_features(target, label, class_num, node_max)
                end_num = start_num + len(node_array)
                positions.append(((start_num, end_num),str(f),G))
                start_num = end_num
                test_node_arrays.append(lil_matrix(np.array(node_array, dtype=np.float32)).tocsr())
                test_labels.append(np.array(label, dtype=np.int32))
                graphs.append(graph)
        except IndexError:
            print(f)

    return test_node_arrays, test_labels, graphs, positions
    
def analyze_ast(ast_string,name):
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
        #G.node(str(i), ruleNames[s_na[i].index(1)],color="blue")
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

def get_node_num(ast_string):
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
    return len(s_na)

if __name__ == '__main__':
    args = sys.argv
    #logging_setting_path = '../resources/logging/utiltools_log.conf'
    #logging.config.fileConfig(logging_setting_path)
    #logger = logging.getLogger(__file__)

    target_file = args[1]
    #analyze_ast(open(target_file, "r").readline(), Path(target_file).name)
    
    node_arrays, labels, graphs, positions = load_test_ast_features(target_file,3)
    print(node_arrays)

    #node_embedding(target_file)
   
