import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse import csgraph
import sys
import time
import argparse
import torch


def aug_random_walk(adj):
    adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv = np.power(row_sum, -1.0).flatten()
    d_mat = sp.diags(d_inv)
    return (d_mat.dot(adj)).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
    

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt).tocoo()


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def laplacian(mx, norm):
    """Laplacian-normalize sparse matrix"""
    assert (all(len(row) == len(mx) for row in mx)), "Input should be a square matrix"

    return csgraph.laplacian(adj, normed=norm)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def load_data(path="../data", dataset="cora"):
    """
    ind.[:dataset].x     => the feature vectors of the training instances (scipy.sparse.csr.csr_matrix)
    ind.[:dataset].y     => the one-hot labels of the labeled training instances (numpy.ndarray)
    ind.[:dataset].allx  => the feature vectors of both labeled and unlabeled training instances (csr_matrix)
    ind.[:dataset].ally  => the labels for instances in ind.dataset_str.allx (numpy.ndarray)
    ind.[:dataset].graph => the dict in the format {index: [index of neighbor nodes]} (collections.defaultdict)
    ind.[:dataset].tx => the feature vectors of the test instances (scipy.sparse.csr.csr_matrix)
    ind.[:dataset].ty => the one-hot labels of the test instances (numpy.ndarray)
    ind.[:dataset].test.index => indices of test instances in graph, for the inductive setting
    """
    print("Upload {} dataset.".format(dataset))

    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []

    for i in range(len(names)):
        with open("{}/ind.{}.{}".format(path, dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)

    test_idx_reorder = parse_index_file("{}/ind.{}.test.index".format(path, dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Citeseer dataset contains some isolated nodes in the graph
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended

        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    print("| # of nodes : {}".format(adj.shape[0]))
    print("| # of edges : {}".format(adj.sum().sum() / 2))

    #features = normalize(features)
    print("| # of features : {}".format(features.shape[1]))
    print("| # of clases   : {}".format(ally.shape[1]))

    features = torch.FloatTensor(np.array(features.todense()))
    sparse_mx = adj.tocoo().astype(np.float32)

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    if dataset == 'citeseer':
        save_label = np.where(labels)[1]
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)
    idx_test = test_idx_range.tolist()

    print("| # of train set : {}".format(len(idx_train)))
    print("| # of val set   : {}".format(len(idx_val)))
    print("| # of test set  : {}".format(len(idx_test)))

    idx_train, idx_val, idx_test = list(map(lambda x: torch.LongTensor(x), [idx_train, idx_val, idx_test]))

    def missing_elements(L):
        start, end = L[0], L[-1]
        return sorted(set(range(start, end + 1)).difference(L))

    if dataset == 'citeseer':
        L = np.sort(idx_test)
        missing = missing_elements(L)

        for element in missing:
            save_label = np.insert(save_label, element, 0)

        labels = torch.LongTensor(save_label)

    return adj, features, labels, idx_train, idx_val, idx_test

def graph_decompose(adj,graph_name,k,metis_p,strategy="edge"):
    '''
    Input:
        adj:the adjacency matrix of original graph
        graph_name:"cora","citeseer","pubmed"
        k:decompose into k subgraphs
        metis_p:"no_skeleton","all_skeleton","number" (depending on metis preprocessing) 
        strategy:"edge" (for edge_decomposition),"node" (for node_decomposition)
    Output:
        the decomposed subgraphs
    '''
    print("Skeleton:",metis_p)
    print("Strategy:",strategy)
    g,g_rest,edges_rest,gs=get_graph_skeleton(adj,graph_name,k,metis_p)
    gs=allocate_edges(g_rest,edges_rest, gs, strategy)
       
    re=[]       
   
    #print the info of nodes and edges of subgraphs 
    edge_num_avg=0
    compo_num_avg=0
    print("Subgraph information:")
    for i in range(k):
        nodes_num=gs[i].number_of_nodes()
        edge_num=gs[i].number_of_edges()
        compo_num=nx.number_connected_components(gs[i])
        print("\t",nodes_num,edge_num,compo_num)
        edge_num_avg+=edge_num
        compo_num_avg+=compo_num
        re.append(nx.to_scipy_sparse_matrix(gs[i])) 
        
    #check the shared edge number in all subgrqphs
    edge_share=set(sort_edge(gs[0].edges()))
    for i in range(k):        
        edge_share&=set(sort_edge(gs[i].edges()))
        
    print("\tShared edge number is: %d"%len(edge_share))
    print("\tAverage edge number:",edge_num_avg/k) 
    print("\tAverage connected component number:",compo_num_avg/k)
    print("\n"+"-"*70+"\n")
    return re

def sort_edge(edges):
    edges=list(edges)
    for i in range(len(edges)):
        u=edges[i][0]
        v=edges[i][1]
        if u > v:
            edges[i]=(v,u)
    return edges

def get_graph_skeleton(adj,graph_name,k,metis_p): 
    '''
    Input:
        adj:the adjacency matrix of original graph
        graph_name:"cora","citeseer","pubmed"
        k:decompose into k subgraphs
        metis_p:"no_skeleton","all_skeleton","k" 
    Output:
        g:the original graph
        g_rest:the rest graph
        edges_rest:the rest edges
        gs:the skeleton of the graph for every subgraph
    '''
    g=nx.from_numpy_matrix(adj.todense())
    num_nodes=g.number_of_nodes()
    print("Original nodes number:",num_nodes)
    num_edges=g.number_of_edges()
    print("Original edges number:",num_edges)  
    print("Original connected components number:",nx.number_connected_components(g),"\n")    
    
    g_dic=dict()
    
    for v,nb in g.adjacency():
        g_dic[v]=[u[0] for u in nb.items()] 
            
    #initialize all the subgrapgs, add the nodes
    gs=[nx.Graph() for i in range(k)]
    for i in range(k):
        gs[i].add_nodes_from([i for i in range(num_nodes)])
    
    if metis_p=="no_skeleton":
        #no skeleton
        g_rest=g
        edges_rest=list(g_rest.edges())
    else:    
        if metis_p=="all_skeleton":
            #doesn't use metis to cut any edge
            graph_cut=g
        else:
            #read the cluster info from file
            f=open("metis_file/"+graph_name+".graph.part.%s"%metis_p,'r')
            cluster=dict()  
            i=0
            for lines in f:
                cluster[i]=eval(lines.strip("\n"))
                i+=1
           
            #get the graph cut by Metis    
            graph_cut=nx.Graph()
            graph_cut.add_nodes_from([i for i in range(num_nodes)])  
            
            for v in range(num_nodes):
                v_class=cluster[v]
                for u in g_dic[v]:
                    if cluster[u]==v_class:
                        graph_cut.add_edge(v,u)
            
        subgs=list(nx.connected_component_subgraphs(graph_cut))
        print("After Metis,connected component number:",len(subgs))
        
                
        #add the edges of spanning tree, get the skeleton
        for i in range(k):
            for subg in subgs:
                T=get_spanning_tree(subg)
                gs[i].add_edges_from(T)
        
        #get the rest graph including all the edges except the shared egdes of spanning trees
        edge_set_share=set(sort_edge(gs[0].edges()))
        for i in range(k):
            edge_set_share&=set(sort_edge(gs[i].edges()))
        edge_set_total=set(sort_edge(g.edges()))
        edge_set_rest=edge_set_total-edge_set_share   
        edges_rest=list(edge_set_rest)
        g_rest=nx.Graph()
        g_rest.add_nodes_from([i for i in range(num_nodes)])
        g_rest.add_edges_from(edges_rest)
       
          
    #print the info of nodes and edges of subgraphs
    print("Skeleton information:")
    for i in range(k):
        print("\t",gs[i].number_of_nodes(),gs[i].number_of_edges(),nx.number_connected_components(gs[i])) 
        
    edge_set_share=set(sort_edge(gs[0].edges()))
    for i in range(k):
        edge_set_share&=set(sort_edge(gs[i].edges()))
    print("\tShared edge number is: %d\n"%len(edge_set_share))
    
    return g,g_rest,edges_rest,gs

def get_spanning_tree(g):
    '''
    Input:Graph
    Output:list of the edges in spanning tree
    '''
    g_dic=dict()
    for v,nb in g.adjacency():
        g_dic[v]=[u[0] for u in nb.items()]
        np.random.shuffle(g_dic[v])
    flag_dic=dict()
    if g.number_of_nodes() ==1:
        return []
    gnodes=np.array(g.nodes)
    np.random.shuffle(gnodes)
    
    for v in gnodes:
        flag_dic[v]=0
    
    current_path=[]
    
    def dfs(u):
        stack=[u]
        current_node=u
        flag_dic[u]=1
        while len(current_path)!=(len(gnodes)-1):
            pop_flag=1
            for v in g_dic[current_node]:
                if flag_dic[v]==0:
                    flag_dic[v]=1
                    current_path.append((current_node,v))  
                    stack.append(v)
                    current_node=v
                    pop_flag=0
                    break
            if pop_flag:
                stack.pop()
                current_node=stack[-1]     
    dfs(gnodes[0])        
    return current_path

def allocate_edges(g_rest,edges_rest, gs, strategy):
    '''
    Input:
        g_rest:the rest graph
        edges_rest:the rest edges
        gs:the skeleton of the graph for every subgraph
        strategy:"edge" (for edge_decomposition),"node" (for node_decomposition)
    Output:
        the decomposed graphs after allocating rest edges
    '''
    k=len(gs)
    if strategy=="edge":  
        print("Allocate the rest edges randomly and averagely.")
        np.random.shuffle(edges_rest)
        t=int(len(edges_rest)/k)
        
        #add edges
        for i in range(k):       
            if i == k-1:
                gs[i].add_edges_from(edges_rest[t*i:])
            else:
                gs[i].add_edges_from(edges_rest[t*i:t*(i+1)])        
        return gs
    
    elif strategy=="node":
        print("Allocate the edges of each nodes randomly and averagely.")
        g_dic=dict()    
        for v,nb in g_rest.adjacency():
            g_dic[v]=[u[0] for u in nb.items()]
            np.random.shuffle(g_dic[v])
        
        def sample_neighbors(nb_ls,k):
            np.random.shuffle(nb_ls)
            ans=[]
            for i in range(k):
                ans.append([])
            if len(nb_ls) == 0:
                return ans
            if len(nb_ls) > k:
                t=int(len(nb_ls)/k)
                for i in range(k):
                    ans[i]+=nb_ls[i*t:(i+1)*t]
                nb_ls=nb_ls[k*t:]
            '''
            if len(nb_ls)>0:
                for i in range(k):
                    ans[i].append(nb_ls[i%len(nb_ls)])
            '''
            
            
            if len(nb_ls)>0:
                for i in range(len(nb_ls)):
                    ans[i].append(nb_ls[i])
            
            np.random.shuffle(ans)
            return ans
        
        #add edges
        for v,nb in g_dic.items():
            ls=np.array(sample_neighbors(nb,k))
            for i in range(k):
                gs[i].add_edges_from([(v,j) for j in ls[i]])
        
        return gs