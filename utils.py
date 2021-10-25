import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import json
from networkx.readwrite import json_graph
import h5py
import torch
import random
from sklearn import metrics

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_data_wiki():
    with h5py.File('../data/wiki/wiki.hdf5', 'r') as f:
        adj = f['adj'].value
        features = f['feats'].value
        labels = f['label'].value
        idx_train = f['idx_train'].value
        idx_val = f['idx_val'].value
        idx_test = f['idx_test'].value

    features = sp.lil_matrix(features)
    adj = sp.csr_matrix(adj)
    
    features = row_normalize(features)
    features = np.array(features.todense())

    features = torch.FloatTensor(features)
    labels = torch.FloatTensor(labels)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test) 

    return adj, features, labels, idx_train, idx_val, idx_test

def load_data_amazon():
    with h5py.File('../data/amazon/amazon.hdf5', 'r') as f:
        adj = f['adj'].value
        features = f['feats'].value
        labels = f['label'].value
        idx_train = f['idx_train'].value
        idx_val = f['idx_val'].value
        idx_test = f['idx_test'].value

    features = sp.lil_matrix(features)
    adj = sp.csr_matrix(adj)
    
    features = row_normalize(features)
    features = np.array(features.todense())

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test) 

    return adj, features, labels, idx_train, idx_val, idx_test

def load_data_social(dataset_str):
    with h5py.File('../data/social/'+dataset_str + '.hdf5', 'r') as f:
        adj = f['adj'].value
        features = f['feats'].value
        labels = f['label'].value
        idx_train = f['idx_train'].value
        idx_val = f['idx_val'].value
        idx_test = f['idx_test'].value

    features = sp.lil_matrix(features)
    adj = sp.csr_matrix(adj)
    
    features = row_normalize(features)
    features = np.array(features.todense())

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test) 

    return adj, features, labels, idx_train, idx_val, idx_test

def load_data_ppi():
    with h5py.File('../data/ppi/ppi_data.hdf5', 'r') as f:
        Gdata = f['Gdata'].value
        features = f['feats'].value
        class_map = f['class_map'].value
        idx_train = f['idx_train'].value
        idx_val = f['idx_val'].value
        idx_test = f['idx_test'].value

    Gdata = json.loads(Gdata)
    features = sp.lil_matrix(features)

    G = json_graph.node_link_graph(Gdata)
    adj = nx.adjacency_matrix(G)

    labels = class_map
    features = row_normalize(features)
    features = np.array(features.todense())

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test) 

    return adj, features, labels, idx_train, idx_val, idx_test

def load_data_cite(dataset_str):
    with h5py.File('../data/cite/'+dataset_str + '.hdf5', 'r') as f:
        adj = f['adj'].value
        features = f['feats'].value
        labels = f['label'].value
        idx_train = f['idx_train'].value
        idx_val = f['idx_val'].value
        idx_test = f['idx_test'].value

    features = sp.lil_matrix(features)
    adj = sp.csr_matrix(adj)

    features = row_normalize(features)
    features = np.array(features.todense())

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(np.where(labels)[1])
    
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test) 

    return adj, features, labels, idx_train, idx_val, idx_test


def row_normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalize(mx):
    """normalize sparse matrix"""
    rowsum = np.array(mx.sum(1)) * 0.5
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    colsum = np.array(mx.sum(0)) * 0.5
    c_inv = np.power(colsum, -1).flatten()
    c_inv[np.isinf(c_inv)] = 0.
    c_mat_inv = sp.diags(c_inv)
    mx = r_mat_inv.dot(mx)
    mx = mx.dot(c_mat_inv)
    return mx

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def calc_f1(output, labels):
    pred = np.array(output.tolist())
    target = np.array(labels.tolist())
    pred[pred > 0.5] = 1
    pred[pred <= 0.5] = 0
    return metrics.f1_score(target, pred, average="micro")

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)