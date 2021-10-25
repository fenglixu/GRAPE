from __future__ import division
from __future__ import print_function

import time
from motif_search import *
from utils import *
from models import GCN
import sys
import os
import setproctitle
import json
import datetime
import GPUtil
import scipy.sparse as sp
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
import search

gpu_module = sys.argv[1]

test_run = 10
attn = True
model_name = 'GRAPE'

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_module
setproctitle.setproctitle(model_name)
early_stopping = 50
method = 'incsec' # 1 for incsec, 2 for gtrie, 3 for esu

flag_acc = True # accumulate motif count or not

# grape model hyperparameter setting
num_genes = 5
seed = 42
nepoch = 500

# Set random seed
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)


cite_parameter = [16,0.01,0.00005,0.3]
social_parameter = [32,0.03,0.00005,0.3]

def evaluate(pred, target, idx):
    pred = F.log_softmax(pred, dim=1)
    loss = F.nll_loss(pred[idx], target[idx])
    acc = accuracy(pred[idx], target[idx]).item() 
    return loss, acc

def train_model(ft, nlayer, nepoch, candidate_adj, features, labels, idx_train, idx_val, idx_test, attn, lr, weight_decay, dropout, hidden):
    # flatten the ADJ of different motifs and add in a self-loop
    ngene = len(candidate_adj)
    nrole = [len(item) for item in candidate_adj]
    nclass = labels.max().item() + 1
    model = GCN(nfeat=features.shape[1], nlayer=nlayer, nhid=hidden, nclass=nclass, nrole=nrole, ngene=ngene, dropout=dropout, attn=attn)
    cur_lr = lr
    optimizer = optim.Adam(model.parameters(), lr=cur_lr, weight_decay=weight_decay)

    if torch.cuda.is_available():
        model.cuda()
        features = features.cuda()
        candidate_adj = [[itemtemp.cuda() for itemtemp in temp] for temp in candidate_adj]
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    loss_val_list = []
    # Train model
    t_total = time.time()
    for epoch in range(nepoch):
        # Construct feed dictionary
        model.train()
        optimizer.zero_grad()
        output = model(features, candidate_adj)
        
        loss_train, acc_train = evaluate(output, labels, idx_train)

        loss_train.backward()
        optimizer.step()

        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, candidate_adj)

        loss_val, acc_val = evaluate(output, labels, idx_val)
        loss_val_list.append(loss_val.item())

        if epoch%10==1:
            ft.write("{:.4f}".format(time.time() - t_total)+",{:.4f};".format(loss_train))
            print('Epoch: {:04d}'.format(epoch+1), 'loss_train: {:.4f}'.format(loss_train.item()), 'acc_train: {:.4f}'.format(acc_train),
                'loss_val: {:.4f}'.format(loss_val.item()),'acc_val: {:.4f}'.format(acc_val))
        if epoch%100==99:
            cur_lr = 0.5 * cur_lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = cur_lr

        if epoch > 200 and loss_val_list[-1] > np.mean(loss_val_list[-(early_stopping+1):-1]):
            break 
        
    ft.write("\n") 
    # Test model
    model.eval()
    output = model(features, candidate_adj)

    loss_test, acc_test = evaluate(output, labels, idx_test)
    print("Train accuracy= {:.4f}".format(acc_train), "Val accuracy= {:.4f}".format(acc_val), "Test accuracy= {:.4f}".format(acc_test), "epoch= {:04d}".format(epoch))

    return acc_test

dataset = ['cite|cora', 'cite|citeseer', 'amazon', 'social|Amherst', 'social|Hamilton', 'social|Rochester', 'social|Lehigh', 'social|Johns Hopkins']


# undirected motif pool
motif_line = np.zeros((2, 2), dtype=np.int32)
motif_line[0, 1] = 1
motif_line[1, 0] = 1
motif_line_gene = [None, motif_line]

motif_twoline = np.zeros((3, 3), dtype=np.int32)
motif_twoline[0, 1] = 1
motif_twoline[1, 0] = 1
motif_twoline[1, 2] = 1
motif_twoline[2, 1] = 1
motif_twoline_gene = [motif_line, motif_twoline]

motif_twostar = np.zeros((3, 3), dtype=np.int32)
motif_twostar[0, 1] = 1
motif_twostar[1, 0] = 1
motif_twostar[0, 2] = 1
motif_twostar[2, 0] = 1
motif_twostar_gene = [motif_line, motif_twostar]

motif_triangle = np.zeros((3, 3), dtype=np.int32)
motif_triangle[0, 1] = 1
motif_triangle[0, 2] = 1
motif_triangle[1, 2] = 1
motif_triangle[1, 0] = 1
motif_triangle[2, 0] = 1
motif_triangle[2, 1] = 1
motif_triangle_gene = [motif_twoline, motif_triangle]

motif_trianglestar = np.zeros((4, 4), dtype=np.int32)
motif_trianglestar[0, 1] = 1
motif_trianglestar[0, 2] = 1
motif_trianglestar[1, 2] = 1
motif_trianglestar[1, 0] = 1
motif_trianglestar[2, 0] = 1
motif_trianglestar[2, 1] = 1
motif_trianglestar[0, 3] = 1
motif_trianglestar[3, 0] = 1
motif_trianglestar_gene = [motif_triangle, motif_trianglestar]

motif_threestar = np.zeros((4, 4), dtype=np.int32)
motif_threestar[0, 1] = 1
motif_threestar[1, 0] = 1
motif_threestar[0, 2] = 1
motif_threestar[2, 0] = 1
motif_threestar[0, 3] = 1
motif_threestar[3, 0] = 1
motif_threestar_gene = [motif_twostar, motif_threestar]

motif_threeline = np.zeros((4, 4), dtype=np.int32)
motif_threeline[0, 1] = 1
motif_threeline[1, 0] = 1
motif_threeline[1, 2] = 1
motif_threeline[2, 1] = 1
motif_threeline[2, 3] = 1
motif_threeline[3, 2] = 1
motif_threeline_gene = [motif_twoline, motif_threeline]

motif_rectangle = np.zeros((4, 4), dtype=np.int32)
motif_rectangle[0, 1] = 1
motif_rectangle[1, 0] = 1
motif_rectangle[1, 2] = 1
motif_rectangle[2, 1] = 1
motif_rectangle[2, 3] = 1
motif_rectangle[3, 2] = 1
motif_rectangle[0, 3] = 1
motif_rectangle[3, 0] = 1
motif_rectangle_gene = [motif_threeline, motif_rectangle]

motif_semifourclique = np.zeros((4, 4), dtype=np.int32)
motif_semifourclique[0, 1] = 1
motif_semifourclique[1, 0] = 1
motif_semifourclique[0, 2] = 1
motif_semifourclique[2, 0] = 1
motif_semifourclique[0, 3] = 1
motif_semifourclique[3, 0] = 1
motif_semifourclique[1, 2] = 1
motif_semifourclique[2, 1] = 1
motif_semifourclique[2, 3] = 1
motif_semifourclique[3, 2] = 1
motif_semifourclique_gene = [motif_rectangle, motif_semifourclique]

motif_fourclique = np.zeros((4, 4), dtype=np.int32)
motif_fourclique[0, 1] = 1
motif_fourclique[1, 0] = 1
motif_fourclique[0, 2] = 1
motif_fourclique[2, 0] = 1
motif_fourclique[0, 3] = 1
motif_fourclique[3, 0] = 1
motif_fourclique[1, 2] = 1
motif_fourclique[2, 1] = 1
motif_fourclique[1, 3] = 1
motif_fourclique[3, 1] = 1
motif_fourclique[2, 3] = 1
motif_fourclique[3, 2] = 1
motif_fourclique_gene = [motif_semifourclique,motif_fourclique]


# directed motif pool
motif_from = np.zeros((2, 2), dtype=np.int32)
motif_from[0, 1] = 1
motif_from_gene = [None, motif_from]

motif_fromto = np.zeros((3, 3), dtype=np.int32)
motif_fromto[0, 1] = 1
motif_fromto[2, 0] = 1
motif_fromto_gene = [motif_from, motif_fromto]

motif_to = np.zeros((2, 2), dtype=np.int32)
motif_to[1, 0] = 1
motif_to_gene = [None, motif_to]


motif_bi_gene = [motif_to, motif_line]



motif_oneto = np.zeros((3, 3), dtype=np.int32)
motif_oneto[1, 0] = 1
motif_oneto[0, 1] = 1
motif_oneto[1, 2] = 1
motif_oneto_gene = [motif_line, motif_oneto]

motif_twobi_gene = [motif_oneto, motif_twoline]

motif_twoback = np.zeros((3, 3), dtype=np.int32)
motif_twoback[1, 0] = 1
motif_twoback[0, 1] = 1
motif_twoback[1, 2] = 1
motif_twoback[2, 1] = 1
motif_twoback[2, 0] = 1
motif_twoback_gene = [motif_twoline, motif_twoback]

motif_threebi_gene = [motif_twoback, motif_triangle]



fr = open('new_result/grape', 'w') # log of the final result
ft = open('new_result/grape_time', 'w')   # log of the training curve VS time



for indtemp in range(8):
    # Load data
    reload(search)
    data = dataset[indtemp]
    layerind = 2

    if(data.startswith('cite')):
        minor_data = data.split('|')[1]
        data = minor_data
        adj, features, labels, idx_train, idx_val, idx_test = load_data_cite(minor_data)
        flag_direct = False
        parameter = cite_parameter
        population_test = [motif_line_gene, motif_twoline_gene, motif_threeline_gene, motif_triangle_gene, motif_trianglestar_gene]
        select_index = [0,1,2,3,4]

    elif(data.startswith('social')):
        minor_data = data.split('|')[1]
        data = minor_data
        adj, features, labels, idx_train, idx_val, idx_test = load_data_social(minor_data)
        flag_direct = False
        parameter = social_parameter
        population_test = [motif_line_gene, motif_twoline_gene,  motif_triangle_gene, motif_trianglestar_gene, motif_threeline_gene, motif_rectangle_gene, motif_semifourclique_gene, motif_fourclique_gene]
        select_index = [0,1,2,3,7]

    elif(data.startswith('amazon')):
        adj, features, labels, idx_train, idx_val, idx_test = load_data_amazon()
        flag_direct = True
        population_test = [motif_from_gene, motif_to_gene, motif_fromto_gene, motif_bi_gene, motif_oneto_gene, motif_twobi_gene, motif_twoback_gene, motif_threebi_gene]
        select_index = [0,1,2,3,7]
        parameter = social_parameter

    method_dic = {'incsec':1, 'gtrie':2, 'esu':3}
    method_ind = method_dic[method]

    # Some preprocessing
    search_base = np.array(adj.toarray(),dtype=np.int32) # dense array of base adj
    print('Dataset contains:',len(search_base),'nodes,', sum(sum(search_base)), 'edges.')

    node_num = len(search_base)

    search.init_incsearch_model(search_base, flag_direct, flag_acc)
    adj_dic = {}
    init_motif = np.zeros((2, 2), dtype=np.int32)
    # adj = normalize(adj)
    if flag_direct:
        init_motif[1, 0] = 1
        adj_dic[str(list(init_motif.flatten()))] = [sparse_mx_to_torch_sparse_tensor(sp.eye(node_num)), sparse_mx_to_torch_sparse_tensor(adj)] # self-loop
        init_motif[0, 1] = 1
        init_motif[1, 0] = 0
        adj_dic[str(list(init_motif.flatten()))] = [sparse_mx_to_torch_sparse_tensor(sp.eye(node_num)), sparse_mx_to_torch_sparse_tensor(adj.T)]    
    else:
        init_motif[0, 1] = 1
        init_motif[1, 0] = 1
        adj_dic[str(list(init_motif.flatten()))] = [sparse_mx_to_torch_sparse_tensor(sp.eye(node_num)), sparse_mx_to_torch_sparse_tensor(adj)]


    motifadj_test, adj_dic = construct_motif_adj_batch([population_test], adj_dic, search_base, flag_direct, flag_acc, method_ind)
    motifadj_test = motifadj_test[0]
    motifadj_test = [motifadj_test[ind] for ind in select_index]

    for dummy in range(0,2):
        # dummy=0 for original feature, dummy=1 for dummy feature.
        if dummy == 1:
            dummy_feat = np.ones((features.shape[0], features.shape[1]))
            #dummy_feat = np.random.rand(features.shape[0], features.shape[1])
            dummy_feat = torch.FloatTensor(dummy_feat)
            features = dummy_feat

        ft.write(data+ '_'+str(dummy)+ ':\n')
        test_score = []
        for ind in range(test_run):
            id_list = range(node_num)
            random.shuffle(id_list)
            id_len = len(id_list)
            idx_train = id_list[:int(id_len*0.6)]
            idx_val = id_list[int(id_len*0.6):int(id_len*0.8)]
            idx_test = id_list[int(id_len*0.8):]
            idx_train = torch.LongTensor(idx_train)
            idx_val = torch.LongTensor(idx_val)
            idx_test = torch.LongTensor(idx_test)

            acc = train_model(ft, layerind, nepoch, motifadj_test, features, labels, idx_train, idx_val, idx_test, attn, parameter[1], parameter[2], parameter[3], parameter[0])
            test_score.append(acc)

        test_acc_mean, test_acc_std = np.mean(test_score), np.std(test_score)
        print('Final result:', test_acc_mean, test_acc_std)
        fr.write(data + '_'+str(dummy)+ ':' + str(test_acc_mean) + ',' + str(test_acc_std) + '\n')

fr.close()
ft.close()