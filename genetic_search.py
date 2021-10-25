from __future__ import division
from __future__ import print_function

import time
import search
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
import h5py
import torch
import torch.nn.functional as F
import torch.optim as optim

gpu_module = sys.argv[1]
data = sys.argv[2]
attn = True

model_name = 'GRAPE'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_module
setproctitle.setproctitle(model_name + '_' + data + '_' + gpu_module)
early_stopping = 50
method = 'incsec' # 1 for incsec, 2 for gtrie, 3 for esu

flag_acc = True # accumulate motif count or not
multi_class = False # can one node belong to multiple class?


# gcn model setting
num_genes = 3
seed = 42
nepoch = 500

# genetic operation setting
population_size = 10
generation = 40
prob_mutation = 0.4
node_mutation = 0.3
edge_mutation = 0.7
num_survival = 7
prob_cross = 0.1

# Set random seed
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

social_parameter_set = [(0.01,0.00005,0.3,32)]
cite_parameter_set = [(0.01,0.00005,0.5,16)]

def evaluate(pred, target, idx, multi_class):
    if multi_class:
        loss_func = torch.nn.BCELoss()
        pred = torch.sigmoid(pred)
        loss = loss_func(pred[idx], target[idx])
        acc = calc_f1(pred[idx], target[idx]) 
    else:
        pred = F.log_softmax(pred, dim=1)
        loss = F.nll_loss(pred[idx], target[idx])
        acc = accuracy(pred[idx], target[idx]).item()
    return loss, acc

def train_model(flog, model_ind, nepoch, candidate_gene, candidate_adj, features, labels, idx_train, idx_val, idx_test, multi_class, attn, lr, weight_decay, dropout, hidden):
    # flatten the ADJ of different motifs and add in a self-loop
    t_test = time.time()
    ngene = len(candidate_adj)
    nrole = [len(item) for item in candidate_adj]
    nclass = labels.shape[1] if multi_class else labels.max().item() + 1
    model = GCN(nfeat=features.shape[1], nhid=hidden, nclass=nclass, nrole=nrole, ngene=ngene, dropout=dropout, attn=attn)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

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
    for epoch in range(nepoch):
        # Construct feed dictionary
        model.train()
        optimizer.zero_grad()
        output = model(features, candidate_adj)        
        loss_train, acc_train = evaluate(output, labels, idx_train, multi_class)

        loss_train.backward()
        optimizer.step()

        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, candidate_adj)

        loss_val, acc_val = evaluate(output, labels, idx_val, multi_class)
        loss_val_list.append(loss_val.item())

        # print('Epoch: {:04d}'.format(epoch+1), 'loss_train: {:.4f}'.format(loss_train.item()), 'acc_train: {:.4f}'.format(acc_train),
        #     'loss_val: {:.4f}'.format(loss_val.item()),'acc_val: {:.4f}'.format(acc_val))
        if epoch > early_stopping and loss_val_list[-1] > np.mean(loss_val_list[-(early_stopping+1):-1]):
            break    
    # Test model
    model.eval()
    output = model(features, candidate_adj)
    loss_test, acc_test = evaluate(output, labels, idx_test, multi_class)
    print("Train accuracy= {:.4f}".format(acc_train), "Val accuracy= {:.4f}".format(acc_val), "Test accuracy= {:.4f}".format(acc_test), "epoch= {:04d}".format(epoch))


    att1 = list(np.squeeze(model.att_weight1.cpu().detach().numpy()))
    att2 = list(np.squeeze(model.att_weight2.cpu().detach().numpy()))

    duration = (time.time() - t_test)

    return acc_test, att1, att2


# Load data
if(data.startswith('ppi')):
    adj, features, labels, idx_train, idx_val, idx_test = load_data_ppi()
    multi_class = True
    flag_direct = False
elif(data.startswith('cite')):
    minor_data = data.split('|')[1]
    data = minor_data
    adj, features, labels, idx_train, idx_val, idx_test = load_data_cite(minor_data)
    multi_class = False
    flag_direct = False
    parameter_set = cite_parameter_set
elif(data.startswith('social')):
    minor_data = data.split('|')[1]
    data = minor_data
    adj, features, labels, idx_train, idx_val, idx_test = load_data_social(minor_data)
    multi_class = False
    flag_direct = False
    parameter_set = social_parameter_set
elif(data.startswith('amazon')):
    adj, features, labels, idx_train, idx_val, idx_test = load_data_amazon()
    multi_class = False
    flag_direct = True
    parameter_set = social_parameter_set
elif(data.startswith('wiki')):
    adj, features, labels, idx_train, idx_val, idx_test = load_data_wiki()
    multi_class = True
    flag_direct = True
else:
    print('Dataset info error!')

method_dic = {'incsec':1, 'gtrie':2, 'esu':3}
method_ind = method_dic[method]

# Some preprocessing
search_base = np.array(adj.toarray(),dtype=np.int32) # dense array of base adj
print('Dataset contains:',len(search_base),'nodes,', sum(sum(search_base)), 'edges.')

node_num = len(search_base)


search.init_incsearch_model(search_base, flag_direct, flag_acc)

adj_dic = {}
init_motif = np.zeros((2, 2), dtype=np.int32)
adj = normalize(adj)
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

population = []
for ind in range(population_size):
    gene_list, adj_dic = motif_initiate(num_genes, flag_direct, flag_acc, adj_dic, search_base, 10)
    population.append(gene_list)


fw = open(method+'_'+data+'_'+str(num_genes),'w')
flog = open(method+'_'+data+'_'+str(num_genes)+'_log','w')


best_score = 0
model_temp = []
duration = 0

for gen in range(generation):   
    t_test = time.time()
    motifadj_pool, adj_dic = construct_motif_adj_batch(population, adj_dic, search_base, flag_direct, flag_acc, method_ind)

    score_list = []
    print("*"*40, "Evaluating the", gen, "generation", "*"*40)
    flog.write("*"*40 + "Evaluating the " + str(gen) + " generation"+ "*"*40+'\n')
    for model_ind in range(len(motifadj_pool)):
        model_perform = []
        att1_list = []
        att2_list = []
        print("-"*5, "Evaluating the candidate:", [list(np.reshape(gene[1], -1)) for gene in population[model_ind]],"-"*5)
        flog.write("-"*5 + "Evaluating the candidate:"+ str([list(np.reshape(gene[1], -1)) for gene in population[model_ind]])+"-"*5 +'\n')

        for _ in range(5):
            #parameter_set.append((lr, wd, do))
            parameter = parameter_set[0]
            
            acc, att1, att2 = train_model(flog, model_ind, nepoch, population[model_ind], motifadj_pool[model_ind], features, labels, idx_train, idx_val, idx_test, multi_class, attn, parameter[0], parameter[1], parameter[2], parameter[3])
            att1_list.append(att1)
            att2_list.append(att2)
            #performance = test_acc if metric_acc else test_mic
            model_perform.append(acc)
        acc, acc_std = np.mean(model_perform), np.std(model_perform)
        att1_list = np.array(att1_list)
        att2_list = np.array(att2_list)
        att1, att1_std = np.mean(att1_list, 0), np.std(att1_list, 0)
        att2, att2_std = np.mean(att2_list, 0), np.std(att2_list, 0)
        
        score_list.append(acc)
        print("Model Performance:"+ str(acc)+"({:.4f})".format(acc_std))
        print("Attention 1:"+ str(att1)+"  "+str(att1_std))
        print("Attention 2:"+ str(att2)+"  "+str(att2_std))
        print("-"*40)
        flog.write("Model Performance:"+ str(acc)+"({:.4f})".format(acc_std)+'\n')
        flog.write("Attention 1:"+ str(att1)+"  "+str(att1_std)+'\n')
        flog.write("Attention 2:"+ str(att2)+"  "+str(att2_std)+'\n')
        flog.write("-"*40+'\n')

            
        if score_list[-1] > best_score:
            best_score = score_list[-1]
            model_temp.append(population[model_ind]) 

    print("Average accuracy of this generation:", np.mean(score_list))
    print("Current best performance:", best_score)
    flog.write("Average accuracy of this generation:"+ str(np.mean(score_list))+'\n')
    flog.write("Current best performance:"+ str(best_score)+'\n')

    survived_population, survived_scores = motif_select(population, score_list, num_survival)
    population = motif_reproduce(survived_population, survived_scores, population_size)
    population = [motif_mutate(candidate, prob_mutation, node_mutation, edge_mutation, flag_direct) for candidate in population]
    population = motif_cross(population, prob_cross)
    duration += (time.time() - t_test)
    fw.write(str(best_score)+'\t'+str(duration) +'\t' + str([list(np.reshape(gene[1], -1)) for gene in model_temp[-1]]) + '\n')
    fw.flush()
fw.close()
flog.close()


fw = open('optimized_motif'+method+'_'+data+'_'+str(num_genes),'w')
for key in adj_dic:
    fw.write(key+'\n')
fw.close()