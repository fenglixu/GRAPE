import search
import search2
import search3
import numpy as np
import copy
import scipy.sparse as sp
from utils import *

max_node = 5


def motif_initiate(numGenes, flag_dir, flag_acc, adj_dic, search_base, mutate_run, method_ind):
	gene_list = []
	flag = 0
	for i in range(numGenes):
		gene_list.append([None, np.zeros((2, 2), dtype=np.int32)])
		if flag_dir:
			if flag==0:
				gene_list[-1][1][0, 1] = 1
				flag = 1
			else:
				gene_list[-1][1][1, 0] = 1
				flag = 0				
		else:
			gene_list[-1][1][0, 1] = 1
			gene_list[-1][1][1, 0] = 1
	for i in range(mutate_run):
		gene_list = motif_mutate(gene_list, 0.3, 0.2, 0.8, flag_dir)
		_, adj_dic = construct_motif_adj_batch([gene_list], adj_dic, search_base, flag_dir, flag_acc, method_ind)

	return gene_list, adj_dic

def motif_initiate(numGenes, flag_dir, mutate_run):
	gene_list = []
	flag = 0
	for i in range(numGenes):
		gene_list.append([None, np.zeros((2, 2), dtype=np.int32)])
		if flag_dir:
			if flag==0:
				gene_list[-1][1][0, 1] = 1
				flag = 1
			else:
				gene_list[-1][1][1, 0] = 1
				flag = 0				
		else:
			gene_list[-1][1][0, 1] = 1
			gene_list[-1][1][1, 0] = 1
	for i in range(mutate_run):
		gene_list = motif_mutate(gene_list, 0.3, 0.2, 0.8, flag_dir)

	return gene_list

def motif_mutate(origGenes, probMutate, probNodes, probEdges, flag_dir):
	mutated_genes = []
	for gene in origGenes:
		if np.random.rand() < probMutate:
			if np.random.rand() < probNodes:
				if len(gene[1]) == max_node:
					mutated_genes.append(gene)
				else:
					gene_plus = np.zeros((len(gene[1])+1, len(gene[1])+1), dtype=np.int32)
					gene_plus[:-1, :-1] = gene[1]
					attach_node = np.random.randint(len(gene[1]))
					if flag_dir:
						if np.random.randint(2):
							gene_plus[-1, attach_node] = 1
						else:
							gene_plus[attach_node, -1] = 1
					else:
						gene_plus[-1, attach_node] = 1
						gene_plus[attach_node, -1] = 1
					mutated_genes.append((gene[1], gene_plus))

			else:
				zeroList = np.where(gene[1]==0)
				edgeChoices = [(zeroList[0][ind], zeroList[1][ind]) for ind in range(len(zeroList[0])) if zeroList[0][ind]!=zeroList[1][ind]]
				if len(edgeChoices) > 0:
					draw = np.random.choice(range(len(edgeChoices)), replace=False)
					gene_plus = np.zeros(gene[1].shape, dtype=np.int32)
					gene_plus[:,:] = gene[1] 
					if flag_dir:
						gene_plus[edgeChoices[draw][0], edgeChoices[draw][1]] = 1
					else:
						gene_plus[edgeChoices[draw][0], edgeChoices[draw][1]] = 1
						gene_plus[edgeChoices[draw][1], edgeChoices[draw][0]] = 1
					mutated_genes.append((gene[1], gene_plus))
				else:
					mutated_genes.append(gene)		
		else:	
			mutated_genes.append(gene)
	return mutated_genes

def motif_select(candidateList, scoreList, numSurvivals):
	#soft selection
	# survived_candidates = []
	# survived_scores = []
	# prob = (np.array(scoreList)-np.mean(scoreList))/np.std(scoreList)
	# prob = np.exp(prob)/sum(np.exp(prob))
	# reproduced_ind = np.random.choice(range(len(candidateList)), size = numSurvivals, replace=False, p = prob)
	# for ind in reproduced_ind:
	# 	survived_candidates.append(candidateList[ind])
	# 	survived_scores.append(scoreList[ind])

	# hard selection
	score_candidate_pair = zip(scoreList, candidateList)		
	score_candidate_pair = sorted(score_candidate_pair, reverse=True, key=lambda x:x[0])
	survived_candidates = [item[1] for item in score_candidate_pair[:numSurvivals]]
	survived_scores = [item[0] for item in score_candidate_pair[:numSurvivals]]
	return survived_candidates, survived_scores

def motif_reproduce(candidateList, scoreList, numPopulation):
	#soft reproduction
	# if numPopulation > len(candidateList):
	# 	prob = np.array(scoreList)/sum(scoreList)
	# 	reproduced_ind = np.random.choice(range(len(candidateList)), size = numPopulation - len(candidateList), replace=True, p = prob)
	# 	for ind in reproduced_ind:
	# 		candidateList.append(candidateList[ind])
	
	#hard reproduction
	if numPopulation > len(candidateList):
		for i in range(numPopulation - len(candidateList)):	
			candidateList.append(candidateList[i])
	return candidateList

def construct_motif_adj_batch(motifCandidates, adj_dic, baseADJ, flagd, flagacc, method):
	adjList = []
	numNodes = len(baseADJ)
	for candidate in motifCandidates:
		candidate_adj = []
		for gene in candidate:
			if str(list(gene[1].flatten())) in adj_dic:
				candidate_adj.append(adj_dic[str(list(gene[1].flatten()))])
			else:
				resultADJ = [np.eye(numNodes)] # self-loop
				if method == 1:
					print("Candidate motif: " + str(list(np.reshape(gene[1], -1)))+ ", with ancestor: ", str(list(np.reshape(gene[0], -1))))
					search.init_incsearch(gene[0], gene[1])
					print("  Start Inc searching...")
					while(1):
						result_temp = np.array(search.readout(numNodes*numNodes+1))
						resultADJ.append(np.reshape(result_temp[:-1], (numNodes, numNodes)))
						if result_temp[-1]==0:
							break

				elif method == 2:
					print("Candidate motif: " + str(list(np.reshape(gene[1], -1))))
					print("  Start Gtrie searching...")
					if flagd:
						search3.init_gtrie(gene[1], baseADJ)
						result_temp = search3.search(numNodes*numNodes+1)
						resultADJ.append(np.reshape(result_temp[:-1], (numNodes, numNodes)))												
					else:
						search2.init_gtrie(gene[1], baseADJ)
						result_temp = search2.search(numNodes*numNodes+1)
						resultADJ.append(np.reshape(result_temp[:-1], (numNodes, numNodes)))			
				else:
					print("Candidate motif: " + str(list(np.reshape(gene[1], -1))))
					print("  Start ESU searching...")
					if flagd:
						search3.init_esu(gene[1], baseADJ)
						result_temp = search3.search(numNodes*numNodes+1)
						resultADJ.append(np.reshape(result_temp[:-1], (numNodes, numNodes)))												
					else:
						search2.init_esu(gene[1], baseADJ)
						result_temp = search2.search(numNodes*numNodes+1)
						resultADJ.append(np.reshape(result_temp[:-1], (numNodes, numNodes)))

				resultADJ = [sparse_mx_to_torch_sparse_tensor(normalize(sp.csr_matrix(item))) for item in resultADJ]
				# resultADJ = [sparse_mx_to_torch_sparse_tensor(sp.csr_matrix(item)) for item in resultADJ]
				candidate_adj.append(resultADJ)
				adj_dic[str(list(gene[1].flatten()))] = resultADJ
		adjList.append(candidate_adj)
	return adjList, adj_dic

def motif_cross(population, probCross):
	cross_choice = []
	for i in range(len(population)-1):
		for j in range(i, len(population)):
			cross_choice.append((i, j))
	choice_index = range(len(cross_choice))
	draw = np.random.choice(choice_index, int(len(choice_index)*probCross), replace=False)
	result_population = population
	for item in draw:
		gene_ind = np.random.choice(range(3))
		temp = copy.copy(result_population[cross_choice[item][0]][gene_ind])
		result_population[cross_choice[item][0]][gene_ind] = result_population[cross_choice[item][1]][gene_ind]
		result_population[cross_choice[item][1]][gene_ind] = temp
	return result_population

def motif_canonical(motif, flagd):
	motif_input = np.reshape(motif, -1)
	search.canonical(motif_input, flagd)
	return str(motif_input)

def construct_motif_adj(motifCandidate, baseADJ, flagd):

	numNodes = len(baseADJ)
	candidate = motifCandidate
	candidate_adj = []
	
	for gene in candidate:
		resultADJ = [np.eye(numNodes)] # self-loop

		print("Candidate motif: " + str(list(np.reshape(gene[1], -1))))
		if flagd:
			search3.init_esu(gene[1], baseADJ)
			result_temp = search3.search(numNodes*numNodes+1)
			resultADJ.append(np.reshape(result_temp[:-1], (numNodes, numNodes)))												
		else:
			search2.init_esu(gene[1], baseADJ)
			result_temp = search2.search(numNodes*numNodes+1)
			resultADJ.append(np.reshape(result_temp[:-1], (numNodes, numNodes)))


		resultADJ = [sparse_mx_to_torch_sparse_tensor(normalize(sp.csr_matrix(item))) for item in resultADJ]
		candidate_adj.append(resultADJ)

	return candidate_adj