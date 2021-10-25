## Readme

This folder contains an implementation of GRAPE model. It is built and tested on pytorch platform with python 2.7.



### 1.Dependencies (other versions may also work):

- python == 2.7

- pytorch == 1.4

- numpy == 1.17.2

- scipy == 1.6.3

- sklearn ==0.24.1

- networkx == 2.5

- h5py == 2.9.0

- GPUtil ==1.4.0

- setproctitle == 1.1.10

  

### 2.How to Run

Evaluating GRAPE model：

```
python grape_model.py [gpu_module_id] 
```

Running genetic search：

```
python genetic_search.py [gpu_module_id] [dataset_id] 
```