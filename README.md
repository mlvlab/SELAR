# Self-supervised Auxiliary Learning with Meta-paths for Heterogeneous Graphs
This repository is the implementation of [SELAR](https://arxiv.org/abs/2007.08294).

> Dasol Hwang<sup>* </sup>, Jinyoung Park<sup>* </sup>, Sunyoung Kwon, Kyung-min Kim, Jung-Woo Ha, Hyunwoo J. Kim, Self-supervised Auxiliary Learning with Meta-paths for Heterogeneous Graphs, In Advanced in Neural Information Processing Systems (NeurIPS 2020).

![](https://github.com/mlvlab/SELAR/blob/main/Figure_Main.png)

### Data Preprocessing
We used datasets from [KGNN-LS](https://github.com/hwwang55/KGNN-LS) and [RippleNet](https://github.com/hwwang55/RippleNet) for link prediction.
Download meta-paths label (`meta_labels/`) from this [link](https://drive.google.com/drive/folders/1sssNbczHD2usnLTk6KoukfO5OipPMKpW?usp=sharing).
- `data/music/`
  - `ratings_final.npy` : preprocessed rating file released by KGNN-LS;
  - `kg_final.npy` : knowledge graph file;
    - `meta_labels/`
      - `pos_meta{}_{}.pickle` : meta-path positive label for auxiliary task
      - `neg_meta{}_{}.pickle` : meta-path negative label for auxiliary task

- `data/book/`
  - `ratings_final.npy` : preprocessed rating file released by RippleNet;
  - `kg_final.npy` : knowledge graph file;
    - `meta_labels/`
      - `pos_meta{}_{}.pickle` : meta-path positive label for auxiliary task
      - `neg_meta{}_{}.pickle` : meta-path negative label for auxiliary task
  
### Required packages
A list of dependencies will need to be installed in order to run the code. We provide the dependency yaml file (env.yml)
```
$ conda env create -f env.yml
```

### Running the code
```
# check optional arguments [-h]
$ python main_music.py
$ python main_book.py
```
### Overview of the results of link prediction
#### Last-FM (Music)
Base GNNs | Vanilla | w/o MP | w/ MP | **SELAR** | **SELAR+Hint** 
-- | -- | -- | -- | -- | -- 
GCN | 0.7963 | 0.7899 | 0.8235 | **0.8296** | 0.8121 
GAT | 0.8115 | 0.8115 | 0.8263 | 0.8294 | **0.8302**
GIN | 0.8199 | 0.8217 | 0.8242 | **0.8361** | 0.8350 
SGC | 0.7703 | 0.7766 | 0.7718 | 0.7827 | **0.7975** 
GTN | 0.7836 | 0.7744 | 0.7865 | 0.7988 | **0.8067** 

#### Book-Crossing (Book)
Base GNNs | Vanilla | w/o MP | w/ MP | **SELAR** | **SELAR+Hint** 
-- | -- | -- | -- | -- | -- 
GCN | 0.7039 | 0.7031 | 0.7110 | 0.7182 | **0.7208**
GAT | 0.6891 | 0.6968 | 0.7075 | 0.7345 | **0.7360**
GIN | 0.6979 | 0.7210 | 0.7338 | **0.7526** | 0.7513 
SGC | 0.6860 | 0.6808 | 0.6792 | 0.6902 | **0.6926** 
GTN | 0.6732 | 0.6758 | 0.6724 | **0.6858** | 0.6850

## Citation
```
@inproceedings{NEURIPS2020_74de5f91,
 author = {Hwang, Dasol and Park, Jinyoung and Kwon, Sunyoung and Kim, KyungMin and Ha, Jung-Woo and Kim, Hyunwoo J},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {H. Larochelle and M. Ranzato and R. Hadsell and M. F. Balcan and H. Lin},
 pages = {10294--10305},
 publisher = {Curran Associates, Inc.},
 title = {Self-supervised Auxiliary Learning with Meta-paths for Heterogeneous Graphs},
 url = {https://proceedings.neurips.cc/paper/2020/file/74de5f915765ea59816e770a8e686f38-Paper.pdf},
 volume = {33},
 year = {2020}
}
```

### License
```
Copyright (c) 2020-present NAVER Corp. and Korea University 
```

