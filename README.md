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

### License
```
Copyright (c) 2020-present NAVER Corp. and Korea University 
```

