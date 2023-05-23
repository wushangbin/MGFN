# Multi-Graph Fusion Networks for Urban Region Embedding (IJCAI-22)
This is the implementation of Multi-Graph Fusion Networks for Urban Region Embedding **(MGFN)** in the following paper: 

Shangbin Wu#, Xu Yan#, Xiaoliang Fan*, Shirui Pan, Shichao Zhu, Chuanpan Zheng, Ming Cheng, Cheng Wang, Multi-Graph Fusion Networks for Urban Region Embedding, International Joint Conference on Artificial Intelligence (IJCAI-22), July 23-29, 2022 Messe Wien, Vienna, Austria.[Acceptance rate=15%]

Multi-Graph Fusion Networks for Urban Region Embedding (MGFN, https://arxiv.org/pdf/2201.09760v1.pdf) was accepted by IJCAI-2022.

## Table of Contents
- [Data](#Data)
- [Requirements](#Requirements)
- [QuickStart](#QuickStart)
- [These Features are Coming Soon](#These-Features-are-Coming-Soon)
- [Citation](#Citation)
- [Contacts](#Contacts)
- [Reference](#Reference)

## Data 
Here we provide the processed data. And the Raw Data can be found in: NYC OpenData: https://opendata.cityofnewyork.us/.  
We followed the settings in [[Zhang et al., 2020]](#R1) that 
apply taxi trip data as human mobility data and take the crime count, check-in count, land usage type as prediction tasks, respectively.  
For New York City zoning dataset, you can contact with Xu Yan(yanxu97@stu.xmu.edu.cn) or Shangbin Wu(shangbin@stu.xmu.edu.cn).

## Requirements 
>Python 3.7.9,   
>pytorch 1.5.1,  
>numpy 1.19.2,  
>pandas 0.25.3,  
>sklearn 0.24.1
>geopandas 0.13.0
>shapely 2.0.1

## QuickStart
run the command below to train the MGFN:
```bash
python mgfn.py
```

## These Features are Coming Soon
The code about...
- Visualization of mobility pattern
- Generalization ability analysis
- Data preprocessing

## Citation
Please cite our paper in your publications if this code helps your research.
```
@article{wu2022multi_graph,
  title={Multi-Graph Fusion Networks for Urban Region Embedding},
  author={Wu, Shangbin and Yan, Xu and Fan, Xiaoliang and Pan, Shirui and Zhu, Shichao and Zheng, Chuanpan and Cheng, Ming and Wang, Cheng},
  journal={arXiv preprint arXiv:2201.09760},
  year={2022}
}
```

## Contacts
Shangbin Wu, shangbin@stu.xmu.edu.cn

Xiaoliang Fan (corresponding author), fanxiaoliang@xmu.edu.cn, https://fanxlxmu.github.io

## Reference  
<div><a name="R1"></a>
[Zhang et al., 2020] Mingyang Zhang, Tong Li, Yong Li,
and Pan Hui. Multi-view joint graph representation learning for urban region embedding. In Christian Bessiere, ed-
itor, Proceedings of the Twenty-Ninth International Joint
Conference on Artificial Intelligence, IJCAI-20, pages
4431–4437. International Joint Conferences on Artificial
Intelligence Organization, 7 2020. Special track on AI for
CompSust and Human well-being.
</div>
