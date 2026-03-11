## Graph-Unlearning-Inversion

This is the PyTorch implementation for our WSDM'26 paper "**Unlearning Inversion Attack for Graph Neural Networks**" (https://dl.acm.org/doi/abs/10.1145/3773966.3777929). Please find the full version of our paper on arXiv (https://arxiv.org/abs/2506.00808).

This codebase was adapted from [GIF-torch](https://github.com/wujcan/GIF-torch/). 



## Environment Requirement
The code runs well under python 3.6.10. The required packages are as follows:

- pytorch == 1.9.0+cu111
- torch-geometric == 2.0.3

## Quick Start

```bash
python main.py --dataset_name cora --target_model=GCN --exp Inversion --method GIF --unlearn_ratio 0.05 --attack_method=trend_steal --num_runs=5 --is_gen_unlearn_request=True --is_gen_unlearned_probs=True
```

```bash
python main.py --dataset_name cora --target_model=GCN --exp Inversion --method GIF --unlearn_ratio 0.05 --attack_method=trend_mia --num_runs=5 --is_gen_unlearn_request=True --is_gen_unlearned_probs=True
```

## BibTeX
If you find our work useful in your research, please cite the following in your manuscript:


```
@inproceedings{zhang2026unlearning,
  title={Unlearning inversion attacks for graph neural networks},
  author={Zhang, Jiahao and Wang, Yilong and Zhang, Zhiwei and Liu, Xiaorui and Wang, Suhang},
  booktitle={Proceedings of the Nineteenth ACM International Conference on Web Search and Data Mining},
  pages={934--945},
  year={2026}
}
```
