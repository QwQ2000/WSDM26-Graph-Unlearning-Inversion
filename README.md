## Graph-Unlearning-Inversion

This is the PyTorch implementation for our WSDM'26 paper "**Unlearning Inversion Attack for Graph Neural Networks**". Please find an early access version of our paper on arXiv (https://arxiv.org/abs/2506.00808).

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
  title={Unlearning Inversion Attacks for Graph Neural Networks}, 
  author={Jiahao Zhang and Yilong Wang and Zhiwei Zhang and Xiaorui Liu and Suhang Wang},
  year={2026},
  booktitle={The ACM International Conference on Web Search and Data Mining (WSDM)},
}
```
