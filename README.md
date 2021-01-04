# GraphNVP: An Invertible Flow Model for Generating Molecular Graphs

Unofficial implementation of GraphNVP(https://arxiv.org/abs/1905.11600) using PyTorch.

<p float="left" align="middle">
  <img src="https://github.com/hlzhang109/PyTorch-GraphNVP/blob/master/framework.png" width="800"/> 
</p>

## Data Preparation
`python ./data/download_data.py --data_name=qm9`

`python ./data/download_data.py --data_name=zinc250k`

## Train GraphNVP
`./train.sh`

## Citation
```
@misc{madhawa2019graphnvp,
      title={GraphNVP: An Invertible Flow Model for Generating Molecular Graphs}, 
      author={Kaushalya Madhawa and Katushiko Ishiguro and Kosuke Nakago and Motoki Abe},
      year={2019},
      eprint={1905.11600},
      archivePrefix={arXiv},
      primaryClass={stat.ML}
}
```
