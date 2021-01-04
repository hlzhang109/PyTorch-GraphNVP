import argparse
import os

from chainer_chemistry import datasets
from chainer_chemistry.datasets import NumpyTupleDataset
from chainer_chemistry.dataset.preprocessors import RSGCNPreprocessor, GGNNPreprocessor

'''
From https://github.com/pfnet-research/graph-nvp/tree/master/data
'''

def parse():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_name', type=str, default='qm9',
                        choices=['qm9', 'zinc250k'],
                        help='dataset to be downloaded')
    parser.add_argument('--data_type', type=str, default='relgcn',
                        choices=['gcn', 'relgcn'],)
    parser.add_argument('--data_dir', type=str, default='./data')
    args = parser.parse_args()
    return args

def to_smiles_txt(smiles, path):
    with open(path, 'w') as f:
        for smile in smiles:
            f.write(smile + '\n')

args = parse()
data_name = args.data_name
data_type = args.data_type
print('args', vars(args))

if data_name == 'qm9':
    max_atoms = 9
elif data_name == 'zinc250k':
    max_atoms = 38
else:
    raise ValueError("[ERROR] Unexpected value data_name={}".format(data_name))

if data_type == 'gcn':
    preprocessor = RSGCNPreprocessor(out_size=max_atoms)
elif data_type == 'relgcn':
    # preprocessor = GGNNPreprocessor(out_size=max_atoms, kekulize=True, return_is_real_node=False)
    preprocessor = GGNNPreprocessor(out_size=max_atoms, kekulize=True)
else:
    raise ValueError("[ERROR] Unexpected value data_type={}".format(data_type))

#data_dir = "."
#os.makedirs(data_dir, exist_ok=True)

if data_name == 'qm9':
    dataset, smiles = datasets.get_qm9(preprocessor, return_smiles=True)
elif data_name == 'zinc250k':
    dataset, smiles = datasets.get_zinc250k(preprocessor, return_smiles=True)
elif data_name == 'tox21':
    dataset, smiles = datasets.get_tox21(preprocessor, return_smiles=True)
else:
    raise ValueError("[ERROR] Unexpected value data_name={}".format(data_name))

to_smiles_txt(smiles, os.path.join(args.data_dir, '{}_kekulized_ggnp.txt'.format(data_name)))
NumpyTupleDataset.save(os.path.join(args.data_dir, '{}_{}_kekulized_ggnp.npz'.format(data_name, data_type)), dataset)
