import argparse
from distutils.util import strtobool

def get_parser():
    parser = argparse.ArgumentParser(description='argparser')
    # data I/O
    parser.add_argument('--data_dir', type=str, default='./data', help='Location for the dataset')
    parser.add_argument('--data_path', type=str, default='./data/qm9', help='path for loading data and dataconfig')
    parser.add_argument('--data_name', type=str, default='qm9', choices=['qm9', 'zinc250k'], help='dataset name')
    parser.add_argument('--data_file', type=str, default='qm9_relgcn_kekulized_ggnp.npz', help='Name of the dataset')
    parser.add_argument('--save_dir', type=str, default='./results',
                        help='Location for parameter checkpoints and samples')
    parser.add_argument('--epochs', type=int, default=20, help='Num of epochs to run in total')
    parser.add_argument('--device', type=str, default='cuda', help='cpu or cuda')
    # reproducibility
    parser.add_argument('--seed', type=int, default=1, help='Random seed to use')
    parser.add_argument('--num_atoms', type=int, default=9, help='Maximum number of atoms in a molecule')
    parser.add_argument('--num_rels', type=int, default=4, help='Number of bond types')
    parser.add_argument('--num_atom_types', type=int, default=4, help='Types of atoms that can be used in a molecule')
    parser.add_argument('--num_node_masks', type=int, default=9,
                        help='Number of node masks to be used in coupling layers')
    parser.add_argument('--num_channel_masks', type=int, default=4,
                        help='Number of channel masks to be used in coupling layers')
    parser.add_argument('--num_node_coupling', type=int, default=12, help='Number of coupling layers with node masking')
    parser.add_argument('--num_channel_coupling', type=int, default=6,
                        help='Number of coupling layers with channel masking')
    parser.add_argument('--node_mask_size', type=int, default=5, help='Number of cells to be masked in the Node '
                                                                      'coupling layer')
    parser.add_argument('--channel_mask_size', type=int, default=-1, help='Number of cells to be masked in the Channel '
                                                                          'coupling layer')
    parser.add_argument('--apply_batch_norm', type=bool, default=False, help='Whether batch '
                                                                             'normalization should be performed')
    parser.add_argument('--additive_transformations', type=bool, default=True,
                        help='if True, apply only addictive coupling layers; else, apply affine coupling layers')
    # Model configuration.
    parser.add_argument('--temp', type=float, default=0.7, help='the tempearture of mcmc steps')
    parser.add_argument('--use_switch', type=bool, default=False, help='use switch activation for R-GCN')
    parser.add_argument('--num_gcn_layer', type=int, default=3)
    parser.add_argument('--show_loss_step', type=int, default=10, help='show loss every n step/epoch')

    # Training configuration
    parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
    parser.add_argument('--mle_lr', type=float, default=0.001, help='learning rate for MLE')
    parser.add_argument('-lr_decay_factor', default=0.5, type=float, help='learning rate decay factor')
    parser.add_argument('-lr_decay_patience', default=200, type=float, help='learning rate decay patience')
    parser.add_argument('-lr_decay_min', default=0.00001, type=float, help='learning rate decay min')
    parser.add_argument('--dropout', type=float, default=0., help='dropout rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.9, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_epoch', type=int, default=0, help='resume training after this epoch')
    parser.add_argument('--resume', type=bool, default=False, help='resume training')

    # Generation args
    parser.add_argument('--min_atoms', type=int, default=2, help='Minimum number of atoms in a generated molecule')
    parser.add_argument('--num_gen', type=int, default=100, help='num of molecules to generate on each call to train.generate')
    parser.add_argument('--min_gen_epoch', type=int, default=5, help='num of molecules to generate on each call to train.generate')
    parser.add_argument('--max_resample', type=int, default=200, help='the times of resampling each epoch')
    parser.add_argument('--atomic_num_list', type=int, default=[6, 7, 8, 9, 0],
                        help='atomic number list for datasets')
    #parser.add_argument('--post_method', type=str, default='softmax', choices=['softmax', 'soft_gumbel', 'hard_gumbel'],
    #                    help='TODO: postprocessing to convert continuous A and X to discrete ones')
    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'gen'])
    parser.add_argument('--save', action='store_true', default=True, help='Save model.')
    # Directories.
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--model_save_dir', type=str, default='./saved_models')
    parser.add_argument('--sample_dir', type=str, default='./samples')
    parser.add_argument('--result_dir', type=str, default='./results')
    parser.add_argument('--img_dir', type=str, default='./results/qm9_img')
    parser.add_argument('--gen_path', type=str, default='./results/mols', help='output path for generated mol')
    parser.add_argument('--model_name', type=str, default='mle', help='model name, crucial for test and checkpoint initialization [epoch3_1gpu]')
    return parser

parser = get_parser()
args = parser.parse_args()

#Step size.
#parser.add_argument('--log_step', type=int, default=10)
#parser.add_argument('--sample_step', type=int, default=1000)
#parser.add_argument('--model_save_step', type=int, default=1000)
#parser.add_argument('--lr_update_step', type=int, default=1000)