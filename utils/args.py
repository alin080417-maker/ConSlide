# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from argparse import ArgumentParser
from datasets import NAMES as DATASET_NAMES
from models import get_all_models


def add_experiment_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the models.
    :param parser: the parser instance
    """
    parser.add_argument('--dataset', type=str, required=True,
                        choices=DATASET_NAMES,
                        help='Which dataset to perform experiments on.')
    parser.add_argument('--exp_desc', type=str, required=True,
                        help='Experiment description.')
    parser.add_argument('--model', type=str, required=True,
                        help='Model name.', choices=get_all_models())

    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate.')

    parser.add_argument('--optim_wd', type=float, default=0.,
                        help='optimizer weight decay.')
    parser.add_argument('--optim_mom', type=float, default=0.,
                        help='optimizer momentum.')
    parser.add_argument('--optim_nesterov', type=int, default=0,
                        help='optimizer nesterov momentum.')    

    parser.add_argument('--n_epochs', type=int, default=50,
                        help='Batch size.')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size.')

    parser.add_argument('--wsi_data_root', type=str, default='../Dataset',
                        help='Root directory containing TCGA cohort folders.')
    parser.add_argument('--wsi_split_root', type=str, default='../HIT/10fold_splits',
                        help='Root directory containing WSI split folders.')
    parser.add_argument('--wsi_feature_subdir', type=str,
                        default='patch_4096/convnexts_l0l1_512_4096',
                        help='Feature directory inside each TCGA cohort folder.')
    parser.add_argument('--wsi_feature_format', type=str, default='h5',
                        choices=['h5', 'pt'],
                        help='On-disk feature format for WSI bags.')
    parser.add_argument('--wsi_backbone', type=str, default='hit',
                        choices=['hit', 'abmil'],
                        help='Backbone architecture for WSI bag classification.')
    parser.add_argument('--wsi_feature_dim', type=int, default=768,
                        help='Feature vector dimension consumed by the WSI backbone.')
    parser.add_argument('--wsi_pt_feature_layout', type=str, default='flat',
                        choices=['flat', 'hit'],
                        help='Tensor layout adaptation for .pt feature bags.')
    parser.add_argument('--wsi_h5_feature_key', type=str, default='features',
                        help='Primary feature key used when reading HDF5 bags.')
    parser.add_argument('--wsi_h5_feature2_key', type=str, default='features2',
                        help='Secondary feature key used when reading HDF5 bags.')
    parser.add_argument('--wsi_h5_coords_key', type=str, default='coords',
                        help='Coordinate key used when reading HDF5 bags.')
    parser.add_argument('--wsi_missing_feature2', type=str, default='error',
                        choices=['error', 'copy', 'zeros'],
                        help='Fallback when the secondary feature key is missing.')
    parser.add_argument('--wsi_num_workers', type=int, default=4,
                        help='Number of DataLoader workers for WSI datasets.')

def add_management_args(parser: ArgumentParser) -> None:
    parser.add_argument('--seed', type=int, default=None,
                        help='The random seed.')
    parser.add_argument('--notes', type=str, default=None,
                        help='Notes for this run.')

    parser.add_argument('--non_verbose', action='store_true')
    parser.add_argument('--csv_log', action='store_true',
                        help='Enable csv logging', default=True)
    parser.add_argument('--tensorboard', action='store_true',
                        help='Enable tensorboard logging')
    parser.add_argument('--validation', action='store_true',
                        help='Test on the validation set')


def add_rehearsal_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the rehearsal-based methods
    :param parser: the parser instance
    """
    parser.add_argument('--buffer_size', type=int, required=True, default=100,
                        help='The size of the memory buffer.')
    parser.add_argument('--minibatch_size', type=int,
                        help='The batch size of the memory buffer.')
