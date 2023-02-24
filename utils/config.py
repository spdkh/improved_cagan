"""
    author: Parisa Daj
    date: May 10, 2022
    parsing and configuration
"""
import random

import argparse

from utils.fcns import check_folder


def check_args(args):
    """
    checking arguments
    """
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)

    # # --result_dir
    # check_folder(args.result_dir)
    #
    # # --result_dir
    # check_folder(args.log_dir)

    # --epoch
    assert args.epoch >= 1, 'number of epochs must be larger than or equal to one'

    # --batch_size
    assert args.batch_size >= 1, 'batch size must be larger than or equal to one'

    # --z_dim
    # assert args.z_dim >= 1, 'dimension of noise vector must be larger than or equal to one'

    return args


def parse_args():
    """
        Define terminal input arguments
    Returns
    -------
    arguments
    """
    desc = "Tensorflow implementation of GAN collections"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--dataset', type=str, default='FixedCell')
    parser.add_argument('--dnn_type', type=str, default='UCAGAN',
                        choices=['CAGAN', 'UCAGAN', 'CGAN', 'SRGAN', 'UGAN'],
                        help='The type of GAN', required=False)
    parser.add_argument("--data_dir", type=str, default="D:\\Data\\FixedCell\\PFA_eGFP\\cropped2d_128",
                        help='The directory of the data')
    parser.add_argument("--norm", type=str, default='max',
                        help='Normalization Method. Current options include: max, min_max, prctile')
    parser.add_argument('--epoch', type=int, default=2000, help='The number of epochs to run')
    parser.add_argument("--sample_interval", type=int, default=10)
    parser.add_argument("--validate_interval", type=int, default=20)
    parser.add_argument("--validate_num", type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=2,
                        help='The size of batch')

    parser.add_argument("--d_start_lr", type=float, default=1e-5)  # 2e-5
    parser.add_argument("--g_start_lr", type=float, default=1e-3)  # 1e-4
    parser.add_argument("--lr_decay_factor", type=float, default=0.5)
    parser.add_argument("--load_weights", type=int, default=0)
    parser.add_argument("--g_opt", type=str, default="adam")
    parser.add_argument("--d_opt", type=str, default="adam")
    parser.add_argument("--train_discriminator_times", type=int, default=1)
    parser.add_argument("--train_generator_times", type=int, default=3)
    parser.add_argument("--unrolling_iter", type=int, default=3)

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')

    parser.add_argument("--weight_wf_loss", type=float, default=0)
    parser.add_argument("--wave_len", type=int, default=525)
    parser.add_argument("--n_ResGroup", type=int, default=3)
    parser.add_argument("--n_RCAB", type=int, default=5)

    parser.add_argument("--n_phases", type=int, default=5)
    parser.add_argument("--n_angles", type=int, default=3)
    random.seed(10)
    return check_args(parser.parse_args())
