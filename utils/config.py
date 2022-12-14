"""
    author: Parisa Daj
    date: May 10, 2022
    parsing and configuration
    todo: revise
"""
import random

import argparse

from fcns import check_folder


def check_args(args):
    """
    checking arguments
    """
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)

    # --result_dir
    check_folder(args.result_dir)

    # --result_dir
    check_folder(args.log_dir)

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

    parser.add_argument('--dnn_type', type=str, default='CGAN',
                        choices=['caGAN', 'CGAN', 'SRGAN', 'UGAN'],
                        help='The type of GAN', required=False)
    parser.add_argument("--data_dir", type=str, default="D:/Data/datasets_luhong/cropped128",
                        help='The directory of the data')
    # parser.add_argument('--patch_dim', nargs='+', default=[128, 128, 1, 15],
    #                     help='Dimension of the patches followed by number of channels')
    parser.add_argument("--scale_factor", type=int, default=2)
    parser.add_argument("--norm_flag", type=int, default=1)
    parser.add_argument('--epoch', type=int, default=20, help='The number of epochs to run')
    parser.add_argument("--sample_interval", type=int, default=2)
    parser.add_argument("--validate_interval", type=int, default=5)
    parser.add_argument("--validate_num", type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=2,
                        help='The size of batch')

    parser.add_argument("--d_start_lr", type=float, default=1e-6)  # 2e-5
    parser.add_argument("--g_start_lr", type=float, default=1e-4)  # 1e-4
    parser.add_argument("--lr_decay_factor", type=float, default=0.5)
    parser.add_argument("--load_weights", type=int, default=0)
    parser.add_argument("--g_opt", type=str, default="adam")
    parser.add_argument("--d_opt", type=str, default="adam")
    parser.add_argument("--train_discriminator_times", type=int, default=1)
    parser.add_argument("--train_generator_times", type=int, default=3)

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')

    parser.add_argument("--weight_wf_loss", type=float, default=0)
    parser.add_argument("--wave_len", type=int, default=525)
    parser.add_argument("--n_ResGroup", type=int, default=2)
    parser.add_argument("--n_RCAB", type=int, default=3)

    random.seed(10)
    return check_args(parser.parse_args())
