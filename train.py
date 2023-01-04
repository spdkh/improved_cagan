"""
    author: SPDKH
    date: Nov 2, 2022
"""


import sys

import tensorflow as tf
from tensorflow.keras.models import Model

# ________________ architecture Variants
from models.CGAN import CGAN
from models.CAGAN import CAGAN
from models.DNN import DNN

from utils.fcns import show_all_variables
from utils.config import parse_args
from utils.data_loader import data_loader


def main():
    """
        Main Training Function
    """

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # parse arguments
    args = parse_args()
    if args is None:
        sys.exit()

    # open session
    model_fns = {'CAGAN': CAGAN,
                 'CGAN': CGAN,
                 'DNN': DNN}


    # data = data_loader()
    # x_train, y_train = \
    # data_loader_multi_channel_3d(images_path, train_images_path, train_gt_path,
    #                              patch_y, patch_x, patch_z, batch_size_d,
    #                              norm_flag=norm_flag)
    # declare instance for GAN
    dnn = model_fns[args.dnn_type](args)

    # build graph
    dnn.build_model()

    # # show network architecture
    # show_all_variables()
    #
    # launch the graph in a session
    dnn.train()
    print(" [*] Training finished!")
    #
    # # visualize learned generator
    # dnn.visualize_results(args.epoch-1)
    # print(" [*] Validation finished!")


if __name__ == '__main__':
    main()
