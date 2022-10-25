import os

## GAN Variants
from CGAN import CGAN

from utils import show_all_variables
from config import parse_args

import tensorflow as tf


def main():
    """main"""

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    # open session
    models = [CGAN]
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # declare instance for GAN

        gan = None
        for model in models:
            if args.gan_type == model.model_name:
                gan = model(sess,
                            epoch=args.epoch,
                            batch_size=args.batch_size,
                            z_dim=args.z_dim,
                            dataset_name=args.dataset,
                            checkpoint_dir=args.checkpoint_dir,
                            result_dir=args.result_dir,
                            log_dir=args.log_dir)
        if gan is None:
            raise Exception("[!] There is no option for " + args.gan_type)

        # build graph
        gan.build_model()

        # show network architecture
        show_all_variables()

        # launch the graph in a session
        gan.train()
        print(" [*] Training finished!")

        # visualize learned generator
        gan.visualize_results(args.epoch-1)
        print(" [*] Validation finished!")


if __name__ == '__main__':
    main()
