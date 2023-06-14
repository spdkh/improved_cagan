# Copyright 2023 The Improved caGAN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
    author: SPDKH
    date: Nov 2, 2023
"""

import sys
import tensorflow as tf

# ________________ architecture Variants
from models.SRGAN import SRGAN
from models.CAGAN import CAGAN
from models.UCAGAN import UCAGAN
from models.CGAN import CGAN
from models.DNN import DNN
from models.RCAN import RCAN
from models.URCAN import URCAN

from utils.config import parse_args

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


def main():
    """
        Main Training Function
    """

    # parse arguments
    args = parse_args()
    if args is None:
        sys.exit()
    tf.random.set_seed(args.seed)

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    # gpu_options = tf.compat.v1.GPUOptions(TF_GPU_ALLOCATOR=cuda_malloc_async, per_process_gpu_memory_fraction=args.gpu_memory_fraction)
    # tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))


    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    # open session
    model_fns = {'CAGAN': CAGAN,
                 'SRGAN': SRGAN,
                 'CGAN': CGAN,
                 'DNN': DNN,
                 'UCAGAN': UCAGAN,
                 'RCAN': RCAN,
                 'URCAN': URCAN}

    # declare instance for GAN
    dnn = model_fns[args.dnn_type](args)

    # build graph
    dnn.build_model()

    # show network architecture
    # show_all_variables()
    #
    # launch the graph in a session
    dnn.train()
    print(" [*] Training finished!")


if __name__ == '__main__':
    main()
