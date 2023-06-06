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

"""Contains the code for prediction using trained models. 
    author: SPDKH
    date: Nov 2, 2023
    updated: Mazhar on June 05, 2023
    
Example usage:
    conda activate tf_gpu
    python -m predict_3d --dnn_type RCAN --data_dir D:\\Data\\FixedCell\\PFA_eGFP\\cropped2d_128 --n_ResGroup 2 --n_RCAB 10 --n_channel 16 --unrolling_iter 0 --model_weights "C:\\Users\\unrolled_caGAN\\Desktop\\mazhar_Unrolled caGAN project\\checkpoint\\FixedCell_RCAN_17-04-2023_time0257weights_best.h5"

"""
import os

import numpy as np
import tensorflow as tf
import tifffile as tiff

from models.CAGAN import CAGAN
from models.CGAN import CGAN
from models.DNN import DNN
from models.RCAN import RCAN
from models.SRGAN import SRGAN
from models.UCAGAN import UCAGAN
from models.URCAN import URCAN
from utils.config import parse_args
from utils.fcns import prctile_norm


def main():
    """
        Main Predicting Function
    """
    # parse arguments
    args = parse_args()
    args.batch_size = 1
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    model_fns = {'CAGAN': CAGAN,
                 'SRGAN': SRGAN,
                 'CGAN': CGAN,
                 'DNN': DNN,
                 'UCAGAN': UCAGAN,
                 'RCAN': RCAN,
                 'URCAN': URCAN}

    # declare instance for GAN
    dnn = model_fns[args.dnn_type](args)

    # if 'gan' in args.dnn_type:
    #     dnn.model = dnn.gen

    # print(np.shape(dnn.model.))
    dnn.model.built = True

    # data = Data(args)
    # dnn = tf.keras.models.load_model(args.model_weights)
    dnn.model.load_weights(args.model_weights, by_name = True, skip_mismatch = True)
    dnn.build_model()
    output_name = 'output_' + args.dnn_type + '-'
    output_dir = args.result_dir + '\\' + output_name
    output_dir = output_dir + 'SIM'

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    input_d, _, _ = \
        dnn.data.data_loader('test',
                              dnn.batch_iterator('test'),
                              100,
                              dnn.scale_factor)

    outputs = dnn.gen.predict(input_d)

    # pr = prctile_norm(np.squeeze(outputs))

    for im_count, output_img in enumerate(outputs):
        outName = os.path.join(output_dir, f"b-{dnn.batch_id['test']}_i-{im_count}.tiff")
        print(outName)
        # pr = np.transpose(65535 * pr, (2, 0, 1)).astype('uint16')
        tiff.imwrite(outName, output_img, dtype='uint16')


if __name__ == '__main__':
    main()
