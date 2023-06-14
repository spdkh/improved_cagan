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
import datetime

from models.RCAN import RCAN

from utils.config import parse_args
from utils.fcns import *


def main():
    """
        Main Predicting Function
    """
    # parse arguments
    args = parse_args()
    args.batch_size = 1
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # declare instance for GAN
    dnn = RCAN(args)
    dnn.build_model()
    # dnn.model.built = True

    dnn.model.load_weights(args.model_weights, by_name=True, skip_mismatch = True)
    dnn.build_model()

    output_folder = '_'.join([args.dataset,
                           datetime.datetime.now().strftime("%d-%m-%Y_time%H%M")])
    results_dir = '/'.join(args.model_weights.split('/')[:-1])
    output_dir = os.path.join(results_dir, 'test_results')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    output_dir = os.path.join(output_dir, output_folder)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # -------------------------------------------------------------------
    #                       about Tensor Board
    # -------------------------------------------------------------------

    val_names = ['val_MSE',
                 'val_SSIM',
                 'val_PSNR',
                 'val_NRMSE',
                 'val_UQI']
    mses, nrmses, psnrs, ssims, uqis = [], [], [], [], []
    imgs, imgs_gt, _ = \
        dnn.data.data_loader('test', 0,
                              len(os.listdir(dnn.data.data_dirs['xtest'])),
                              dnn.scale_factor)
    outputs = dnn.model.predict(imgs)
    patch_y, patch_x, patch_z, _ = dnn.data.input_dim
    for im_count, (img, output, img_gt) in enumerate(zip(imgs, outputs, imgs_gt)):
        output = prctile_norm(output)
        img_gt = prctile_norm(img_gt)
        img = prctile_norm(img)
        output_proj = np.max(output, 2)
        gt_proj = np.max(np.reshape(img_gt,
                                    dnn.data.output_dim[:-1]),
                         2)
        metrics = \
            img_comp(gt_proj,
                          output_proj,
                          mses,
                          nrmses,
                          psnrs,
                          ssims,
                          uqis)
        outName = os.path.join(output_dir, f"b-{dnn.batch_id['test']}_{im_count}.jpg")

        writer = tf.summary.create_file_writer(output_dir)

        for val_name, metric in zip(val_names, metrics):
            dnn.write_log(writer, val_name, str(metric), mode=0)

        plt.figure(figsize=(22, 6))
        # figures equal to the number of z patches in columns
        for j in range(patch_z):
            output_results = {'Raw Input': img[:, :, j, 0],
                              'Super Resolution Output': output[:, :, j, 0],
                              'Ground Truth': img_gt[:, :, j, 0]}

            plt.title('Z = ' + str(j))
            for i, (label, img) in enumerate(output_results.items()):
                # first row: input image average of angles and phases
                # second row: resulting output
                # third row: ground truth
                plt.subplot(3, patch_z, j + patch_z * i + 1)
                plt.ylabel(label)
                plt.imshow(img, cmap=plt.get_cmap('hot'))

                plt.gca().axes.yaxis.set_ticklabels([])
                plt.gca().axes.xaxis.set_ticklabels([])
                plt.gca().axes.yaxis.set_ticks([])
                plt.gca().axes.xaxis.set_ticks([])
                plt.colorbar()

        plt.savefig(outName)  # Save sample results
        plt.close("all")  # Close figures to avoid memory leak

if __name__ == '__main__':
    main()
