from abc import ABC, abstractmethod
import os
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from utils import fcns


class Data(ABC):
    def __init__(self, args):
        self.args = args
        self.args.data_dir = fcns.fix_path(self.args.data_dir)
        input_dir = os.path.join(self.args.data_dir, 'training', 'gt')
        in_sample_dir = os.path.join(input_dir,
                                     os.listdir(input_dir)[0])
        self.input_dim = self.load_sample(in_sample_dir)
        self.output_dim = np.append(self.input_dim[:1] * self.args.scale_factor,
                                    self.input_dim[2:])

        self.train_dir = None
        self.x_train_dir = None
        self.y_train_dir = None
        self.valid_dir = None
        self.x_valid_dir = None
        self.y_valid_dir = None
        super().__init__()

    def load_sample(self, path):
        """
        todo: convert any type of data to [x, y, z, ch] then return dimension
        """
        img = np.transpose(tiff.imread(path), (1, 2, 0))

        plt.imshow(img)
        plt.show()
        in_size = list(np.shape(img))
        n_channels = self.args.n_phases * self.args.n_angles
        in_size[0] //= self.args.scale_factor
        in_size[1] //= self.args.scale_factor

        in_size.append(n_channels)
        return in_size
    # def load_psf(self):
    # --------------------------------------------------------------------------------
    #                             Read OTF and PSF
    # --------------------------------------------------------------------------------
    #     pParam = parameters3D()
    #     # 128*128*11 otf and psf numpy array
    #     # 525 loads FairSIM PSF
    #     OTF_Path = {488: './OTF/3D-488-OTF-smallendian.mrc', 560: './OTF/3D-560-OTF-smallendian.mrc',
    #                 525: './OTF/splinePSF_128_128_11.mat'}
    #     psf, _ = cal_psf_3d(OTF_Path[wave_len],
    #                         pParam.Ny, pParam.Nx, pParam.Nz,
    #                         pParam.dky, pParam.dkx, pParam.dkz)
    #
    #     # print(np.shape(psf))
    #
    #     sigma_y, sigma_x, sigma_z = psf_estimator_3d(psf)  # Find the most effective region of OTF
    #     ksize = int(sigma_y * 4)
    #     halfy = pParam.Ny // 2
    #     psf = psf[halfy - ksize:halfy + ksize, halfy - ksize:halfy + ksize, :]
    #     psf = np.reshape(psf, (2 * ksize, 2 * ksize, pParam.Nz, 1, 1)).astype(np.float32)np.reshape(psf, (2 * ksize, 2 * ksize, pParam.Nz, 1, 1)).astype(np.float32)