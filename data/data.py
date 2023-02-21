from abc import ABC, abstractmethod

import numpy as np
import tifffile as tiff
from matplotlib import pyplot as plt
import matplotlib


class Data(ABC):
    def __init__(self, args):
        self.args = args
        super().__init__()

    def load_sample(self, path, show=0):
        """
        convert any type of data to [x, y, z, ch] then return dimension
        """
        img = np.transpose(tiff.imread(path), (1, 2, 0))
        img_size = list(np.shape(img))
        if img_size[-1]%(self.args.n_phases * self.args.n_angles) == 0:
            img_size[-1] = img_size[-1]//(self.args.n_phases * self.args.n_angles)
            img_size.append(self.args.n_phases * self.args.n_angles)
        else:
            img_size.append(1)

        if show:
            plt.figure()
            plt.imshow(img[:, :, 0])
            plt.show()

        return img_size

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

    def data_loader(self):
        pass
