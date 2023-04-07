"""
    author: SPDKH
"""
import os

import numpy as np

from data.data import Data
from utils import fcns
from utils.psf_generator import Parameters3D, cal_psf_3d, psf_estimator_3d


class FairSIM(Data):
    def __init__(self, args):
        Data.__init__(self, args)
        self.data_groups = {'train': 'training',
                            'test': 'testing',
                            'val': 'validation'}

        self.data_types = {'x': 'raw_data', 'y': 'gt'}
        self.args.data_dir = fcns.fix_path(self.args.data_dir)
        input_dir = os.path.join(self.args.data_dir,
                                 self.data_groups['train'],
                                 self.data_types['x'])
        output_dir = os.path.join(self.args.data_dir,
                                  self.data_groups['train'],
                                  self.data_types['y'])
        in_sample_dir = os.path.join(input_dir,
                                     os.listdir(input_dir)[0])
        self.input_dim = self.load_sample(in_sample_dir)
        print('input', self.input_dim)
        out_sample_dir = os.path.join(output_dir,
                                      os.listdir(output_dir)[0])

        self.output_dim = self.load_sample(out_sample_dir)
        print('output', self.output_dim)

        for data_group in self.data_groups.keys():
            self.data_dirs[data_group] = os.path.join(self.args.data_dir,
                                                      self.data_groups[data_group])
            for data_type in self.data_types.keys():
                self.data_dirs[data_type + data_group] = \
                    os.path.join(self.data_dirs[data_group],
                                 self.data_types[data_type])

        print(self.data_dirs)

        self.psf = self.init_psf()

    def init_psf(self):
        # --------------------------------------------------------------------------------
        #                             Read OTF and PSF
        # --------------------------------------------------------------------------------
        pParam = Parameters3D()
        # 128*128*11 otf and psf numpy array
        # 525 loads FairSIM PSF
        OTF_Path = {525: './OTF/splinePSF_128_128_11.mat'}
        psf, _ = cal_psf_3d(OTF_Path[self.args.wave_len],
                            pParam.Ny, pParam.Nx, pParam.Nz,
                            pParam.dky, pParam.dkx, pParam.dkz)

        sigma_y, sigma_x, sigma_z = psf_estimator_3d(psf)  # Find the most effective region of OTF
        ksize = int(sigma_y * 4)
        halfy = pParam.Ny // 2
        psf = psf[halfy - ksize:halfy + ksize, halfy - ksize:halfy + ksize, :]
        return np.reshape(psf,
                          (2 * ksize, 2 * ksize, pParam.Nz, 1)).astype(np.float32)
