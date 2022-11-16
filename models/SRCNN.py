"""
    author: SPDKH
    todo: complete
"""
from tensorflow.keras.layers import Conv3D, UpSampling3D, LeakyReLU, Lambda, ReLU

from models import DNN


class SRCNN(DNN):
    def __init__(self, args):
        DNN.__init__(self, args)

    def arch(self, filters=[9, 1, 5], coeffs=[64, 32]):
        """
        important: deeper is not better here
        The best performance is as in the examples but takes more time

        Parameters
        ----------
        filters:
            a list of filters (int)
            example(9, 5, 5)

        coeffs:
            a list of coefficient values (1 item less than filters)
            example(128, 64)

        Returns
        -------
        the output tensorflow layer from the architecture

        todo: start with imagenet weights
        todo: check in the beginning if the lengths are not correct, drop from coeffs
        """
        conv = self.input
        for i, ni in enumerate(coeffs):
            conv = Conv3D(ni,
                          kernel_size=filters[i],
                          padding='same')(conv)
            conv = ReLU()(conv)

        conv = Conv3D(1, kernel_size=filters[-1], padding='same')(conv)
        output = UpSampling3D(size=(self.args.scale_factor,
                                    self.args.scale_factor,
                                    1))(conv)
        display()
        return output
