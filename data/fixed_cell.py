"""
    author: SPDKH
    todo: complete
"""
import os
from data.data import Data


class FixedCell(Data):
    def __init__(self, args):
        Data.__init__(self, args)
        train_dir = os.path.join(self.args.data_dir, 'training')
        self.x_train_dir = os.path.join(train_dir, 'raw_data')
        # train_wf_path = data_dir + '/training_wf/'
        self.y_train_dir = os.path.join(train_dir, 'gt')

        valid_dir = os.path.join(self.args.data_dir, 'validation')
        self.x_valid_dir = os.path.join(valid_dir, 'raw_data')
        self.y_valid_dir = os.path.join(valid_dir, 'gt')

        test_dir = os.path.join(self.args.data_dir, 'testing')
        self.x_valid_dir = os.path.join(test_dir, 'raw_data')
        self.y_valid_dir = os.path.join(test_dir, 'gt')
