"""
pytorch/torch/utils/data/path.py
读取图片，构建dataset，返回字典
"""
import os.path
import torch.utils.data as data


class PATH(data.Dataset):
    r"""
        pytorch文档中torch.utils.data.Dataset
        An abstract class representing a Dataset.
        All other datasets should subclass it. All subclasses should override
        ``__len__``, that provides the size of the dataset, and ``__getitem__``,
        supporting integer indexing in range from 0 to len(self) exclusive.
        """
    def __init__(self, args):
        self.args = args
        r"""
        DIV2K数据集比较特殊，HR可以被2/3/4整除，因此不需要设置倍数
        BasicSR使用data.util里的_get_paths_from_images(path)读取图片返回图片list
        """
        #设置路径


        # train LR(args.data_train/args.data_train+'_train_LR'/'x'+args.scale
        #       HR(args.data_train/args.data_train+'_train_HR')
        self.paths_train = os.path.join(args.dir_data, args.data_train)
        self.paths_train_HR = os.path.join(self.paths_train, '{}_train_HR'.format(args.data_train))
        self.paths_train_LR_ = os.path.join(self.paths_train, '{}_train_LR'.format(args.data_train))
        self.paths_train_LR = os.path.join(self.paths_train_LR_, 'x{}'.format(args.scale))

        # validation LR/HR
        # validation LR(args.args.data_validation/LR/'x'+args.scale
        #            HR(args.args.data_validation/HR)
        self.paths_validation = os.path.join(args.dir_data, args.data_validation)
        self.paths_validation_HR = os.path.join(self.paths_validation, 'HR')
        self.paths_validation_LR_ = os.path.join(self.paths_validation, 'LR')
        self.paths_validation_LR = os.path.join(self.paths_validation_LR_, 'x{}'.format(args.scale))








        # test LR/HR
        # test LR(args.args.data_test/LR/'x'+args.scale
        #      HR(args.args.test/HR)
        self.paths_test = os.path.join(args.dir_data, args.data_test)
        self.paths_test_HR = os.path.join(self.paths_test, 'HR')
        self.paths_test_LR_ = os.path.join(self.paths_test, 'LR')
        self.paths_test_LR = os.path.join(self.paths_test_LR_, 'x{}'.format(args.scale))





