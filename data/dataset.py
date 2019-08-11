"""
pytorch/torch/utils/data/path.py
读取图片，构建dataset，返回字典
"""
import os.path
import random
import numpy as np
import cv2
import torch
import torch.utils.data as data
import data.util as util
import data.path as path

class Dataset(data.Dataset):
    r"""
        pytorch文档中torch.utils.data.Dataset
        An abstract class representing a Dataset.
        All other datasets should subclass it. All subclasses should override
        ``__len__``, that provides the size of the dataset, and ``__getitem__``,
        supporting integer indexing in range from 0 to len(self) exclusive.
        """
    def __init__(self, args, LRpath, HRpath, phase):
        # 举例：Dataset(args, path.paths_train_LR, path.paths_train_HR)
        self.args = args
        self.phase = phase
        self.LRpath = LRpath
        self.HRpath = HRpath
        r"""
        DIV2K数据集比较特殊，HR可以被2/3/4整除，因此不需要设置倍数
        BasicSR使用data.util里的_get_paths_from_images(path)读取图片返回图片list
        """
        # 得到具体图片路径
        self.paths_LR = util.get_image_paths(self.LRpath)
        self.paths_HR = util.get_image_paths(self.HRpath)

        # 判断文件夹是否为空
        assert self.paths_HR, 'Error: HR path is empty.'
        assert self.paths_LR, 'Error: LR path is empty.'

        # 判断HR与LR图片数目是否相符
        if self.paths_LR and self.paths_HR:
            assert (len(self.paths_HR) == len(self.paths_LR)) , \
                'HR and LR datasets have different number of images - {}, {}.'.format( \
                    len(self.paths_HR), len(self.paths_LR))

        # 是否对训练的patch进行缩放，不缩放就只写个1
        self.random_scale_list = [1]


    def __getitem__(self, index):
        scale = self.args.scale
        HR_size = self.args.patch_size
        # HR_path[0]就是路径文件夹里的第一张图，HR_path[1]就是路径文件夹里的第二张图
        LR_path = self.paths_LR[index]
        img_LR = util.read_img(LR_path)
        H_l, W_l, C_l = img_LR.shape
        # 非test需要HR
        if not self.args.test_only:
            HR_path = self.paths_HR[index]
            img_HR = util.read_img(HR_path)
            # 如果没有预处理，需要crop
            # img_HR = util.modcrop(img_HR, scale)

            # 根据图片和设置通道数转换通道，扩充通道
            img_HR = util.channel_convert(img_HR.shape[2], self.args.n_colors, [img_HR])[0]
            img_LR = util.channel_convert(img_LR.shape[2], self.args.n_colors, [img_LR])[0]
            # 裁剪转换后的长宽高
            H_h, W_h, C_h = img_HR.shape
            # 缩放函数
            def _mod(n, random_scale, scale, thres):
                rlt = int(n * random_scale)
                rlt = (rlt // scale) * scale
                return thres if rlt < thres else rlt
            # 是否缩放 这里先不缩放了，因为HR与LR在文件中都有对应
            # random_scale = random.choice(self.random_scale_list)
            # 缩放图片，缩放结果不能小于patch szie
            # if self.phase == 'train':
            #     H_h = _mod(H_h, random_scale, scale, HR_size)
            #     W_h = _mod(W_h, random_scale, scale, HR_size)
                # 裁剪patch
            if self.phase == 'train':
                LR_size = HR_size // scale
                rnd_h = random.randint(0, max(0, H_l - LR_size))
                rnd_w = random.randint(0, max(0, W_l - LR_size))
                rnd_h_HR, rnd_w_HR = int(rnd_h * scale), int(rnd_w * scale)
                img_LR = img_LR[rnd_h:rnd_h + LR_size, rnd_w:rnd_w + LR_size, :]
                img_HR = img_HR[rnd_h_HR:rnd_h_HR + HR_size, rnd_w_HR:rnd_w_HR + HR_size, :]
                # augmentation - flip, rotate
                img_LR, img_HR = util.augment([img_LR, img_HR], self.args.no_augment, C_h)



        else:
            img_LR = util.channel_convert(img_LR.shape[2], self.args.n_colors, [img_LR])[0]
            img_HR = img_LR

        # BGR to RGB, HWC to CHW, numpy to tensor 转换，单通道同样可以转换
        if img_HR.shape[2] == 3:
            img_HR = img_HR[:, :, [2, 1, 0]]
            img_LR = img_LR[:, :, [2, 1, 0]]
        #np.ascontiguousarray 返回和传入的数组类似的内存中连续的数组
        img_HR = torch.from_numpy(np.ascontiguousarray(np.transpose(img_HR, (2, 0, 1)))).float()
        img_LR = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LR, (2, 0, 1)))).float()

        if HR_path is None:
            HR_path = LR_path
        return {'LR': img_LR, 'HR': img_HR, 'LR_path': LR_path, 'HR_path': HR_path}

    def __len__(self):
        return len(self.paths_HR)


