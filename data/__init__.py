"""
需要的参数：
1.dataset
2.args

要完成的功能：
把数据集封装到batch

实现步骤：
先把文件夹中的图片构成一个字典
再把这个字典用torch.utils.data.DataLoader封装成batch

torch.utils.data.DataLoader的输入有：
dataset, batch_size=1, shuffle=False, sampler=None,
batch_sampler=None, num_workers=0, collate_fn=default_collate,
pin_memory=False, drop_last=False, timeout=0,
worker_init_fn=None

use opencv (cv2) to read and process images.

read from image files OR from .lmdb for fast IO speed.

can downsample images using matlab bicubic function. However, the speed is a bit slow.
Implemented in util.py. More about matlab bicubic function.

BasicSR的作者的代码可以选择把图片转换成了IMDB加快速度，这里我不做转换

其中dataset是

"""

from data.dataset import Dataset
import torch.utils.data
import logging  #日志

#===================================
#===================================

def create_dataloader(dataset, args, phase):
    '''create dataloader '''
    if phase == 'train':
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.n_threads,
            drop_last=True,logging
            pin_memory=not args.cpu)
    else:
        return torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)


def create_dataset(args, LRpath, HRpath, phase):
    dataset = Dataset(args, LRpath, HRpath, phase)
    # 以后要写上日志文件
    # logger = logging.getLogger('base')
    # logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
    #                                                        args['name']))

    return dataset