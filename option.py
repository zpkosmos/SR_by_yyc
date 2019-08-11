# 一步一步抄的
r"""
    需要哪些参数？
    0.debug，template（用于调用template.py中默认保存的一些参数设置）
    1.硬件设置，cuda，gpu，thread（在dataloader中使用），
      seed（每次获取随机数时，设置seed，发现每次获得的都一样（前提是函数调用方式一样））
    2.数据集设置，数据文件夹，测试集，训练集，数据及划分（如果测试集和训练集来自同一个数据集），
      文件扩展名，放大倍数，patchsize，颜色范围，颜色通道数，是否使用数据增强
      其他：chop
    3.网络设置，使用哪个网络，激活函数，预训练（pre_train,extend），残差模块数，模块通道数，
      残差模块权重，是否减去数据集平均值，是否使用空洞卷积，参数精度
    4.专属网络参数设置
    5.训练规范，训练重启，每训练多少次进行测试，epochs，batchsize，是否self_ensemble
      其他：split_batch，test_only，gan_k
    6.优化设置，学习率，多少epoch下降，下降多少，optimizer选项，adam参数（batas和epsilon）
      其他：SGD参数，weight_decay，gradient clipping
    7.loss设置，损失函数的选择和权重，是否跳过误差太大的batch
    8.参数文件设置，保存，读取，恢复，是否保存超分辨结果
      save_models，print_every，save_gt


"""

import argparse


parser = argparse.ArgumentParser(description='SRyyc')

# 0
# --debug没加

r"""

parser.add_argument('--debug', action='store_true',
                    help='Enables debug mode')

"""

# 1.硬件设置Hardware specifications
# --chop --data_rage没加
parser.add_argument('--n_threads', type=int, default=6,
                    help='读入数据的进程数，越多越好，但具体设置根据不明，number of threads for data loading')
parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')

# 2.数据集设置Data specifications
parser.add_argument('--dir_data', type=str, default='DATASET',
                    help='dataset directory')
parser.add_argument('--dir_demo', type=str, default='../test',
                    help='demo image directory')
parser.add_argument('--data_train', type=str, default='DIV2K',
                    help='train dataset name')
parser.add_argument('--data_validation', type=str, default='Set5',
                    help='validation dataset name')
parser.add_argument('--data_test', type=str, default='Set5',
                    help='test dataset name')
parser.add_argument('--ext', type=str, default='sep',
                    help='dataset file extension')
parser.add_argument('--scale', type=int, default='4',
                    help='super resolution scale')
parser.add_argument('--patch_size', type=int, default=192,
                    help='output patch size')
parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')

parser.add_argument('--no_augment', action='store_true',
                    help='do not use data augmentation')

r"""
parser.add_argument('--chop', action='store_true',
                    help='enable memory-efficient forward')
parser.add_argument('--data_range', type=str, default='1-800/801-810',  #修改数值或者删掉
                    help='train/test data range')

"""

# 3.网络设置Model specifications
parser.add_argument('--model', default='TESTNETWORK',
                    help='model name')
parser.add_argument('--act', type=str, default='relu',
                    help='activation function')
parser.add_argument('--pre_train', type=str, default='',
                    help='pre-trained model directory')
parser.add_argument('--extend', type=str, default='.',
                    help='pre-trained model directory')
parser.add_argument('--n_resblocks', type=int, default=16,
                    help='number of residual blocks')
parser.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')
parser.add_argument('--res_scale', type=float, default=1,
                    help='residual scaling')
parser.add_argument('--shift_mean', default=True,
                    help='subtract pixel mean from the input')
parser.add_argument('--dilation', action='store_true',
                    help='use dilated convolution')
parser.add_argument('--precision', type=str, default='single',
                    choices=('single', 'half'),
                    help='FP precision for test (single | half)')
# 4.专属网络参数设置
r"""
    暂无

"""
# 5.训练规范Training specifications
parser.add_argument('--reset', action='store_true',
                    help='reset the training')
parser.add_argument('--test_every', type=int, default=1000,
                    help='do test per every N batches')
parser.add_argument('--epochs', type=int, default=400,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='input batch size for training')
parser.add_argument('--self_ensemble', action='store_true',
                    help='use self-ensemble method for test')

r""" 
parser.add_argument('--split_batch', type=int, default=1,
                    help='split the batch into smaller chunks')
parser.add_argument('--gan_k', type=int, default=1,
                    help='k value for adversarial loss')       
"""
parser.add_argument('--test_only', action='store_true',
                    help='set this option to test the model')
parser.add_argument('--evalution', action='store_true',
                    help='set this option to test the model')


# 6.优化设置Optimization specifications
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--decay', type=str, default='100',
                    help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999),
                    help='ADAM beta')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')

r"""
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')
parser.add_argument('--gclip', type=float, default=0,
                    help='gradient clipping threshold (0 = no clipping)')

"""

# 7.loss设置Loss specifications
parser.add_argument('--loss', type=str, default='1*L1',
                    help='loss function configuration')
parser.add_argument('--skip_threshold', type=float, default='1e8',
                    help='skipping batch that has large error')

# 8.参数文件设置Log specifications
parser.add_argument('--save', type=str, default='test',
                    help='file name to save')
parser.add_argument('--load', type=str, default='',
                    help='file name to load')
parser.add_argument('--resume', type=int, default=0,
                    help='resume from specific checkpoint')
parser.add_argument('--save_models', action='store_true',
                    help='save all intermediate models')

r"""
parser.add_argument('--print_every', type=int, default=100,
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_results', action='store_true',
                    help='save output results')
parser.add_argument('--save_gt', action='store_true',
                    help='save low-resolution and high-resolution images together')  

"""

args = parser.parse_args()


r"""
args.scale = list(map(lambda x: int(x), args.scale.split('+')))
args.data_train = args.data_train.split('+')
args.data_test = args.data_test.split('+')

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False

"""

