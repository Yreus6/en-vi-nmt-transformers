import os
import argparse
import torch

from helper.nmt_trainer import NMTTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--train-src-data-path', default='./data',
                        help='training src data file')
    parser.add_argument('--train-tgt-data-path', default='./data',
                        help='training tgt data file')
    parser.add_argument('--val-src-data-path', default='./data',
                        help='training src data file')
    parser.add_argument('--val-tgt-data-path', default='./data',
                        help='training tgt data file')
    parser.add_argument('--src-vocab-path', default='./data',
                        help='src vocab data file')
    parser.add_argument('--tgt-vocab-path', default='./data',
                        help='tgt vocab data file')
    parser.add_argument('--save-dir', default='./checkpoints',
                        help='directory to save models.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='the initial learning rate')
    parser.add_argument('--lr-decay', type=float, default=0.5,
                        help='learning rate decay')
    parser.add_argument('--label-smoothing', type=float, default=0.0,
                        help='use label smoothing')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='the weight decay')
    parser.add_argument('--clip-grad', type=float, default=5.0,
                        help='gradient clipping')
    parser.add_argument('--patience', type=int, default=5,
                        help='wait for how many iterations to decay learning rate')
    parser.add_argument('--max-num-trial', type=int, default=5,
                        help='terminate training after how many trials')
    parser.add_argument('--resume', default='',
                        help='the path of resume training model')
    parser.add_argument('--max-model-num', type=int, default=1,
                        help='max models num to save ')
    parser.add_argument('--max-epoch', type=int, default=100,
                        help='max training epoch')
    parser.add_argument('--val-epoch', type=int, default=5,
                        help='the num of steps to start val')
    parser.add_argument('--val-start', type=int, default=5,
                        help='the epoch start to val')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='train batch size')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='the num of training process')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    torch.backends.cudnn.benchmark = True
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    trainer = NMTTrainer(args)
    trainer.setup()
    trainer.train()
