from __future__ import unicode_literals
import random
import os
import time
import sys
import numpy as np
import torch
from utils.dotdict import dotdict
from AmazonNet import AmazonNet as annet
from utils.bar import Bar
import torch.optim as optim
from utils.AverageMeter import AverageMeter
import torch.nn.functional as f
sys.path.append('../../')


args = dotdict({
    'lr': 0.001,                        # 学习率or步长
    'dropout': 0.3,                     # dropout率
    'epochs': 10,                       # 每次新传入数据后神经网络的训练次数
    'batch_size': 64,                   #
    'cuda': torch.cuda.is_available(),
    'num_channels': 512,                # 通道数
})


class NNet:
    """
    神经网络训练类
    """
    def __init__(self, game):
        self.board_size = game.board_size
        self.nnet = annet(game, args)
        self.board_x, self.board_y = game.get_board_size()
        self.action_size = game.get_action_size()

        if args.cuda:
            self.nnet.cuda()

    # def predict(self, board):
    #     pi = [random.random() for i in range(3 * self.board_size ** 2)]
    #     pi = np.array(pi)
    #     pi = pi / sum(pi)
    #     # print(pi)
    #     return pi, 2 * (random.random() - 0.5)

# 两个问题：
# 温度超参数是不是应该随着棋局进行改变？
# 5*5 开始时可走步数：260 第四步以后可走的步数就基本小于100
# 可不可以前几步输赢奖励小，后面奖励大。

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        # 使用
        optimizer = optim.Adam(self.nnet.parameters())

        for epoch in range(args.epochs):
            print('EPOCH ::: ' + str(epoch+1))
            # 开始训练模式
            self.nnet.train()
            data_time = AverageMeter()
            batch_time = AverageMeter()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()
            end = time.time()

            bar = Bar('Training Net', max=int(len(examples)/args.batch_size))
            batch_idx = 0

            while batch_idx < int(len(examples)/args.batch_size):
                """
                每次从传入的example中随机选取 args.batch_size (64)个样本作为训练数据
                """
                # 从 0~len(examples) 中产生 batch_size(64)个随机数
                sample_ids = np.random.randint(len(examples), size=args.batch_size)
                # boards[61, 1], pis:[64, 75], vs:[64, 75] zip(*example):解压操作:这里将棋盘,概率,奖励分开存储
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))

                # astype:强制类型转换成浮点型
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                label_pis = torch.FloatTensor(np.array(pis))
                label_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                # gpu support
                if args.cuda:
                    boards, label_pis, label_vs = boards.contiguous().cuda(), label_pis.contiguous().cuda(), label_vs.contiguous().cuda()

                # measure data loading time
                data_time.update(time.time() - end)

                # 带入NN图训练
                out_pi, out_v = self.nnet(boards)
                # print('神经网络输出:', out_pi, out_v, 'NNet.py_train()')
                l_pi = self.loss_pi(label_pis, out_pi)
                l_v = self.loss_v(label_vs, out_v)
                total_loss = l_pi + l_v

                # record loss
                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))

                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                batch_idx += 1

                # plot progress
                bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss_pi: {lpi:.4f} | Loss_v: {lv:.3f}'.format(
                            batch=batch_idx,
                            size=int(len(examples)/args.batch_size),
                            data=data_time.avg,
                            bt=batch_time.avg,
                            total=bar.elapsed_td,
                            eta=bar.eta_td,
                            lpi=pi_losses.avg,
                            lv=v_losses.avg,
                            )
                bar.next()
            bar.finish()

    def predict(self, board):
        """
        @params:board: np array with board
        """
        # 转化成浮点型
        board = torch.FloatTensor(board.astype(np.float64))
        if args.cuda:
            board = board.contiguous().cuda()
        board = board.view(1, self.board_x, self.board_y)
        # 开启预测模式
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board)

        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    @staticmethod
    def loss_pi(labels, outputs):
        """
        计算概率损失值
        @params labels: [64, 75]真值标签
                outputs: [64, 75]NN输出值
        @return loss_pi: 概率的损失函数
        """
        # print("真值:", labels[0], "输出:",  outputs[0])
        return 10 * torch.sum((labels - outputs) ** 2).view(-1) / labels.size()[0]

    @staticmethod
    def loss_v(labels, outputs):
        """
        计算奖励损失值
        @params labels: 真值标签
                outputs: NN输出值
        @return loss_v: 奖励的损失函数
        """
        return torch.sum((labels - outputs.view(-1)) ** 2) / labels.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        """
        保存神经网络模型
        """
        file_path = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            # 保存神经网络模型放到 checkpoint.pth.tar 目录中
            'state_dict': self.nnet.state_dict(),
        }, file_path)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        """
        加载神经网络模型
        """
        file_path = os.path.join(folder, filename)
        if not os.path.exists(file_path):
            raise("No model in path {}".format(file_path))
        map_location = None if args.cuda else 'cpu'
        # 从 checkpoint.pth.tar 中加载模型参数
        checkpoint = torch.load(file_path, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])

