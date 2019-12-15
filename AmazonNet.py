import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append('..')


class AmazonNet(nn.Module):
    """
    亚马逊棋神经网络图定义类
    """
    def __init__(self, game, args):
        """
        torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                        stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        nn.BatchNorm2d():数据的归一化处理
        :param game:
        :param args:
        """
        super(AmazonNet, self).__init__()
        # game params
        self.board_x, self.board_y = game.get_board_size()
        self.action_size = game.get_action_size()
        self.args = args

        self.conv1 = nn.Conv2d(1, args.num_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1)
        self.conv4 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1)
        # 泛化数据，归一化，防止
        self.bn1 = nn.BatchNorm2d(args.num_channels)
        self.bn2 = nn.BatchNorm2d(args.num_channels)
        self.bn3 = nn.BatchNorm2d(args.num_channels)
        self.bn4 = nn.BatchNorm2d(args.num_channels)

        self.fc1 = nn.Linear(args.num_channels * (self.board_x - 4) * (self.board_y - 4), 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        # 评估策略
        self.fc3 = nn.Linear(512, self.action_size)
        # 评估v
        self.fc4 = nn.Linear(512, 1)

    def forward(self, s):
        """
        定义前向传播的图
        @params: s :....张 n*n棋盘
        @return: (pi, v): pi: 3*n*n的概率
                          v: 输赢概率  [-1, 1]
        """
        # s: batch_size x board_x x board_y
        s = s.view(-1, 1, self.board_x, self.board_y)
        # batch_size * 1 * board_x * board_y
        s = F.relu(self.bn1(self.conv1(s)))
        # batch_size * num_channels * board_x * board_y
        s = F.relu(self.bn2(self.conv2(s)))
        # batch_size * num_channels * board_x * board_y
        s = F.relu(self.bn3(self.conv3(s)))
        # batch_size * num_channels * (board_x-2) * (board_y-2)
        s = F.relu(self.bn4(self.conv4(s)))
        # batch_size * num_channels * (board_x-4) * (board_y-4)
        s = s.view(-1, self.args.num_channels * (self.board_x - 4) * (self.board_y - 4))
        # batch_size * 1024
        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.args.dropout, training=self.training)
        # batch_size * 512
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.args.dropout, training=self.training)
        # 3 * batch_size * action_size
        pi = self.fc3(s)
        # batch_size x 1
        v = self.fc4(s)

        return F.softmax(pi, dim=1), torch.tanh(v)
