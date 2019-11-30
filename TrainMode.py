import time
from collections import deque
from random import shuffle
from Game import Game
import numpy as np
from Mcts import Mcts
from NNet import NNet


BLACK = -2
WHITE = 2
EMPTY = 0
ARROW = 1


class dotdict(dict):
    def __getattr__(self, name):
        return self[name]


# 训练模式的参数
args = dotdict({
    'num_iter': 1000,          # 神经网络训练次数
    'num_play_game': 10,       # 下“num_play_game”盘棋训练一次NNet
    'max_len_queue': 200000,   # 双向列表最大长度
    'num_mcts_search': 500,   # 从某状态模拟搜索到叶结点次数
    'max_batch_size': 20,      # NNet每次训练的最大数据量
    'Cpuct': 0.3,                # 置信上限函数中的“温度”超参数
    'arenaCompare': 40,
    'tempThreshold': 35,       # 探索效率
    'updateThreshold': 0.55,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/models/', 'best.pth.tar'),
})


class TrainMode:

    def __init__(self, game, nnet):
        """
        :param game: 棋盘对象
        :param nnet: 神经网络对象
        """
        self.num_white_win = 0
        self.num_black_win = 0
        self.args = args
        self.player = WHITE
        self.game = game
        self.nnet = nnet
        self.mcts = Mcts(self.game, self.nnet, self.args)
        self.batch = []                 # 每次给NNet喂的数据量,但类型不对（多维列表）
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()

    # 调用NNet开始训练
    def learn(self):
        for i in range(1, self.args.num_iter + 1):
            print('#######################################################################################')
            print('####################################  IterNum： ' + str(i) + ' ####################################')
            print('#######################################################################################')

            # 每次都执行
            if not self.skipFirstSelfPlay or i > 1:
                # deque：双向队列  max_len：队列最大长度：self.args.max_len_queue
                iter_train_data = deque([], maxlen=self.args.max_len_queue)

                # 下“num_play_game”盘棋训练一次NNet
                for i in range(self.args.num_play_game):
                    # 重置搜索树
                    print("====================================== 第", i+1, "盘棋 ======================================")
                    self.mcts = Mcts(self.game, self.nnet, self.args)
                    self.player = WHITE
                    iter_train_data += self.play_one_game()
                print('白棋赢：', self.num_white_win, '盘；', '黑棋赢：', self.num_black_win, '盘')
                # 打印一次迭代后给NN的数据
                print(len(iter_train_data))
                # save the iteration examples to the history
                self.batch.append(iter_train_data)

            # 不断更新训练数据
            # 如果 训练数据 大于规定的训练长度，则将最旧的数据删除
            if len(self.batch) > self.args.max_batch_size:
                print("len(max_batch_size) =", len(self.batch),
                      " => remove the oldest batch")
                self.batch.pop(0)
            
            # 保存训练数据

            self.saveTrainExamples(i - 1)

            # 原batch是多维列表，此处标准化batch
            standard_batch = []
            for e in self.batch:
                # extend() 在列表末尾一次性追加其他序列中多个元素
                standard_batch.extend(e)
            # 打乱数据，是数据服从独立同分布（排除数据间的相关性）
            shuffle(standard_batch)

            # 这里保存的是一个temp也就是一直保存着最近一次的网络，这里是为了和最新的网络进行对弈
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')

            # 开启训练
            self.nnet.train(standard_batch)

            print('PITTING AGAINST PREVIOUS VERSION')
            # 旧、新网路赢的次数 和 平局
            pwins, nwins, draws = 10, 100, 1
            print('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            # 如果旧网路和新网路赢得和为0 或 新网络/ 新网络＋旧网路 小于 更新阈值（0.55）则不更新，否则更新成新网络参数
            if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.args.updateThreshold:
                print('REJECTING NEW MODEL')
                # 如果拒绝了新模型，这老模型就能发挥作用
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                print('ACCEPTING NEW MODEL')
                # 保存当前模型并更新最新模型
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')

    # 完整下一盘游戏
    def play_one_game(self):
        """
        使用Mcts完整下一盘棋
        :return: 4 * [(board, pi, z)] : 返回四个训练数据元组：（棋盘，策略，输赢）
        """
        one_game_train_data = []
        board = self.game.get_init_board(self.game.board_size)
        play_step = 0
        while True:
            play_step += 1
            ts = time.time()
            print('---------------------------')
            print('第', play_step, '步')
            print(board)
            self.mcts.episodeStep = play_step
            # 在MCTS中，始终以白棋视角选择
            transformed_board = self.game.get_transformed_board(board, self.player)
            # 进行多次mcts搜索得出来概率（以白棋视角）
            next_action, steps_train_data = self.mcts.get_best_action(transformed_board)
            one_game_train_data += steps_train_data
            te = time.time()
            print("下一步：", next_action, '用时：', int(te-ts), 's')
            board, self.player = self.game.get_next_state(board, self.player, next_action)

            r = self.game.get_game_ended(board, self.player)
            if r != 0:  # 胜负已分
                if self.player == WHITE:
                    print('白棋输')
                    self.num_black_win += 1
                else:
                    print('黑棋输')
                    self.num_white_win += 1
                print("##### 终局 #####")
                print(board)

                return [(board, pi, r*((-1)**(player != self.player))) for board, player, pi in one_game_train_data]


if __name__ == "__main__":
    game = Game(5)
    nnet = NNet(game)
    train = TrainMode(game, nnet)
    train.learn()
