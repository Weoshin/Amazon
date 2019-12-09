import numpy as np
import math
from utils.PrintAction import PrintAction

EPS = 1e-8
BLACK = -2
WHITE = 2
EMPTY = 0
ARROW = 1


class Mcts:
    """
    蒙特卡洛树搜索类：对给定棋盘状态始终使用“白棋视角”搜索得到下一步最优行动
    """
    def __init__(self, game, nnet, args):
        """
        :param game: 当前棋盘对象
        :param nnet: 神经网络
        :param args: 训练参数
        """
        self.game = game
        self.nnet = nnet
        self.args = args
        self.episodeStep = 0

        self.Game_End = {}        # 输赢状态字典
        self.Actions = {}         # 某状态下所有可走的行动
        self.Pi = {}              # 行动时选点的概率 value: 3 * board_size **2 的一维列表
        self.N = {}               # 某状态的访问次数
        self.Nsa = {}             # 某状态s+动作a（下一状态）访问次数 == N[s+1]
        self.Qsa = {}             # 某状态s+动作a（下一个状态）的奖励值

        self.N_start = {}
        self.N_end = {}
        self.N_arrow = {}
        self.p = PrintAction(self.game)

    def get_best_action(self, board, player):
        """
        使用白棋视角判断最优选择
        :param board:  当前棋盘
        :return best_action: 下一步最优动作
        """
        s = self.game.to_string(board)
        for i in range(self.args.num_mcts_search):
            # print('==============================第', i, '次搜索=================================')
            self.search(board)

        # 断言：当前棋盘在字典N中
        assert s in self.N   # 此处 N 偶尔比真实次数少 1
        # print(self.N[s])
        # print('Mcts-get_best_action: ', self.N[s])
        # 如果一个动作存在Ns_start记录中，则将该点设为 Ns_start[(s, a)]，else 0
        counts_start = [self.N_start[(s, a)] if (s, a) in self.N_start else 0 for a in range(self.game.board_size**2)]
        # 归一化
        p_start = [x / float(self.N[s]) for x in counts_start]
        counts_end = [self.N_end[(s, a)] if (s, a) in self.N_end else 0 for a in range(self.game.board_size**2)]
        p_end = [x / float(self.N[s]) for x in counts_end]
        counts_arrow = [self.N_arrow[(s, a)] if (s, a) in self.N_arrow else 0 for a in range(self.game.board_size**2)]
        p_arrow = [x / float(self.N[s]) for x in counts_arrow]

        # '''
        #     打印
        # '''
        # for i in range(3*self.game.board_size**2):
        #     if i < self.game.board_size**2:
        #         if i == 0:
        #             print('选皇后位置--------由NN处理过的概率----------探索次数')
        #         print(i, ':------------', self.Pi[s][i], ':-----', counts_start[i])
        #     elif i < 2*self.game.board_size**2:
        #         if i == self.game.board_size**2:
        #             print('放皇后位置--------由NN处理过的概率----------探索次数')
        #         print(i-self.game.board_size**2, ':------------', self.Pi[s][i], ':-----', counts_end[i-self.game.board_size**2])
        #     else:
        #         if i == 2*self.game.board_size**2:
        #             print('放箭位置--------由NN处理过的概率----------探索次数')
        #         print(i - 2*self.game.board_size**2, ':------------', self.Pi[s][i], ':-----', counts_arrow[i-2*self.game.board_size**2])

        # 方法二：使用softmax策略选择动作
        pi = p_start
        pi = np.append(pi, p_end)
        pi = np.append(pi, p_arrow)

        # 存储每步棋的 4 * [board, WHITE, pi] 数据
        steps_train_data = []
        # 将局面和策略顺时针旋转180度，返回4个棋盘和策略组成的元组
        sym = self.game.get_symmetries(board, pi)
        for boards, pis in sym:
            steps_train_data.append([boards, player, pis])

        # 使用依概率随机策略选择下一步
        best_action = self.get_action_on_random_pi(board, pi)
        # 使用最大概率对应的值进行训练
        # best_action = self.get_action_on_max_pi(board, pi)
        # self.p.print_action(board, pi)
        return best_action, steps_train_data

    def search(self, board):
        """
        对状态进行一次递归的模拟搜索，添加各状态（棋盘）的访问结点信息（始终以白棋视角存储）
        :param board: 棋盘当前
        :return: None
        """
        board_copy = np.copy(board)
        board_key = self.game.to_string(board_copy)
        # 判断是否胜负已分（叶子节点）
        if board_key not in self.Game_End:
            self.Game_End[board_key] = self.game.get_game_ended(board_copy, WHITE)

        if self.Game_End[board_key] != 0:
            # print("模拟到根节点", self.Game_End[board_key])
            return -self.Game_End[board_key]

        # 判断board_key是否为新扩展的节点
        if board_key not in self.Pi:
            # 由神经网路预测策略与v([-1,1]) PS[s] 为[1:300]数组
            self.Pi[board_key], v = self.nnet.predict(board_copy)
            # print(len(self.Pi[board_key]), self.Pi[board_key])
            # print(v)
            # 始终寻找白棋可走的行动
            self.Pi[board_key], legal_actions = self.game.get_valid_actions(board_copy, WHITE, self.Pi[board_key])
            # print(legal_actions.shape)
            # 存储该状态下所有可行动作
            self.Actions[board_key] = legal_actions
            self.N[board_key] = 0
            # print('第一次的', self.Qsa)
            # 存储该状态下所有可行动作
            self.Actions[board_key] = legal_actions
            self.N[board_key] = 0
            return -v
        legal_actions = self.Actions[board_key]
        best_uct = -float('inf')
        # 最好的行动
        best_action = -1
        psa = list()                  # 状态转移概率，长度为当前状态下可走的动作数

        # 将选点概率 Pi 转换成动作的概率 psa
        for a in legal_actions:
            p = 0
            for i in [0, 1, 2]:
                assert self.Pi[board_key][a[i] + i * self.game.board_size ** 2] > 0
                p += math.log(self.Pi[board_key][a[i] + i * self.game.board_size ** 2])
            psa.append(p)
        # print('第一次打印', psa)
        psa = np.array(psa)
        # print('第二次打印', psa)
        psa = np.exp(psa) / sum(np.exp(psa))
        # print('第三次打印', psa)
        # print('------------------------------------------------------------')
        # print(sum(psa), '可选动作数：', len(psa))   # 近似等于 1
        # 求置信上限函数：Q + Cpuct * p * (Ns的开方)/ Nsa
        for i, a in enumerate(legal_actions):              # enumerate():将一个元组加上序号，其中 i 为序号：0，1.... a为中的legal_actions元组
            if (board_key, a[0], a[1], a[2]) in self.Qsa:  # board_key:棋盘字符串，a[0], a[1], a[2]分别为起始点，落子点，放箭点
                u = self.args.Cpuct * psa[i] * math.sqrt(self.N[board_key]) / (1 + self.Nsa[(board_key, a[0], a[1], a[2])])
                # sigmoid_u = 2 * (1/(1+np.exp(-u)) - 0.5)
                uct = self.Qsa[(board_key, a[0], a[1], a[2])] + u
                # print('遍历过的动作', a, 'Q值', self.Qsa[(board_key, a[0], a[1], a[2])], 'U值', u, 'UCT', uct)

            else:
                uct = self.args.Cpuct * psa[i] * math.sqrt(self.N[board_key] + EPS)   # 防止乘积为0
                # print('Qsa为0的点u值：', uct, a)
            if uct > best_uct:
                best_uct = uct
                best_action = a

        # print('max_uct：', best_uct, 'best_action: ', best_action)
        a = best_action
        # next_player反转
        next_board, next_player = self.game.get_next_state(board_copy, WHITE, a)
        # 下一个状态，将棋盘颜色反转 (next_player = BLACK)
        next_board = self.game.get_transformed_board(next_board, next_player)

        v = self.search(next_board)

        if (board_key, a[0], a[1], a[2]) in self.Qsa:
            self.Qsa[(board_key, a[0], a[1], a[2])] = (self.Nsa[(board_key, a[0], a[1], a[2])] *
                                                       self.Qsa[(board_key, a[0], a[1], a[2])] + v)\
                                                      / (self.Nsa[(board_key, a[0], a[1], a[2])]+1)
            self.Nsa[(board_key, a[0], a[1], a[2])] += 1

        else:
            self.Qsa[(board_key, a[0], a[1], a[2])] = v
            self.Nsa[(board_key, a[0], a[1], a[2])] = 1

        if (board_key, a[0]) in self.N_start:
            self.N_start[(board_key, a[0])] += 1
        else:
            self.N_start[(board_key, a[0])] = 1

        if (board_key, a[1]) in self.N_end:
            self.N_end[(board_key, a[1])] += 1
        else:
            self.N_end[(board_key, a[1])] = 1

        if (board_key, a[2]) in self.N_arrow:
            self.N_arrow[(board_key, a[2])] += 1
        else:
            self.N_arrow[(board_key, a[2])] = 1

        self.N[board_key] += 1

        return -v

    def get_action_on_random_pi(self, board, pi):
        """
        使用依概率随机策略选择下一步
        :param board: 棋盘
        :param pi: 整体概率
        :return best_action: 依概率随机选择的动作
        """

        pi_start = pi[0:self.game.board_size**2]
        pi_end = pi[self.game.board_size**2:2 * self.game.board_size**2]
        pi_arrow = pi[2 * self.game.board_size**2:3 * self.game.board_size**2]
        # 深拷贝
        copy_board = np.copy(board)
        while True:
            # 将1*100的策略概率的数组传入得到 0~99 的行动点 , action_start,end,arrow都是选出来的点 eg: 43,65....
            action_start = np.random.choice(len(pi_start), p=pi_start)
            # print('start:', action_start)
            action_end = np.random.choice(len(pi_end), p=pi_end)
            # print('end', action_end)
            action_arrow = np.random.choice(len(pi_arrow), p=pi_arrow)
            # print('arrow', action_arrow)
            # 加断言保证起子点有棋子，落子点和放箭点均无棋子
            assert copy_board[action_start // self.game.board_size][action_start % self.game.board_size] == WHITE
            assert copy_board[action_end // self.game.board_size][action_end % self.game.board_size] == EMPTY
            # 不能断言箭的位置一定位空，有可能是该位置是皇后
            if self.game.is_legal_move(copy_board, action_start, action_end):
                copy_board[action_start // self.game.board_size][action_start % self.game.board_size] = EMPTY
                copy_board[action_end // self.game.board_size][action_end % self.game.board_size] = WHITE
                if self.game.is_legal_move(copy_board, action_end, action_arrow):
                    best_action = [action_start, action_end, action_arrow]
                    # 跳出While循环
                    break
                else:
                    copy_board[action_start // self.game.board_size][action_start % self.game.board_size] = WHITE
                    copy_board[action_end // self.game.board_size][action_end % self.game.board_size] = EMPTY
        return best_action

    def get_action_on_max_pi(self, board, pi):

        poo, legal_actions = self.game.get_valid_actions(board, WHITE, pi)
        # print(pi)
        max_pi = -float('inf')
        best_action = []
        for a in legal_actions:
            p = 0
            for i in [0, 1, 2]:
                # 此处不能加断言是因为 Mcts 不可能把所有的动作都探索完，所以会导致有些动作点概率为0
                if pi[a[i] + i * self.game.board_size ** 2] == 0:
                    p = -float('inf')
                    break
                p += math.log(pi[a[i] + i * self.game.board_size ** 2])
            # print(a, np.exp(p) * 100)
            if p > max_pi:
                max_pi = p
                best_action = a
        return best_action
