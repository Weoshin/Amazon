import matplotlib.pyplot as plt

BLACK = -2
WHITE = 2
EMPTY = 0
ARROW = 1
board_size = 5


class PrintBoard:
    def __init__(self, game):
        self.game = game
        self.board = game.board
        self.board_size = game.board_size

    @staticmethod
    def print_board(board, epoch):
        white_chess_x = []
        white_chess_y = []
        black_chess_x = []
        black_chess_y = []
        arrow_x = []
        arrow_y = []

        for i in range(board_size):
            for j in range(board_size):
                if board[i][j] == WHITE:
                    white_chess_x.append(j)
                    white_chess_y.append(board_size - i)
                if board[i][j] == BLACK:
                    black_chess_x.append(j)
                    black_chess_y.append(board_size - i)
                if board[i][j] == ARROW:
                    arrow_x.append(j)
                    arrow_y.append(board_size - i)
        # print(board)
        plt.style.use('Solarize_Light2')
        plt.subplot(4, 5, epoch)
        # print(plt.style.available)  # 显示背景风格种类

        plt.scatter(white_chess_x, white_chess_y, c='white', s=100, marker='s')
        plt.scatter(black_chess_x, black_chess_y, c='black', s=100, marker='s')
        plt.scatter(arrow_x, arrow_y, c='blue', s=100, marker='x')
        plt.axis('equal')  # 设置坐标为相等长度
        plt.grid()
        plt.xlim((-1, 5))
        plt.ylim((0, 6))
        epoch += 1

    @staticmethod
    def save_figure(game_num):
        """
        保存图片到figures文件夹种
        @parm game_num:当前游戏盘数
        """
        plt.suptitle('The ' + str(game_num) + 'th game')
        # 此路径为绝对路径，更换计算机时注意修改
        plt.savefig('E:\\PyCharm_workspaces\\NeuqAmazonGame\\figures\\' + str(game_num) + 'th_game.png')
        plt.show()
