from utils.AverageMeter import AverageMeter
from utils.bar import Bar
import time


class UpdateNet:
    def __init__(self, player1, player2, game, display=None):
        """
        使用新旧网络对弈，根据结果更新网络
        @params: player1:
                 player2:
                 game:
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display

    def play_game(self, verbose=False):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        players = [self.player2, None, self.player1]
        curPlayer = 1
        board = self.game.getInitBoard()
        it = 0
        while self.game.getGameEnded(board, curPlayer) == 0:
            it += 1
            if verbose:
                assert (self.display)
                print("Turn ", str(it), "Player ", str(curPlayer))
                self.display(board)
            action = players[curPlayer + 1](self.game.getCanonicalForm(board, curPlayer))

            valids = self.game.getValidMoves(self.game.getCanonicalForm(board, curPlayer), 1)

            if valids[action] == 0:
                print(action)
                assert valids[action] > 0
            board, curPlayer = self.game.getNextState(board, curPlayer, action)
        if verbose:
            assert (self.display)
            print("Game over: Turn ", str(it), "Result ", str(self.game.getGameEnded(board, 1)))
            self.display(board)
        return self.game.getGameEnded(board, 1)

    def playGames(self, num, verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """
        eps_time = AverageMeter()
        bar = Bar('Arena.playGames', max=num)
        end = time.time()
        eps = 0
        maxeps = int(num)

        num = int(num / 2)
        oneWon = 0
        twoWon = 0
        draws = 0
        for _ in range(num):
            gameResult = self.play_game(verbose=verbose)
            if gameResult == 1:
                oneWon += 1
            elif gameResult == -1:
                twoWon += 1
            else:
                draws += 1
            # bookkeeping + plot progress
            eps += 1
            eps_time.update(time.time() - end)
            end = time.time()
            bar.suffix = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps,
                                                                                                       maxeps=maxeps,
                                                                                                       et=eps_time.avg,
                                                                                                       total=bar.elapsed_td,
                                                                                                       eta=bar.eta_td)
            bar.next()

        self.player1, self.player2 = self.player2, self.player1

        for _ in range(num):
            gameResult = self.play_game(verbose=verbose)
            if gameResult == -1:
                oneWon += 1
            elif gameResult == 1:
                twoWon += 1
            else:
                draws += 1
            # bookkeeping + plot progress
            eps += 1
            eps_time.update(time.time() - end)
            end = time.time()
            bar.suffix = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps,
                                                                                                       maxeps=maxeps,
                                                                                                       et=eps_time.avg,
                                                                                                       total=bar.elapsed_td,
                                                                                                       eta=bar.eta_td)
            bar.next()

        bar.finish()

        return oneWon, twoWon, draws

    pass