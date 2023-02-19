from abc import ABC, abstractmethod
import numpy as np


class SingleMoveGamePlayer(ABC):
    """
    Abstract base class for a symmetric, zero-sum single move game player.
    """
    def __init__(self, game_matrix: np.ndarray):
        self.game_matrix = game_matrix
        self.n_moves = game_matrix.shape[0]
        super().__init__()

    @abstractmethod
    def make_move(self) -> int:
        pass


class IteratedGamePlayer(SingleMoveGamePlayer):
    """
    Abstract base class for a player of an iterated symmetric, zero-sum single move game.
    """
    def __init__(self, game_matrix: np.ndarray):
        super(IteratedGamePlayer, self).__init__(game_matrix)

    @abstractmethod
    def make_move(self) -> int:
        pass

    @abstractmethod
    def update_results(self, my_move, other_move):
        """
        This method is called after each round is played
        :param my_move: the move this agent played in the round that just finished
        :param other_move:
        :return:
        """
        pass

    @abstractmethod
    def reset(self):
        """
        This method is called in between opponents (forget memory, etc.)
        :return:
        """
        pass


class UniformPlayer(IteratedGamePlayer):
    def __init__(self, game_matrix: np.ndarray):
        super(UniformPlayer, self).__init__(game_matrix)

    def make_move(self) -> int:
        """
        :return:
        """
        return np.random.randint(0, self.n_moves)

    def update_results(self, my_move, other_move):
        """
        The UniformPlayer player does not use prior rounds' results during iterated games.
        :param my_move:
        :param other_move:
        :return:
        """
        pass

    def reset(self):
        """
        This method is called in between opponents (forget memory, etc.)
        :return:
        """
        pass


class FirstMovePlayer(IteratedGamePlayer):
    def __init__(self, game_matrix: np.ndarray):
        super(FirstMovePlayer, self).__init__(game_matrix)

    def make_move(self) -> int:
        """
        Always chooses the first move
        :return:
        """
        return 0

    def update_results(self, my_move, other_move):
        """
        The FirstMovePlayer player does not use prior rounds' results during iterated games.
        :param my_move:
        :param other_move:
        :return:
        """
        pass

    def reset(self):
        """
        This method is called in between opponents (forget memory, etc.)
        :return:
        """
        pass


class CopycatPlayer(IteratedGamePlayer):
    def __init__(self, game_matrix: np.ndarray):
        super(CopycatPlayer, self).__init__(game_matrix)
        self.last_move = np.random.randint(self.n_moves)

    def make_move(self) -> int:
        """
        Always copies the last move played
        :return:
        """
        return self.last_move

    def update_results(self, my_move, other_move):
        """
        The CopyCat player simply remembers the opponent's last move.
        :param my_move:
        :param other_move:
        :return:
        """
        self.last_move = other_move

    def reset(self):
        """
        This method is called in between opponents (forget memory, etc.)
        :return:
        """
        self.last_move = np.random.randint(self.n_moves)


def play_game(player1, player2, game_matrix: np.ndarray, N: int = 1000) -> (int, int):
    """
    :param player1: instance of an IteratedGamePlayer subclass for player 1
    :param player2: instance of an IteratedGamePlayer subclass for player 2
    :param game_matrix: square payoff matrix
    :param N: number of rounds of the game to be played
    :return: tuple containing player1's score and player2's score
    """
    p1_score = 0.0
    p2_score = 0.0
    n_moves = game_matrix.shape[0]
    legal_moves = set(range(n_moves))
    for idx in range(N):
        move1 = player1.make_move()
        move2 = player2.make_move()
        if move1 not in legal_moves:
            print("WARNING: Player1 made an illegal move: {:}".format(move1))
            if move2 not in legal_moves:
                print("WARNING: Player2 made an illegal move: {:}".format(move2))
            else:
                p2_score += np.max(game_matrix)
                p1_score -= np.max(game_matrix)
            continue
        elif move2 not in legal_moves:
            print("WARNING: Player2 made an illegal move: {:}".format(move2))
            p1_score += np.max(game_matrix)
            p2_score -= np.max(game_matrix)
            continue
        player1.update_results(move1, move2)
        player2.update_results(move2, move1)

        p1_score += game_matrix[move1, move2]
        p2_score += game_matrix[move2, move1]

    return p1_score, p2_score


class StudentAgent(IteratedGamePlayer):
    """
    The general strategy for this game playing agent is to first detect if the opponent is one of three predefined
    agents: the first move agent, the copycat agent, or the goldfish agent. If it is, then we can always play an
    optimal strategy against the opponent's. To detect the opponent's strategy, we record the first 5 moves of the
    opponent and compare it to our own. During these first five moves, we just initially play a random strategy. Below
    describes the optimal strategies against the three opponent strategies:
    FIRST MOVE AGENT: see if every move the opponent made is the same. If so, we can simply keep playing the corresponding
    winning move.
    COPYCAT AGENT: see if the opponent's move is always our previous move. If so, we play the winning move corresponding
    to our last move.
    GOLDFISH AGENT: see if the opponent's move is always the corresponding winning move to our last move. If so, we
    play the winning move to the winning move of our last move.
    If the opponent uses none of the strategies above, then we resort back to playing the random strategy as before,
    since this is the mixed strategy Nash equilibrium for a standard rock paper scissors game.
    """
    def __init__(self, game_matrix: np.ndarray):
        """
        Initialize your game playing agent. here
        :param game_matrix: square payoff matrix for the game being played.
        """
        super(StudentAgent, self).__init__(game_matrix)
        self.opp_prev_moves = []
        self.my_prev_moves = []
        self.gm = game_matrix
        self.count = 0
        self.n_moves = game_matrix.shape[0]

    def make_move(self) -> int:
        """
        Play your move based on previous moves or whatever reasoning you want.
        :return: an int in (0, ..., n_moves-1) representing your move
        """

        # Given a move, return the corresponding winning move
        def get_winning(move):
            return (move + self.n_moves + 1) % self.n_moves

        # Function to detect if the opponent is using the goldfish agent
        def is_goldfish():
            me = self.my_prev_moves[:-1].copy()
            opp = self.opp_prev_moves[1:].copy()
            for i in range(len(me)):
                if get_winning(me[i]) != opp[i]:
                    return False
            return True

        # Play randomly in the first 5 moves while obtaining data on opponent's strategy
        if self.count < 5:
            return np.random.randint(0, self.n_moves)

        else:
            # First move agent
            if len(set(self.opp_prev_moves)) == 1:
                return get_winning(self.opp_prev_moves[0])

            # Copycat agent
            elif self.my_prev_moves[:-1] == self.opp_prev_moves[1:]:
                return get_winning(self.my_prev_moves[-1])

            # Goldfish agent
            elif is_goldfish():
                return get_winning(get_winning(self.my_prev_moves[-1]))

            # None of the agents above, then play randomly
            else:
                return np.random.randint(0, self.n_moves)

    def update_results(self, my_move, other_move):
        """
        Update your agent based on the round that was just played.
        :param my_move:
        :param other_move:
        :return: nothing
        """

        # Store our move as well as the opponent's move and increment turn counter
        self.opp_prev_moves.append(other_move)
        self.my_prev_moves.append(my_move)
        self.count += 1

    def reset(self):
        """
        This method is called in between opponents (forget memory, etc.).
        :return: nothing
        """

        # Reset move lists and counter when switching agents
        self.my_prev_moves = []
        self.opp_prev_moves = []
        self.count = 0


if __name__ == '__main__':
    """
    Simple test on standard rock-paper-scissors
    The game matrix's row (first index) is indexed by player 1 (P1)'s move (i.e., your move)
    The game matrix's column (second index) is indexed by player 2 (P2)'s move (i.e., the opponent's move)
    Thus, for example, game_matrix[0, 1] represents the score for P1 when P1 plays rock and P2 plays paper: -1.0 
    because rock loses to paper.
    """
    game_matrix = np.array([[0.0, -1.0, 1.0],
                            [1.0, 0.0, -1.0],
                            [-1.0, 1.0, 0.0]])
    uniform_player = UniformPlayer(game_matrix)
    first_move_player = FirstMovePlayer(game_matrix)
    uniform_score, first_move_score = play_game(uniform_player, first_move_player, game_matrix)

    print("Uniform player's score: {:}".format(uniform_score))
    print("First-move player's score: {:}".format(first_move_score))

    # Now try your agent
    student_player = StudentAgent(game_matrix)
    student_score, first_move_score = play_game(student_player, first_move_player, game_matrix)

    print("Your player's score: {:}".format(uniform_score))
    print("First-move player's score: {:}".format(first_move_score))