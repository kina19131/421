from abc import ABC, abstractmethod
from urllib.request import proxy_bypass
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
    YOUR DOCUMENTATION GOES HERE!

    Known informations: 
    1. Dumb agent: chooses the first move
    2. Copycat agent: copies its opponent's last move 
    3. Goldfish agent: has a short memory 
        => remembers my last move and take the winning move of my last move. 
        => so I need to give my last move's winning move's winning move. 
    4. Uniform agent: randomly chooses one of the three with equal probabiility
    5. Nash agent: uses the mixed Nash equilibrium strategy
        -> Nash equilibrium: each player's strategy is the best response to the other player's response. (cornell)
        -> https://blogs.cornell.edu/info2040/2014/09/12/applying-nash-equilibrium-to-rock-paper-and-scissors/
    6. Markov agent: agent that follows a random Markov process that depends on the last round

    If I know which opponent I am playing against and if I do have a knowledge on what kind of stragety it is using, 
    I can take that advantage. Things like the copycat agent and goldfish agent depends on my move. 
    Therefore, I take random moves for the first couple trials. I am taking this as a learning phase. Then 
    I can see if the agent is a known (identifiable) one. If it is known and I have a corresponding action for it, I can take that to win. 
    
    For the uniform player, they play with equal probability of 1/3. When the parameters on the game matrix, a, b, and c are all
    equal to 1, then I will play the random move with equal proabilities. However, if certain moves become more favorable with 
    greater reward (and the opposite is true for the corresponding move), the equal probabilities are not the best move. 
    A different mixed strategy Nash equilibrium is needed. The new mixed strategy will be computed using the game matrix 
    for the each moves' probabilities. According to the probabilities, I will genereate a sample pool. Then I will be using the 
    moves from the sample.
    -> ref: https://blogs.cornell.edu/info2040/2015/10/19/game-theory-in-rock-paper-scissors/
    -> ref: https://stackoverflow.com/questions/13635448/generate-random-numbers-within-a-range-with-different-probabilities
    p

    For Nash and Markov agent, they play the games based on the probability. These, I will just play with random moves. 
    """

    def __init__(self, game_matrix: np.ndarray):
        """
        Initialize your game playing agent. here
        :param game_matrix: square payoff matrix for the game being played.
        """
        # https://www.youtube.com/watch?v=C6_72XPpKNQ
        
        super(StudentAgent, self).__init__(game_matrix)
        # YOUR CODE GOES HERE
        self.compile_opp_moves = []
        self.compile_my_moves = [] 
        self.game_matrix = game_matrix
        self.count = 0 
        self.n_moves = game_matrix.shape[0]
        #pass

    def make_move(self) -> int:
        """
        Play your move based on previous moves or whatever reasoning you want.
        :return: an int in (0, ..., n_moves-1) representing your move
        """
        # YOUR CODE GOES HERE

        # Ref: https://stackoverflow.com/questions/36354807/solving-linear-equations-w-three-variables-using-numpy

        def value_of_ab():
            # ref: https://people.math.sc.edu/dw7/class_webpages/Math170/Chapter%204/Section%204.4a%20Solutions.pdf
            a = abs(self.game_matrix[0][1])
            b = abs(self.game_matrix[0][2])
            c = abs(self.game_matrix[2][1])

            if a == b and b == c:
                return(0)
            else: 
                x1 = np.array([list(self.game_matrix[0]), list(self.game_matrix[1]), list(self.game_matrix[2])])
                x2 = np.array([1/3, 1/3, 1/3])
                ret = np.linalg.solve(x1, x2)
                return(ret)

        def identify_agent():
            if len(set(self.compile_opp_moves)) == 1:
                return ('first')
            
            if self.compile_opp_moves[1:] == self.compile_my_moves[:-1]:
                return ('copy')
            
            goldfish = False
            for i in range(len(self.compile_my_moves)):
                my_move_taken = (self.compile_my_moves[i] + self.n_moves + 1) % self.n_moves
                my_next_move = (my_move_taken + self.n_moves + 1) % self.n_moves
                if my_next_move == self.compile_opp_moves[i]:
                    goldfish = True
            if (goldfish):
                return ('gold')
            else:
                return ('none')
                

        def next_move(move):
            # Rock, paper, scissor 
            # paper, scissor, rock <- the patern
            return(move + self.n_moves+1) % self.n_moves
        
        '''
        This is the actual action I take 
        '''
        if self.count < 6:
            return np.random.randint(0, self.n_moves)
        else:
            identified = identify_agent()
            # ab = value_of_ab()
            
            if identified == 'first':
                return(next_move(self.compile_opp_moves[0]))

            elif identified == 'copy':
                return(next_move(self.compile_my_moves[-1]))

            elif identified == 'gold':
                return(next_move(next_move(self.compile_my_moves[-1])))

            else:
                return(np.random.randint(0, self.n_moves))

            ## umm... wasn't able to compute the new mixed strategy using code 
            ## if was successful, I should be able to take strategic play for the nash too 
                

        
        # pass

    def update_results(self, my_move, other_move):
        """
        Update your agent based on the round that was just played.
        :param my_move:
        :param other_move:
        :return: nothing
        """
        # YOUR CODE GOES HERE
        self.compile_opp_moves.append(other_move)
        self.compile_my_moves.append(my_move)
        self.count += 1 

    def reset(self):
        """
        This method is called in between opponents (forget memory, etc.).
        :return: nothing
        """
        # YOUR CODE GOES HERE
        self.compile_my_moves = [] 
        self.compile_opp_moves = [] 
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

    print("Your player's score: {:}".format(student_score))
    print("First-move player's score: {:}".format(first_move_score))
