import numpy as np
### WARNING: DO NOT CHANGE THE NAME OF THIS FILE, ITS FUNCTION SIGNATURE OR IMPORT STATEMENTS


def min_conflicts_n_queens(initialization: list) -> (list, int):
    """
    Solve the N-queens problem with no conflicts (i.e. each row, column, and diagonal contains at most 1 queen).
    Given an initialization for the N-queens problem, which may contain conflicts, this function uses the min-conflicts
    heuristic(see AIMA, pg. 221) to produce a conflict-free solution.

    Be sure to break 'ties' (in terms of minimial conflicts produced by a placement in a row) randomly.
    You should have a hard limit of 1000 steps, as your algorithm should be able to find a solution in far fewer (this
    is assuming you implemented initialize_greedy_n_queens.py correctly).

    Return the solution and the number of steps taken as a tuple. You will only be graded on the solution, but the
    number of steps is useful for your debugging and learning. If this algorithm and your initialization algorithm are
    implemented correctly, you should only take an average of 50 steps for values of N up to 1e6.

    As usual, do not change the import statements at the top of the file. You may import your initialize_greedy_n_queens
    function for testing on your machine, but it will be removed on the autograder (our test script will import both of
    your functions).

    On failure to find a solution after 1000 steps, return the tuple ([], -1).

    :param initialization: numpy array of shape (N,) where the i-th entry is the row of the queen in the ith column (may
                           contain conflicts)

    :return: solution - numpy array of shape (N,) containing a-conflict free assignment of queens (i-th entry represents
    the row of the i-th column, indexed from 0 to N-1)
             num_steps - number of steps (i.e. reassignment of 1 queen's position) required to find the solution.
    """

    N = len(initialization)
    solution = initialization.copy()
    num_steps = 0
    max_steps = 1000

    ## tar cvf handin.tar *.py

    def conflict_count_table(N, row, col, table, fnc_type):
        oper = fnc_type # if wnat to decrement, will be -1. To increment, will be +1. 
        
        for i in range(1, N):
            # check positions - diagonals and left and right of itself. 
            up_right = 0 <= row - i < N and 0 <= col + i < N
            up_left = 0 <= row - i < N and 0 <= col-i < N
            down_right = 0 <= row + i < N and 0 <= col + i < N
            down_left = 0 <= row + i < N and 0 <= col - i < N
            right = 0 <= row < N and 0 <= col+i < N
            left = 0 <= row < N and 0 <= col-i < N
            
            # if true do the operation 
            if up_right: table[row-i][col+i] += oper 
            
            if up_left: table[row-i][col-i] += oper 
        
            if down_right: table[row+i][col+i] += oper 

            if down_left: table[row+i][col-i] += oper 

            if right: table[row][col+i] += oper  
    
            if left:table[row][col-i] += oper  

        return table

    con_table = np.zeros([N,N])
    for col, row in enumerate(solution):
        con_table = conflict_count_table(N, row, col, con_table, 1)

    for idx in range(max_steps):     
        # want '0' in con_table for each column
        zero_counter = 0 
        for i in range (0,N):
            if 0 in con_table[:, i]:
                zero_counter +=1 
            if zero_counter == N:
                #print(con_table)
                print("sol:", solution)
                return solution, num_steps

        # a random column containing conflicts
        conflict_cols = []
        for col, row in enumerate(solution):
            if con_table[row][col] > 0:
                conflict_cols.append(col)

        rand_col = np.int(np.random.choice(conflict_cols, 1))

        old_row = solution[rand_col]
        con_table = conflict_count_table(N, old_row, rand_col, con_table, -1)

        min_conf_inds = []
        min_conf = np.amin(con_table[:, rand_col])
        for ind in range(len(con_table[:, rand_col])):
            if (con_table[ind][rand_col] == min_conf):
                min_conf_inds.append(ind)
 
        solution[rand_col] = np.int(np.random.choice(min_conf_inds, 1))
        new_row = solution[rand_col]
        con_table = conflict_count_table(N, new_row, rand_col, con_table, 1)

        num_steps += 1
    print("sol:", solution)
    return solution, num_steps



if __name__ == '__main__':
    # Test your code here!
    from initialize_greedy_n_queens import initialize_greedy_n_queens
    from support import plot_n_queens_solution

    N = 10
    # Use this after implementing initialize_greedy_n_queens.py
    assignment_initial = initialize_greedy_n_queens(N)
    # Plot the initial greedy assignment
    plot_n_queens_solution(assignment_initial)

    assignment_solved, n_steps = min_conflicts_n_queens(assignment_initial)
    # Plot the solution produced by your algorithm
    plot_n_queens_solution(assignment_solved)
