from netrc import netrc
import queue
import numpy as np
from search_problems import Node, GridSearchProblem, get_random_grid_problem


def a_star_search(problem):
    """
    Uses the A* algorithm to solve an instance of GridSearchProblem. Use the methods of GridSearchProblem along with
    structures and functions from the allowed imports (see above) to implement A*.

    :param problem: an instance of GridSearchProblem to solve
    :return: path: a list of states (ints) describing the path from problem.init_state to problem.goal_state[0]
             num_nodes_expanded: number of nodes expanded by your search
             max_frontier_size: maximum frontier size during search
    """
    ####
    #   COMPLETE THIS CODE
    ####
    '''
    a* search, smarter search, make use of priorityqueue, reference AIMA
    '''

    num_nodes_expanded = 0
    max_frontier_size = 0
    path = []
    
    first_node = Node(None, problem.init_state, (0,0), 0)

    frontier = queue.PriorityQueue()
    frontier.put((0 + problem.heuristic(first_node.state), first_node))

    visited = set()
    visited.add(first_node.state)

    #As long as there's something in the frontier
    while frontier:
        nextNode = frontier.get()[1]

        # if the next node got to the goal state 
        # then compose the path and return
        if nextNode.state == problem.goal_states[0]:
            traverse = nextNode
            while traverse.state != first_node.state:
                path.append(traverse.state)
                traverse = traverse.parent
            path.append(first_node.state)
            path.reverse()
            return path, num_nodes_expanded, max_frontier_size
        
        # else, need to go through unvisited neighbours 
        else:
            for action in problem.get_actions(nextNode.state):
                child = problem.get_child_node(nextNode, action)
                if child.state not in visited: 
                    visited.add(child.state) 
                    frontier.put((child.path_cost + problem.heuristic(child.state), child)) # heuristic used 
        
    return path, num_nodes_expanded, max_frontier_size


def search_phase_transition():
    """
    Simply fill in the prob. of occupancy values for the 'phase transition' and peak nodes expanded within 0.05. You do
    NOT need to submit your code that determines the values here: that should be computed on your own machine. Simply
    fill in the values!

    :return: tuple containing (transition_start_probability, transition_end_probability, peak_probability)
    """
    ####
    #   REPLACE THESE VALUES
    ####
    transition_start_probability = 0.30
    transition_end_probability = 0.45
    peak_nodes_expanded_probability = 0.35
    return transition_start_probability, transition_end_probability, peak_nodes_expanded_probability


if __name__ == '__main__':
    # Test your code here!
    # Create a random instance of GridSearchProblem
    p_occ = 0.25
    M = 10
    N = 10
    problem = get_random_grid_problem(p_occ, M, N)
    # Solve it
    path, num_nodes_expanded, max_frontier_size = a_star_search(problem)
    # Check the result
    correct = problem.check_solution(path)
    print("Solution is correct: {:}".format(correct))
    # Plot the result
    problem.plot_solution(path)

    # Experiment and compare with BFS