from collections import deque
import numpy as np
from search_problems import Node, GraphSearchProblem

def breadth_first_search(problem):
    """
    Implement a simple breadth-first search algorithm that takes instances of SimpleSearchProblem (or its derived
    classes) and provides a valid and optimal path from the initial state to the goal state. Useful for testing your
    bidirectional and A* search algorithms.

    :param problem: instance of SimpleSearchProblem
    :return: path: a list of states (ints) describing the path from problem.init_state to problem.goal_state[0]
             num_nodes_expanded: number of nodes expanded by your search
             max_frontier_size: maximum frontier size during search
    """

    '''
    DETAILS ON SimpleSearchProblem (parameter):
    search problems where acitons are described 
    as tuple b/w states named by numbers, and actions are all uniform cost. 
    
    functions: 
    - .get_child_node
    - .transition 
    - .goal_test
    - .action_cost
    '''
    ####
    #   COMPLETE THIS CODE
    ####
    '''
    Simple BFS, implemented using AIMA textbook 
    '''
    
    frontier_size = 0 
    max_frontier_size = 0
    num_nodes_expanded = 0
    path = []

    if type(problem.init_state) == list: 
        problem.init_state = problem.init_state[0] 
    
    node = Node(parent=None, state=problem.init_state, action = None, path_cost = 0) 
    
    # edge case of the init state being in the goal state rigth away
    if problem.goal_test(node.state):
        path = problem.trace_path(node)
    
    frontier = deque([node])
    explored = set()

    while frontier: # while there's elements in the frontier
        node = frontier.popleft() # the one to explore
        explored.add(node.state)
        
        for action in (problem.get_actions(node.state)):
            child = problem.get_child_node(node, action) # check the child
            num_nodes_expanded+=1
            if child.state not in explored and child not in frontier:
                frontier_size += 1
                if problem.goal_test(child.state): # check if the "child" is where we're trying to get
                    path = problem.trace_path(child) # then trace path
                frontier.append(child) # otherwise add to the frontier for next exploration 
                max_frontier_size = max(max_frontier_size, len(frontier))
   
            
    return path, num_nodes_expanded, max_frontier_size 


if __name__ == '__main__':
    # Simple example
    goal_states = [0]
    init_state = 9
    V = np.arange(0, 10)
    E = np.array([[0, 1],
                  [1, 2],
                  [2, 3],
                  [3, 4],
                  [4, 5],
                  [5, 6],
                  [6, 7],
                  [7, 8],
                  [8, 9],
                  [0, 6],
                  [1, 7],
                  [2, 5],
                  [9, 4]])
    problem = GraphSearchProblem(goal_states, init_state, V, E)
    path, num_nodes_expanded, max_frontier_size = breadth_first_search(problem)
    correct = problem.check_graph_solution(path)
    print("Solution is correct: {:}".format(correct))
    print(path)
    print("num_nodes_expanded:", num_nodes_expanded) #
    print("max_frontier_size:",max_frontier_size) #

    # Use stanford_large_network_facebook_combined.txt to make your own test instances
    E = np.loadtxt('../datasets/stanford_large_network_facebook_combined.txt', dtype=int)
    V = np.unique(E)
    goal_states = [349]
    init_state = 0
    problem = GraphSearchProblem(goal_states, init_state, V, E)
    path, num_nodes_expanded, max_frontier_size = breadth_first_search(problem)
    correct = problem.check_graph_solution(path)
    print("Solution is correct: {:}".format(correct))
    print(path)
    print("num_nodes_expanded:", num_nodes_expanded) #
    print("max_frontier_size:",max_frontier_size) #
    
