from collections import deque
import numpy as np
from search_problems import Node, GraphSearchProblem

def bidirectional_search(problem):
    """
        Implement a bidirectional search algorithm that takes instances of SimpleSearchProblem (or its derived
        classes) and provides a valid and optimal path from the initial state to the goal state.

        :param problem: instance of SimpleSearchProblem
        :return: path: a list of states (ints) describing the path from problem.init_state to problem.goal_state[0]
                 num_nodes_expanded: number of nodes expanded by your search
                 max_frontier_size: maximum frontier size during search
        """
    ####
    #   COMPLETE THIS CODE
    ####
    max_frontier_size = 0
    num_nodes_expanded = 0
    path = []

    FWD_Node = Node(None, problem.init_state, None, 0)
    BWD_Node = Node(None, problem.goal_states[0], None, 0)


    # Do FWD path finding, BWD path finding 
    # Meet in the middle
    # Time to backtrack and get all the paths 

    ## FWD VARIABLES ##
    # To find intersection 
    FWD_find_intersect = [] 
    FWD_find_intersect.append(FWD_Node.state)
    # For path traversal
    FWD_dict = {} #Parent storage 
    FWD_dict[FWD_Node.state] = None 
    FWD = deque([FWD_Node])

    ## BWD VARIABLES ##
    # To find intersection 
    BWD_find_intersect = []
    BWD_find_intersect.append(BWD_Node.state)
    # For path traversal
    BWD_dict = {} #Parent storage 
    BWD_dict[BWD_Node.state] = None 
    BWD = deque([BWD_Node])

    ## SEARCHING ##
    # Traversing Forward and Backward at the same time
    # 2 BFS

    while FWD and BWD: 
        # FWD BFS
        node = FWD.popleft()
        FWD_find_intersect.append(node.state)
        #Termination Condition
        if node.state in BWD_find_intersect: 
            if (FWD):
                top_fwd_node = FWD.popleft()
                topFtopB = top_fwd_node.path_cost + BWD_dict[node.state].path_cost
                mu = node.path_cost + BWD_dict[node.state].path_cost
                if mu <= topFtopB:
                    break
                else:
                    node = top_fwd_node

        # Forward BFS procedure 
        for action in (problem.get_actions(node.state)):
            childFwd = problem.get_child_node(node, action)
            if childFwd.state not in FWD_find_intersect:
                num_nodes_expanded += 1 
                FWD.append(childFwd)
                FWD_dict[childFwd.state] = node
                FWD_find_intersect.append(childFwd.state)

        # BWD BFS
        node = BWD.popleft()
        BWD_find_intersect.append(node.state)
        # Termination Condition
        if node.state in FWD_find_intersect: 
            if (BWD):
                top_bwd_node = BWD.popleft()
                topFtopB = FWD_dict[node.state].path_cost + top_bwd_node.path_cost 
                mu = node.path_cost + FWD_dict[node.state].path_cost
                if mu <= topFtopB:
                    break
                else:
                    node = top_bwd_node

        # Backward BFS procedure
        for action in problem.get_actions(node.state): 
            childBwd = problem.get_child_node(node, action)
            if childBwd.state not in BWD_find_intersect:
                BWD.append(childBwd)
                BWD_dict[childBwd.state] = node
                BWD_find_intersect.append(childBwd.state)

        num_nodes_expanded = len(FWD_find_intersect) + len(BWD_find_intersect)


    ## COMPOSING PATH ##
    # Edge case: If reached the end of the while but there wasn't intersection 
    if BWD_Node.state not in BWD_find_intersect and BWD_Node.state not in FWD_find_intersect:
        return [], None, None

    # If was successful, the exited 'node' will be the "intersection"  node. 
    path = [node.state] # Add itself (intersection node) into the path
    traverse_parents = FWD_dict[node.state]

    while True:
        if traverse_parents is None:
            break
        path.append(traverse_parents.state)
        traverse_parents = FWD_dict[traverse_parents.state] #Back tracking FWD parents
    
    path.reverse() #need to reverse to put into the right order

    traverse_parents = BWD_dict[node.state]
    while True:
        if traverse_parents is None:
            break
        path.append(traverse_parents.state)
        traverse_parents = BWD_dict[traverse_parents.state] #Back tracking BWD parents
    

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
    path, num_nodes_expanded, max_frontier_size = bidirectional_search(problem)
    correct = problem.check_graph_solution(path)
    print("Solution is correct: {:}".format(correct))
    print(path)
    print("num_nodes_expanded:", num_nodes_expanded) #
    print("max_frontier_size:",max_frontier_size) #
    
    
    # Use stanford_large_network_facebook_combined.txt to make your own test instances
    E = np.loadtxt('./stanford_large_network_facebook_combined.txt', dtype=int)
    V = np.unique(E)
    goal_states = [349]
    init_state = 0
    problem = GraphSearchProblem(goal_states, init_state, V, E)
    path, num_nodes_expanded, max_frontier_size = bidirectional_search(problem)
    correct = problem.check_graph_solution(path)
    print("Solution is correct: {:}".format(correct))
    print(path)

    


    # Be sure to compare with breadth_first_search!
