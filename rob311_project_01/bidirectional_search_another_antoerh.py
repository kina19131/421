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
    FWD_find_intersect = set()
    FWD_find_intersect.add(FWD_Node.state)
    # For path traversal
    FWD_dict = {} #Parent storage 
    FWD_dict[FWD_Node.state] = None 
    FWD = deque([FWD_Node]) # frontier
    
    ## BWD VARIABLES ##
    # To find intersection 
    BWD_find_intersect = set()
    BWD_find_intersect.add(BWD_Node.state)
    # For path traversal
    BWD_dict = {} #Parent storage 
    BWD_dict[BWD_Node.state] = None 
    BWD = deque([BWD_Node])


       # If was successful, the exited 'node' will be the "intersection"  node. 
    def construct_path(intersection):
        path = [intersection.state] # Add itself (intersection node) into the path
        traverse_parents = FWD_dict[intersection.state]

        while True:
            if traverse_parents is None:
                break
            path.append(traverse_parents.state)
            traverse_parents = FWD_dict[traverse_parents.state] #Back tracking FWD parents
        
        path.reverse() #need to reverse to put into the right order

        traverse_parents = BWD_dict[intersection.state]
        while True:
            if traverse_parents is None:
                break
            path.append(traverse_parents.state)
            traverse_parents = BWD_dict[traverse_parents.state] #Back tracking BWD parents
        return(path)







    iteration = 0 
    stop = np.inf
    pathCost = np.inf
    while FWD and BWD and iteration < stop: 
        # FRONT 
        node = FWD.popleft()
        FWD_find_intersect.add(node.state)

        if node.state in BWD_find_intersect:
            curr_path = construct_path(node)
            if BWD_dict[node.state] is not None: curr_pathcost = node.path_cost + BWD_dict[node.state].path_cost 
            stop = iteration + 20
            if curr_pathcost < pathCost: #better than currently stored path 
                    path = curr_path
                    pathCost = curr_pathcost
            while FWD:
                v = FWD.popleft()
                if v.state in BWD_find_intersect and v.path_cost < node.path_cost + 1: 
                    print("ENTERED1")
                    path = construct_path(v)
                    pathCost = v.path_cost + BWD_dict[v.state].path_cost

            #print("here1", path)
            #return path, num_nodes_expanded, max_frontier_size
   
        FWD_states = set() #Frontier States storae
        for action in (problem.get_actions(node.state)):
            childFwd = problem.get_child_node(node, action)
            if childFwd.state not in FWD_find_intersect and childFwd.state not in FWD_states:
                FWD.append(childFwd)
                FWD_states.add(childFwd.state)
                FWD_dict[childFwd.state] = node
                FWD_find_intersect.add(childFwd.state)
        
            elif childFwd.state in FWD_states:
                if (childFwd.path_cost < FWD[FWD.index(childFwd)].path_cost):
                    del FWD[FWD.index(childFwd)]
                    FWD.append(childFwd)  
    
        # BACK 
        node = BWD.popleft()
        BWD_find_intersect.add(node.state)

        if node.state in FWD_find_intersect:
            stop = iteration + 20 
            curr_path = construct_path(node)
            if FWD_dict[node.state] is not None: curr_pathCost = node.path_cost + FWD_dict[node.state].path_cost
            if curr_pathCost < pathCost: 
                path = curr_path
                pathCost = curr_pathCost
            while BWD:
                u = BWD.popleft()
                if u.state in FWD_find_intersect and u.path_cost < node.path_cost + 1:
                    path = construct_path(u) 
                    pathCost = u.path_cost + FWD_dict[u.state].path_cost 

        BWD_states = set() #Frontier States storage
        for action in problem.get_actions(node.state): 
            childBwd = problem.get_child_node(node, action)
            if childBwd.state not in BWD_find_intersect and childBwd.state not in BWD_states: 
                BWD.append(childBwd)
                BWD_states.add(childBwd.state)
                BWD_dict[childBwd.state] = node
                BWD_find_intersect.add(childBwd.state)
        
            elif childBwd.state in BWD_states:
                if (childBwd.path_cost < BWD[BWD.index(childBwd)].path_cost):
                    del BWD[BWD.index(childFwd)]
                    BWD.append(childFwd)
        iteration+=1
    return path, num_nodes_expanded, max_frontier_size
    ## COMPOSING PATH ##
    # Edge case: If reached the end of the while but there wasn't intersection 
    if BWD_Node.state not in BWD_find_intersect and BWD_Node.state not in FWD_find_intersect:
        return [], None, None



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


# num_nodes_expanded = len(FWD_find_intersect) + len(BWD_find_intersect)