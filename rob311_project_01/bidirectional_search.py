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
    '''
    Do FWD path finding, BWD path finding 
    Meet in the middle
    Need to consider what if the first intersection met is not the most optimal path. 
    Backtrack and get all the paths 
    '''
    max_frontier_size = 0
    num_nodes_expanded = 0
    path = []

    FWD_Node = Node(None, problem.init_state, None, 0)
    BWD_Node = Node(None, problem.goal_states[0], None, 0)

    ## FWD VARIABLES ##
    # To find intersection, store explored 
    FWD_find_intersect = set() 
    FWD_find_intersect.add(FWD_Node.state)
    # For path traversal
    FWD_dict = {} #Parent storage 
    FWD_dict[FWD_Node.state] = None 
    FWD = deque([FWD_Node]) # frontier

    
    ## BWD VARIABLES ##
    # To find intersection, store explored 
    BWD_find_intersect = set()
    BWD_find_intersect.add(BWD_Node.state)
    # For path traversal
    BWD_dict = {} #Parent storage 
    BWD_dict[BWD_Node.state] = None 
    BWD = deque([BWD_Node])


    # FUNCTION THAT CREATES PATH WHEN GIVEN AN INTERSECTION  
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


    iteration = 0 # to find the OPTIMAL, force to repeat the process through before halting
    stop = np.inf # to find the OPTIMAL, force to repeat the process through before halting
    pathcost = np.inf #GLOBAL MIN PATH_COST
    
    ## MAIN ALGO ##
    while FWD and BWD and iteration < stop: #FWD is forward frontier, BWD is backward frontier
        # print("iteration", iteration)
        # FRONT BFS - implemented from uniform search AIMA
        nodeFwd = FWD.popleft()
        FWD_find_intersect.add(nodeFwd.state)
        FWD_states = set() #TO SPEED UP "NOT IN" COMPARISON
        for action in (problem.get_actions(nodeFwd.state)):
            childFwd = problem.get_child_node(nodeFwd, action)
            if childFwd.state not in FWD_find_intersect and childFwd.state not in FWD_states:
                FWD.append(childFwd)
                FWD_states.add(childFwd.state)
                FWD_dict[childFwd.state] = nodeFwd
                FWD_find_intersect.add(childFwd.state)
        
            elif childFwd.state in FWD_states:
                if (childFwd.path_cost < FWD[FWD.index(childFwd)].path_cost):
                    del FWD[FWD.index(childFwd)]
                    FWD.append(childFwd)  
    
        # BACK BFS - implemented from uniform cost search in AIMA
        nodeBwd = BWD.popleft()
        BWD_find_intersect.add(nodeBwd.state)

        BWD_states = set() # FRONTIER STATES STORAGE
        for action in problem.get_actions(nodeBwd.state): 
            childBwd = problem.get_child_node(nodeBwd, action)
            if childBwd.state not in BWD_find_intersect and childBwd.state not in BWD_states: 
                BWD.append(childBwd)
                BWD_states.add(childBwd.state)
                BWD_dict[childBwd.state] = nodeBwd
                BWD_find_intersect.add(childBwd.state)
        
            elif childBwd.state in BWD_states:
                if (childBwd.path_cost < BWD[BWD.index(childBwd)].path_cost):
                    del BWD[BWD.index(childFwd)]
                    BWD.append(childFwd)


        # CHECK INTERSECT
        if nodeFwd.state in BWD_find_intersect and BWD_dict[nodeFwd.state] is not None:
            stop = iteration + 10
            #print("int1", iteration)
            intersect_cost = nodeFwd.path_cost + BWD_dict[nodeFwd.state].path_cost 
            if intersect_cost < pathcost: 
                path = construct_path(nodeFwd)
                pathcost = intersect_cost
            while FWD: #check better cost along the frontier elements 
                #print("while1")
                v = FWD.popleft()
                if v.state in BWD_find_intersect:
                    if BWD_dict[v.state]is not None and v.path_cost + BWD_dict[v.state].path_cost < nodeFwd.path_cost + BWD_dict[nodeFwd.state].path_cost:
                        path = construct_path(v)
                        pathcost = v.path_cost + BWD_dict[v.state].path_cost 
                
        
        if nodeBwd.state in FWD_find_intersect and FWD_dict[nodeBwd.state] is not None:
            stop = iteration + 10 
            intersect_cost = nodeBwd.path_cost + FWD_dict[nodeBwd.state].path_cost 
            if intersect_cost < pathcost: 
                path = construct_path(nodeBwd)
                pathcost = intersect_cost 
            while BWD: #check better cost among the frontier elements 
                u = BWD.popleft()
                if u.state in FWD_find_intersect:
                    if FWD_dict[u.state] is not None and u.path_cost + FWD_dict[u.state].path_cost < nodeBwd.path_cost + FWD_dict[nodeBwd.state].path_cost:
                        path = construct_path(u) 
                        pathcost = u.path_cost + FWD_dict[u.state].path_cost
                
                       
        iteration+=1
        
    # Edge case: If reached the end of the while but there wasn't intersection 
    if BWD_Node.state not in BWD_find_intersect and BWD_Node.state not in FWD_find_intersect:
        return [], None, None

    print(path)
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


# num_nodes_expanded = len(FWD_find_intersect) + len(BWD_find_intersect)

'''
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
    pathcost = np.inf
    while FWD and BWD and iteration < stop: 
        # FRONT 
        node = FWD.popleft()
        FWD_find_intersect.add(node.state)
        
        FWD_states = set() #BFS1
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

        BWD_states = set() #BFS2
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

        if node.state in FWD_find_intersect and FWD_dict[node.state] is not None:
            stop = iteration + 20 
            intersect_cost = node.path_cost + FWD_dict[node.state].path_cost 
            if intersect_cost < pathcost: 
                path = construct_path(node)
                pathcost = intersect_cost 
            while BWD: #check better cost along the frontier elements 
                u = BWD.popleft()
                if u.state in FWD_find_intersect and u.path_cost < node.path_cost+1:
                    path = construct_path(u) 
                break

        if node.state in BWD_find_intersect and BWD_dict[node.state] is not None:
            stop = iteration + 20
            intersect_cost = node.path_cost + BWD_dict[node.state].path_cost 
            if intersect_cost < pathcost: 
                path = construct_path(node)
                pathcost = intersect_cost
            while FWD: #check better cost along the frontier elements 
                v = FWD.popleft()
                if v.state in BWD_find_intersect and v.path_cost < node.path_cost+1:
                    path = construct_path(v)
                    pathcost = v.path_cost + BWD_dict[v.state].path_cost 
                break
        
        iteration+=1
        
    # Edge case: If reached the end of the while but there wasn't intersection 
    if BWD_Node.state not in BWD_find_intersect and BWD_Node.state not in FWD_find_intersect:
        return [], None, None

    print(path)
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


# num_nodes_expanded = len(FWD_find_intersect) + len(BWD_find_intersect)
'''




'''
         
'''