import random

import numpy as np
import scipy.spatial


class GoalCollection(object):
    """ A collection of goals, with minimum distance threshold for new goals. """
    
    def __init__(self, distance_threshold=None):
        self.distance_threshold = distance_threshold
        self.goal_list = []
    
    def sample(self, size, replace=False):
        return sample_matrix_row(np.array(self.goal_list), size, replace)
    
    def _process_goals(self, goals):
        goals = np.array(goals)
        
        results = [goals[0]]
        for goal in goals[1:]:
            if np.amin(scipy.spatial.distance.cdist(results, goal.reshape(1, -1))) > self.distance_threshold:
                results.append(goal)
        return np.array(results)
                
    def append(self, goals):
        goals = np.array(goals)
        if self.distance_threshold is not None:
            goals = self._process_goals(goals)
            if len(self.goal_list) > 0:
                dists = scipy.spatial.distance.cdist(self.goal_list, goals)
                indices = np.amin(dists, axis=0) > self.distance_threshold
                goals = goals[indices, :]
        self.goal_list.extend(goals)
        
    @property
    def goals(self):
        return np.array(self.goal_list)
            

def sample_matrix_row(M, size, replace=False):
    if size > M.shape[0] or replace:
        indices = np.random.randint(0, M.shape[0], size)
    else:
        indices = np.random.choice(M.shape[0], size)
    return M[indices, :]
