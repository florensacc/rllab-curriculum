from __future__ import absolute_import
from sandbox.dave.rllab.spaces import Box, Crown

class GoalGenerator(object):
    def __init__(self):
        self.episodes_with_goal = 1
        #self.max_episodes_with_goal = 50000 #TODO - use batch size for training
        self.goal = None

    def generate_goal(self, obs):
        """
        Generate the next goal for the agent to try to achieve.
        Input
        -----
        obs: The current observation of the agent.
        Outputs
        -------
        A numpy vector containing the next goal.
        """
        raise NotImplementedError
    # TODO - implement general functionality for stubborn goals - keep track of whether the
    # most recent reward (max reward?) was below a threshold and if so, return the
    # most recent goal, up to some maximum number of times.
    # TODO - implement general adaptive curriculum functionality - whether the last goal completed was good
    # and whether we want to make the next task harder or easier.

    def update(self, paths):
        pass

    def get_diagnostics(self):
        return dict()


class FixedGoalGenerator(GoalGenerator):
    def __init__(self, goal):
        self.goal = goal

    def generate_goal(self, obs):
        return self.goal


class BoxGoalGenerator(GoalGenerator):
    """ Generate a goal from within a box """
    def __init__(self, low, high, shape=None, max_episodes_with_goal=1):
        """ Initialize the bounds of the box """
        self.box = Box(low, high, shape)
        #print "Box init: " + str(self.box.low) + " " + str(self.box.high)

        #self.max_episodes_with_goal = max_episodes_with_goal
        super(BoxGoalGenerator, self).__init__()
        #print "Box init2: " + str(self.box.low) + " " + str(self.box.high)

    def generate_goal(self, obs):
        # Generate a goal randomly from a box
        self.goal = self.box.sample()

        # if self.episodes_with_goal < self.max_episodes_with_goal and self.goal is not None:
        #     self.episodes_with_goal += 1
        #     #print "Reusing goal, episodes with goal: " + str(self.episodes_with_goal)
        # else:
        #     self.goal = self.box.sample()
        #     self.episodes_with_goal = 1
        #     #print "Generating new goal"

        # print "Box: " + str(self.box.low) + " " + str(self.box.high)
        # print "Goal: " + str(self.goal)
        return self.goal

        # TODO - if using a curriculum, generate a goal from within the box near to the current observation

class SphereGoalGenerator(GoalGenerator):
    """ Generate a goal from within a sphere with a given center and radius"""
    def generate_goal(self, obs):
        pass
        # TODO - generate a goal randomly from a sphere
        # TODO - if using a curriculum, generate a goal from within the sphere near to the current observation

class ListGoalGenerator(GoalGenerator):
    """ Generate a goal from a list of goals"""
    def generate_goal(self, obs):
        pass
        # TODO - store an index, and the generate a goal from the next goal on the list


class CrownGoalGenerator(GoalGenerator):
    """ Generate a goal from within a Crown normal to the z axis """
    def __init__(self, radius_low, radius_high, shape=None, max_episodes_with_goal=1):
        """ Initialize the bounds of the crown """
        self.crown = Crown(radius_low, radius_high, shape)
        #print "Box init: " + str(self.box.low) + " " + str(self.box.high)

        #self.max_episodes_with_goal = max_episodes_with_goal
        super(CrownGoalGenerator, self).__init__()
        #print "Box init2: " + str(self.box.low) + " " + str(self.box.high)

    def generate_goal(self, center):
        # Generate a goal randomly from a box
        self.goal = self.crown.sample(center)

        # if self.episodes_with_goal < self.max_episodes_with_goal and self.goal is not None:
        #     self.episodes_with_goal += 1
        #     #print "Reusing goal, episodes with goal: " + str(self.episodes_with_goal)
        # else:
        #     self.goal = self.box.sample()
        #     self.episodes_with_goal = 1
        #     #print "Generating new goal"

        # print "Box: " + str(self.box.low) + " " + str(self.box.high)
        # print "Goal: " + str(self.goal)
        return self.goal

