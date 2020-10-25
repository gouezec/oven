import copy
import time
from utils import Queue, PriorityQueue
MAX_DAYS = 10


class Action:
    def do(self, state):
        raise NotImplementedError

class Problem:

    """The abstract class for a formal problem.  You should subclass
    this and implement the methods actions and result, and possibly
    __init__, goal_test, and path_cost. Then you will create instances
    of your subclass and solve them with the various search functions."""

    def __init__(self, initial):
        """The constructor specifies the initial state."""
        self.initial = initial

    def initial_state(self):
        return self.initial

    def actions(self, state):
        """Return the actions that can be executed in the given
        state. The result would typically be a list, but if there are
        many actions, consider yielding them one at a time in an
        iterator, rather than building them all at once."""
        raise NotImplementedError

    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        raise NotImplementedError

    def goal_test(self, state):
        """Return True if the state is a goal."""
        raise NotImplementedError


    def successors(self, state):
        successors = []
        actions = self.actions(state)
        for action in actions:
            new_state = self.result(state, action)
            cost = self.path_cost(0, state, action, new_state)
            successor = (new_state, str(action), cost)
            successors.append(successor)
        return successors

    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2.  If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        return c + 1

    def value(self, state):
        """For optimization problems, each state has a value.  Hill-climbing
        and related algorithms try to maximize this value."""
        raise NotImplementedError


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    def frontier_last_states(frontier):
        states = [path[2][-1][0] for path in frontier.heap]
        return states
    
    def path_cost(path):
        last_state = path[-1][0]
        cost = sum([cost for _,_, cost in path])
        hcost = heuristic(last_state, problem)
        return cost + hcost
    
    frontier = PriorityQueue()
    start_path = [(problem.initial_state(), 'START', 0.0)]
    frontier.push(start_path, path_cost(start_path) )
    explored = set()
    limit = 50000
    while limit > 0:
        if frontier.isEmpty():
            return []
        path = frontier.pop()
        end_step = path[-1]
        end_state = end_step[0]
        explored.add(end_state.immutable())
        if problem.goal_test(end_state):
            return [action for _, action, _ in path[1:]], True, (len(explored), len(frontier))
        successors = problem.successors(end_state)
        for successor in successors:
            last_state = successor[0]
            if (last_state not in frontier_last_states(frontier) and last_state.immutable() not in explored) or problem.goal_test(last_state): 
                extended_path = path.copy()
                extended_path.append(successor)
                frontier.push(extended_path, path_cost(extended_path) )
            else:
                pass
        limit -= 1
    return  [action for _, action, _ in frontier.pop()[1:]], False, (len(explored), len(frontier))

class Oven:
    def __init__(self, mcu, is_on=False, mcu_occupied=0):
        self.max_mcu = mcu
        self.is_on = is_on
        self.mcu_occupied = mcu_occupied
        self.cycle = 1

    def turn_on(self, day_num):
        self.is_on = True
        self.start_day = day_num

    def turn_off(self):
        self.is_on = False
        self.mcu_occupied = 0
        self.start_day = None
    
    def mcu_free(self):
        return self.max_mcu - self.mcu_occupied

    def load(self, mcu):
        assert (not self.is_on) and (mcu <=  self.max_mcu - self.mcu_occupied)
        self.mcu_occupied += mcu

    def is_empty(self):
        return self.mcu_occupied == 0

    def immutable(self):
        return (self.max_mcu, self.is_on, self.mcu_occupied, self.cycle)


class State:
    def __init__(self, ovens, queues, day_num):
        self.ovens = ovens
        self.queues = queues
        self.day_num = day_num

    def move_forward(self, days):
        # Turn off ovens if over
        for oven in self.ovens:
            if oven.is_on and (self.day_num + days - oven.start_day >= oven.cycle):
                oven.turn_off()

        # Moves forward in time
        self.day_num += days

    def is_final(self):
        ovens_turned_off = all([not oven.is_on for oven in self.ovens])
        ovens_empty = all([oven.mcu_occupied == 0 for oven in self.ovens])
        queues_empty = all([len(q)==0 for _, q in self.queues.items()])
        return ovens_turned_off and ovens_empty and queues_empty 

    def immutable(self):
        ovens = (o.immutable() for o in self.ovens)
        queues = ( (mcu,tuple(q)) for mcu, q in self.queues.items())
        return (ovens, queues, self.day_num)


class LoadAction(Action):
    """ Load oven <oven_id> with a computer with a size of <mcu> 
    """
    def __init__(self, oven_id, mcu):
        self.oven_id = oven_id
        self.mcu = mcu

    def do(self, state):
        new_state = copy.deepcopy(state)
        new_state.ovens[self.oven_id].load(self.mcu)
        new_state.queues[self.mcu].pop()
        return new_state

    def __str__(self):
        return "LOAD {} MCU INTO OVEN {}".format(self.mcu, self.oven_id)

class TurnOnAction(Action):
    def __init__(self, oven_id):
        self.oven_id = oven_id

    def do(self, state):
        new_state = copy.deepcopy(state)
        new_state.ovens[self.oven_id].turn_on(new_state.day_num)
        return new_state

    def __str__(self):
        return "TURN ON OVEN {}".format(self.oven_id)

class WaitAction(Action):
    def __init__(self, days):
        self.days = days

    def do(self, state):
        new_state = copy.deepcopy(state)
        new_state.move_forward(self.days)
        return new_state

    def __str__(self):
        return "WAIT {} DAYS".format(self.days)


class OvenProblem(Problem):

    def __init__(self):
        self.initial = State(ovens=[Oven(7), Oven(5)],
                             queues = {2:[0,0,0,0,0,0], 3:[0,0,0,0,0,0], 7:[0]},
                             day_num = 0)

    def actions(self, state):
        actions = []
        # Check wheteher Load actions are possible
        for oven_id, oven in enumerate(state.ovens):
            if not oven.is_on:
                for queue_mcu, queue_production_days in state.queues.items():
                    if queue_mcu <= oven.mcu_free() and len(queue_production_days) > 0:
                        actions.append(LoadAction(oven_id, queue_mcu))
        # Check wheteher Turn On actions are possible
        for oven_id, oven in enumerate(state.ovens):
            if not oven.is_on and not oven.is_empty():
                actions.append(TurnOnAction(oven_id))
        # Wait action only if any ovens are on
        if any([o.is_on for o in state.ovens]):
            actions.append(WaitAction(1))
        return actions

    def result(self, state, action):
        return action.do(state)

    def path_cost(self, c, state1, action, state2):
        if isinstance(action, WaitAction):
            return 2.0
        elif isinstance(action, TurnOnAction):
            return 1.0
        elif isinstance(action, LoadAction):
            return 0.0
        else:
            return 0.0

    def goal_test(self, state):
        return state.is_final()


def heuristic1(state, problem=None):
    mcu_in_queue = sum([mcu*len(q) for mcu, q in state.queues.items()])
    mcu_in_oven = sum([o.max_mcu for o in state.ovens])
    return 4.0 * float(mcu_in_queue) / mcu_in_oven

problem = OvenProblem()
import time

start = time.time()
#actions = aStarSearch(problem, nullHeuristic)
actions, done, stats = aStarSearch(problem, heuristic1)
end = time.time()
for action in actions:
    print(action)
print('Time: {} s'.format(end - start))
print('Explored: {}, Frontier: {}'.format(*stats))
