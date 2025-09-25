from mymdp import MDP
import math

class ValueAgent:
    """Value-based Agent template (Used as a parent class for VIAgent and PIAgent)
    An agent should maintain:
    - q table (dict[state,dict[action,q-value]])
    - v table (dict[state,v-value])
    - policy table (dict[state,dict[action,probability]])
    - mdp (An MDP instance)
    - v_update_history (list of the v tables): [Grading purpose only] Every time when you update the v table, you need to append the v table to this list. (include the initial value)
    """    
    def __init__(self, mdp: MDP, conv_thresh: float=0.000001) -> None:
        """Initialization

        Args:
            mdp (MDP): An MDP instance
            conv_thresh (float, optional): a threshold for convergence approximation. Defaults to 0.000001.            
        """        
        self.q = dict()
        self.v = {s: 0.0 for s in mdp.states()}
        self.pi = dict()
        self.mdp = mdp
        self.thresh = conv_thresh
        self.v_update_history = list()

    def init_random_policy(self):
        """Initialize the policy function with equally distributed random probability.

        When n actions are available at state s, the probability of choosing an action should be 1/n.
        """        
        self.pi = {} # empty policy
        for state in self.mdp.states(): # prob for each state
            actions = self.mdp.actions(state) # all posisible actions at state
            num_actions = len(actions)
            if num_actions > 0:
                prob = 1.0 / num_actions # equally distributed random probability
                self.pi[state] = {action: prob for action in actions}
            else:
                self.pi[state] = {}
                    
    def computeq_fromv(self, v: dict[str,float]) -> dict[str,dict[str,float]]:
        """Given a state-value table, compute the action-state values.
        For deterministic actions, q(s,a) = E[r] + v(s'). Check the lecture slides.

        Args:
            v (dict[str,float]): a state value table (before update) {state:v-value}

        Returns:
            dict[str,dict[str,float]]: a q value table {state:{action:q-value}}
        """
        q = {} # q table
        for s in self.mdp.states():
            q[s] = {} # map with in a state
            for a in self.mdp.actions(s): # Every possible action at state s
                expected_value = 0.0
                for s_prime, prob in self.mdp.T(s, a): # Every possible next state and its probability
                    reward = self.mdp.R(s, a, s_prime) # immediate reward (s -> a -> s_prime)
                    expected_value += prob * (reward + self.mdp.gamma * v[s_prime]) # prob * (reward + gamma * V(s_prime))
                q[s][a] = expected_value
        return q

    def greedy_policy_improvement(self, v: dict[str,float]) -> dict[str,dict[str,float]]:
        """Greedy policy improvement algorithm. Given a state-value table, update the policy pi.

        Args:
            v (dict[str,float]): a state value table (before update) {state:v-value}

        Returns:
            pi (dict[str,dict[str,float]]): a policy table {state:{action:probability}}
        """
        q = self.computeq_fomv(v) # compute q table from v table
        new_pi = {} # new policy table
        for s in self.mdp.states(): # for each state
            actions = self.mdp.actions(s) # all possible actions at state s
            if not actions:
                new_pi[s] = {}
                continue
            best_action = max(q[s], key=q[s].get) # action with the highest q value
            new_pi[s] = {action: 0.0 for action in actions}
            new_pi[s][best_action] = 1.0 # greedy action
        return new_pi


    def check_term(self, v: dict[str,float], next_v: dict[str,float]) -> bool:
        """Return True if the state value has NOT converged.
        Convergence here is defined as follows: 
        For ANY state s, the update delta, abs(v'(s) - v(s)), is within the threshold (self.thresh).

        Args:
            v (dict[str,float]): a state value table (before update) {state:v-value}
            next_v (dict[str,float]): a state value table (after update)

        Returns:
            bool: True if continue; False if converged
        """
        # Bellman error check for convergence
        for s in v.keys(): 
            if abs(next_v[s] - v[s]) > self.thresh:
                return True
        return False
             


class PIAgent(ValueAgent):
    """Policy Iteration Agent class
    """    
    def __init__(self, mdp: MDP, conv_thresh: float=0.000001) -> None:
        """Initialization (Use the functions from the parent class)
        - set up values for member variables
        - init the policy to the random policy

        Args:
            mdp (MDP): An MDP
            conv_thresh (float, optional): a threshold for convergence approximation. Defaults to 0.000001.
        """        
        super().__init__(mdp, conv_thresh)
        super().init_random_policy() # initialize its policy function with the random policy

    def __iter_policy_eval(self, pi: dict[str,dict[str,float]]) -> dict[str,float]:
        """Iterative policy evaluation algorithm. Given a policy pi, evaluate the value of states (v).

        This function should be called in policy_iteration().

        Args:
            pi (dict[str,dict[str,float]]): a policy table {state:{action:probability}}

        Returns:
            dict[str,float]: state-value table {state:v-value}
        """
        v = self.v.copy() # current v table
        while True:
            prev_v = v.copy() # previous v table
            next_v = {}
            for s in self.mdp.states():
                v_s = 0.0
                for a, action_prob in pi[s].items(): # for each action and its probs givein policy
                    if action_prob == 0:
                        continue
                    q_sa = 0.0
                    for s_prime, trans_prob in self.mdp.T(s, a): # for each possible next state and its transition prob
                        reward = self.mdp.R(s, a, s_prime) # immediate reward
                        q_sa += trans_prob * (reward + self.mdp.gamma * prev_v[s_prime]) # prob * (reward + gamma * V(s_prime))
                    v_s += action_prob * q_sa # sum over all actions
                next_v[s] = v_s
            self.v_update_history.append(next_v.copy()) 
            
            if not self.check_term(prev_v, next_v):
                v = next_v
                break
            v = next_v
        return v


    def policy_iteration(self) -> dict[str,dict[str,float]]:
        """Policy iteration algorithm. Iterating iter_policy_eval and greedy_policy_improvement, update the policy pi until convergence of the state-value function.

        You must use:
         - __iter_policy_eval
         - greedy_policy_improvement        

        This function is called to run PI. 
        e.g.
        mdp = MDP("./mdp1.json")
        dpa = PIAgent(mdp)
        dpa.policy_iteration()

        Returns:
            pi (dict[str,dict[str,float]]): a policy table {state:{action:probability}}
        """
        while True:
            self.v = self.__iter_policy_eval(self.pi) # policy eval
            new_pi = self.greedy_policy_improvement(self.v) # policy improvement

            if self.pi == new_pi: 
                break
            self.pi = new_pi
        return self.pi


class VIAgent(ValueAgent):
    """Value Iteration Agent class
    """
    def __init__(self, mdp: MDP, conv_thresh: float=0.000001) -> None:
        """Initialization (Use the functions from the parent class)
        - set up values for member variables
        - init the policy to the random policy     

        Args:
            mdp (MDP): An MDP
            conv_thresh (float, optional): a threshold for convergence approximation. Defaults to 0.000001.
        """        
        super().__init__(mdp, conv_thresh)
        super().init_random_policy() # initialize its policy function with the random policy

    def value_iteration(self) -> dict[str,dict[str,float]]:
        """Value iteration algorithm. Compute the optimal v values using the value iteration. After that, generate the corresponding optimal policy pi.

        You must use:
         - greedy_policy_improvement           

        This function is called to run VI. 
        e.g.
        mdp = MDP("./mdp1.json")
        via = VIAgent(mdp)
        via.value_iteration()

        Returns:
            pi (dict[str,dict[str,float]]): a policy table {state:{action:probability}}
        """
        pass
