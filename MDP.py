
import json

class MDP:
    # A Markov Decision Process 
    # Every state-action pair has 
    #                   a) deterministic reward
    #                   b) stochastic transition to the next state
    
    def __init__(self, states, actions, state_action_pairs=None, transitions=None):
        # states is a list of state-labels
        # actions is a list of action-labels
        # state_action_pairs is a dict of sets: st -> {a1,a2,a3..}
        #   ... where the key 'st' is a state and (a1,a2,a3..) are possible actions from that state  
        # transitions is a dict of lists: (st,a) -> [(p1,s1,r1), (p2,s2,r2), ..], such that (p1+p2+..) = 1
        #   ... it maps a key of state-action pair to a list of triplets (p,s,r)
        #   ... where 'p' is the probability of this transition, 's' is the new state, 'r' is the reward obtained
        #   ... This way, you could have multiple transitions to the same new state, but with different rewards (each with a given prob)
        
        if(type(states)==int):
            self.states = [f"State-{i}" for i in range(states)]
        elif(type(states)==list):
            self.states = states
        else:
            raise TypeError(f"Unknown value for arg 'states': {states} in call to MDP.__init__()")
        
        if(type(actions)==int):
            self.actions = [f"Action-{i}" for i in range(actions)]
        elif(type(actions)==list):
            self.actions = actions
        else:
            raise TypeError(f"Unknown value for arg 'actions': {states} in call to MDP.__init__()")
            
        if(state_action_pairs is None):
            self.state_action_pairs = None
        elif(state_action_pairs == 'all'):
            self.state_action_pairs = {stIdx:set(range(len(self.actions)))  for stIdx in range(len(self.states))}
        elif(type(state_action_pairs) == dict):
            self.setStateActionPairs(state_action_pairs)
        else:
            raise TypeError(f"Unknown value for arg 'state_action_pairs': {state_action_pairs}\
                                in call to MDP.__init__()")
        
        if(transitions is None):
            self.transitions = None
        elif(type(transitions) == dict):
            self.setTransitions(transitions)
        else:
            raise TypeError(f"Unknown value for arg 'transitions': {transitions}\
                                in call to MDP.__init__()")
        
        
    # Return an iterator over possible actions in state 'st'
    # See a usage example in self.set_joint_probs
    def possibleActions(self, st: int):
        return ( (i) for i in range(self.nActions) if (i in self.state_action_pairs[st]) )
    
    def __getattr__(self, name):
        if(name=='nStates'):
            return len(self.states)
        elif(name=='nActions'):
            return len(self.actions)
        
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
    def __repr__(self):
        return f"MarkovDecisionProcess({self.nStates} states, {self.nActions} actions)"
    
    def __str__(self):
        return f"<Markov Decision Process with {self.nStates} states and {self.nActions} actions>"
    
        
    def setStateActionPairs(self, sa_pairs : dict[int, set[int]]):
        if(type(sa_pairs) != dict):
            raise ValueError(f"The given state-action-pairs data: '{sa_pairs}' \
                                     is not a python dictionary")
        invalid_states = [st for st in sa_pairs.keys() if st not in range(self.nStates)]
        if(len(invalid_states)):
            msg = f"The given state-action-pairs data has some invalid states: {invalid_states}"
            raise ValueError(msg)
            
        missing_states = [st for st in range(self.nStates) if st not in sa_pairs.keys()]
        if(len(missing_states)):
            msg = f"The given state-action-pairs data is missing some states: {missing_states}"
            raise ValueError(msg)
        
        invalid_actions = {}
        for st in sa_pairs.keys():
            inv = [a for a in sa_pairs[st] if a not in range(self.nActions)]
            if(len(inv)):
                invalid_actions[st] = inv
        if(invalid_actions):        # if not {}
            msg = "The given state-action-pairs data has some invalid actions:\t\t"
            msg += [str(tuple(invalid_actions[st]))+f" in State '{st}'" \
                        for st in invalid_actions.keys()].join(", \t\t")
            raise ValueError(msg)
        
        self.state_action_pairs = sa_pairs
        
        
    def setTransitions(self, transitions : dict[  tuple[int,int],  list[tuple[float,int,float]]  ]) -> None:
        if(type(transitions) != dict):
            raise ValueError(f"The given Transitions data: '{transitions}' \
                                 is not a python dictionary")
        
        # Collect all state-action pairs missing from the transitions' keys
        missing = \
        [(st,a) for st in range(self.nStates) for a in self.possibleActions(st) if (st,a) not in transitions.keys()]
        
        if(len(missing)):
            raise ValueError(f"The given Transitions data: '{transitions}' \
                                 did not have some state-action pairs: '{missing}'")
        else:        
            self.transitions = transitions
    
    # Makes sure that the probabilities of transitions from each (st,a) pair, add up to 1.
    # To set equal probabilities for all transitions, set every p=0 before calling this method.
    def normalizeTransitionProbs(self) -> None:
        for st in range(self.nStates):
            for a in self.possibleActions(st):
                probs = self.transitions[st,a]          # the probabilities of trasitions from state-action pair (st,a)
                T = sum((p for (p,s,r) in probs))       # the sum of all probabilities
                if(T==0):
                    self.transitions[st,a] = [(1/len(probs),s,r) for (p,s,r) in probs]
                self.transitions[st,a] = [(p/T,s,r) for (p,s,r) in probs]
                
                
    # get the next_state and reward, based on transition-probs for the given state-action pair and a random value 'x' in [0,1) 
    def next_state_and_reward(self, state, action, x: float) -> tuple[int,float]:
        probs = self.transitions[state,action]
        tot = 0.
        for (p,s,r) in probs:
            tot += p
            if(tot > x):
                return s,r
        else:
            msg = f"Value '{x}' wasn't reached on adding probs[{state},{action}]: {probs}, total={tot}"
            raise RuntimeError(msg)
        
    def copy(self):
        states = self.states.copy()
        actions = self.actions.copy()
        state_action_pairs = {stIdx  :  self.state_action_pairs[stIdx].copy()     for stIdx in self.state_action_pairs}
        transitions = {(st,a) : self.transitions[st,a].copy()     for (st,a) in self.transitions}
        return MDP(states = states, actions = actions, 
                       state_action_pairs=state_action_pairs, transitions=transitions)
    
    
    # Convert MDP -> string
    def serialize(self) -> str:
        # json doesn't handle 2 values in dict keys
        modified_transitions = \
            { st*self.nActions+a :  self.transitions[st,a]     for st,a in self.transitions}
            
        d = {'version_no'        :  1,   # the version number of the serializer
             # for future-proofing, every redesign of this serialization method will have a different version number
             'states'            :  json.dumps(self.states),
             'actions'           :  json.dumps(self.actions),
             'sa-pairs'          :  json.dumps({ st : list(self.possibleActions(st)) for st in range(self.nStates)}),
             'transitions'       :  json.dumps(modified_transitions)
             }
        return json.dumps(d)
    
    # Convert string -> MDP 
    @staticmethod
    def deserialize(msg : str):
        d = json.loads(msg)
        if(d['version_no'] != 1):
            raise ValueError(f"Encoding mismatch while trying to deserialize data. \
                             Expected version '1', got version '{d['version_no']}'")
        
        states = json.loads(d['states'])
        actions = json.loads(d['actions'])
        nActions = len(actions)
        
        sa_pairs = json.loads(d['sa-pairs'])
        sa_pairs = {int(k):set(sa_pairs[k]) for k in sa_pairs.keys()}
        
        transitions = json.loads(d['transitions'])
        transitions = {  (int(k)//nActions, int(k)%nActions) : [tuple(triplet) for triplet in transitions[k]]   \
                                                                       for k in transitions.keys()}
        
        return MDP(states=states, actions=actions, state_action_pairs=sa_pairs, transitions=transitions)
    
    
    # Save this object to a file 'filename'
    def save(self, filename : str) -> None:
        with open(filename, "w") as outfile:
            outfile.write(self.serialize())
        
    @staticmethod
    def load(filename : str):
        with open(filename, "r") as infile:
            msg = infile.read()
        return MDP.deserialize(msg)
        
    
                
                
                
                    
                    
                    