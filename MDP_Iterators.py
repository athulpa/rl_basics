
import random

from MDP import MDP

class MDP_Iterator:
    def __init__(self, mdp: MDP, start=None, gamma=1, rng=None, maxIter=1_000):
        
        self.mdp = mdp
        
        self.tot_reward = 0.   # the 'return' of this run of the MDP
        
        self.iterCnt = 0       # no. of steps taken till now
        
        if((type(rng) is int) or (type(rng) is float)):     # preset seed-value
            self.rng = random.Random(rng)
        elif(rng is None):                                  # random seed-value
            self.rng = random.Random(None)  
        else:
            raise ValueError(f"Argument '{rng}' is not a valid RNG or seed-value in {type(self).__name__} con'r")
       
        if(start is None):
            self.state = self.rng.randint(0, self.chain.nStates-1)
        else:
            self.state = start
        
        
    def _select_action(self):
        raise NotImplementedError("Trying to call step() on the abstract base-class iterator. Forgot to overload?")
    
    def step(self):
        if(self.iterCnt == self.maxiter):
            raise StopIteration()    
        
        action = self._select_action()
        s,r = self.mdp.next_state_and_reward(self.state, action, self.rng.random())
        
        self.state = s
        self.tot_reward += r
        return s,r
    
    def __iter__(self):
        return self
    
    def __next__(self):
        return self.step()
        
                

# selects an action at random, using the self.rng
class Random_MDP_Iterator(MDP_Iterator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def _select_action(self):
        self.rng.choice(self.mdp.possibleActions(self.state))


        
# selects an action based on user input
class UserInput_MDP_Iterator(MDP_Iterator):
    def _select_action(self):
        while(True):
            print(f"In '{self.mdp.states[self.state]}', possible actions: {list(self.mdp.possibleActions(self.state))}")
            resp = input("Select Action (or 'q' to stop iteration): ").strip()
        
            if(resp.lower() == 'q'):
                raise StopIteration()
                
            try:
                ret = int(resp)
            except ValueError:
                print(f"Your response '{resp}' is not an integer value. Try again.\n")
                continue
            else:
                if(ret in self.mdp.state_action_pairs[self.state]):
                    break
                else:
                    print(f"Your action '{ret}' is not one of the possible actions. Try again.\n")
                    continue
        return ret
    
    

# selects an action according to a preset policy (that maps each state to a single action)
class Policy_MDP_Iterator(MDP_Iterator):
    def __init__(self, policy=None, *args, **kwargs):  
        # 'policy' must be passsed as the first argument when initializing this object
        
        super().__init__(*args, **kwargs)
        
        if(policy is None):
            self.policy = None
        elif(type(policy) == dict):
            self.setPolicy(policy)
        else:
            raise ValueError(f"Argument policy={policy} is not a dict")
            
    def _select_action(self):
        return self.policy[self.state]

    # set self.policy = policy, after validating it for mistakes
    def setPolicy(self, policy: dict[int,int]):
        if(not hasattr(policy, '__getitem__')):
            msg = f"Expected a dict-like as policy, got '{policy}' of type '{type(policy).__name__}' instead"
            raise ValueError(msg)
            
        invalid_states = [st for st in policy.keys() if st not in range(self.mdp.nStates)]
        if(len(invalid_states)):
            raise ValueError(f"The given policy has some invalid states: {invalid_states}")
            
        missing_states = [st for st in range(self.mdp.nStates) if st not in policy.keys()]
        if(len(missing_states)):
            raise ValueError(f"The given policy has some missing states: {missing_states}")
            
        invalid_actions = \
            {st:policy[st] for st in policy.keys() if policy[st] not in self.mdp.state_action_pairs[st]}
        if(len(invalid_actions)):
            raise ValueError(f"The given policy has some invalid actions {invalid_actions}")
        
        self.policy = policy
            

    


        






        