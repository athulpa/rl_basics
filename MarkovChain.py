
import random

class MarkovChain:
    def __init__(self, states, trans_probs='equal'):
        # states is a list of state-labels
        # tprob is a list of lists; tprob[i][j] being the probability to transition to 'j' from 'i'
        # there are no disallowed states from any given state, although you can set Prob[S(i) -> S(j)]=0
        # The markov chain doesn't store state
        
        if(type(states)==int):
            self.states = [f"State-{i}" for i in range(states)]
        elif(type(states)==list):
            self.states = states
        else:
            raise TypeError(f"Unknown value for arg 'states': {states} in call to MarkovChain.__init__()")
            
        if(trans_probs == 'equal'):
            self.tprob = [[1/len(self.states) for j in range(len(self.states))] for i in range(len(self.states))]
        elif(type(trans_probs) == list):
            self.tprob = trans_probs
        else:
            raise TypeError(f"Unknown value for arg 'trans_probs': {trans_probs} in call to MarkovChain.__init__()")
        
        
    def __getattr__(self, name):
        if(name=='nStates'):
            return len(self.states)
        
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
            
    def copy(self):
        states = self.states.copy()
        tprob = [self.tprob[st].copy() for st in range(self.nStates)]
        return MarkovChain(states=states, trans_probs=tprob)
    
    def equalize_tprobs(self):
        for st in range(self.nStates):
            T = sum(self.tprob[st])
            if(T==0):                   # when all probabilities are set to '0'
                self.tprob[st] = [(1/self.nStates) for i in range(self.nStates)]
            self.tprob[st] = [p/T for p in self.tprob[st]]
        
        
    def next_state_from(self, st: int, x: float) -> int:
        # 'x' must be a value b/w 0 and 1, generated using rng.random()
        tot = 0
        for newSt in range(self.nStates):
            tot += self.tprob[st][newSt]
            if(tot > x):
                return newSt
        else:
            msg = f"Value '{x}' wasn't reached on adding probs: {self.tprob[st]}, total={tot}"
            raise RuntimeError(msg)
        
    def next_state_sequence(self, start: int, Xvals: list[float]) -> list[int]:
        # 'Xvals' must be a list of numbers in [0,1)
        st = start; ret = []
        for x in Xvals:
            st = self.next_state_from(st, x)
            ret.append(st)
        return ret
    
 
class MarkovChainIterator:
    pass

# Estimate the steady-state distribution of a given markov chain
def estimate_steady_dist(mchain: MarkovChain, start=None, nIter=100_000, rng=None, counts_to_prob=False) -> list[float]:
    if((type(rng) is int) or (type(rng) is float)):     # preset seed-value
        rng = random.Random(rng)
    elif(rng is None):                            # random seed-value
        rng = random.Random(None)
        
    if(start is None):
        start = rng.randint(0, mchain.nStates-1)
    
    cnt = {i:0 for i in range(mchain.nStates)}
    
    st = start
    for i in range(nIter):
        st = mchain.next_state_from(st, rng.random())
        cnt[st] += 1
        
    if(counts_to_prob):
        return {i:cnt[i]/nIter for i in range(mchain.nStates)}
    else:
        return cnt