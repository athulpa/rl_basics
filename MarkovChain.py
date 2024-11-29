
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
            
    # Get an iterator for this markov-chain
    # Feed all arguments into the con'r of the iterator-class
    def iterate(self, *args, **kwargs):
        return MarkovChainIterator(self, *args, **kwargs)
    
    # Get the default iterator for this markov-chain
    def __iter__(self):
        return MarkovChainIterator(self)
    
    # make a copy of this MarkovChain object
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


    # calculate the next state of this markov chain, based on a random value 'x' in [0,1)        
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
 
    
class MarkovChainIterator:
    # Iterate over a markov chain
    # Has a state and uses RNG's
    # Make this iterator from a MarkovChain using .iterate() or __iter__()
    # Make this an infinite iterator by passing a -ve value as maxIter
    # See an example of usage in estimate_steady_dist() below
    
    def __init__(self, mchain, start=None, rng=None, maxIter=100_000, skipInitial=True):
        # by giving rng=<some_fixed_int>, we can make this object a fixed (non-random) iterator
        
        self.chain = mchain
        self.maxIter = maxIter
        self.iterCnt = 0
        
        if((type(rng) is int) or (type(rng) is float)):     # preset seed-value
            self.rng = random.Random(rng)
        elif(rng is None):                                  # random seed-value
            self.rng = random.Random(None)  
        
        if(start is None):
            self.state = self.rng.randint(0, self.chain.nStates-1)
        else:
            self.state = start
            
        if(skipInitial):
            next(self, 0)
            
    def __next__(self):
        if(self.iterCnt >= self.maxIter):
            raise StopIteration()
        self.state = self.chain.next_state_from(self.state, self.rng.random())
        self.iterCnt += 1
        return self.state

    def __iter__(self):
        return self
    
    

# Estimate the steady-state distribution of a given markov chain
def estimate_steady_dist(mchain: MarkovChain, start=None, nIter=100_000, rng=None, counts_to_prob=False) -> list[float]:
   
    cnt = {i:0 for i in range(mchain.nStates)}
    
    for st in mchain.iterate(start=start, rng=rng, maxIter=nIter, skipInitial=True):
        cnt[st] += 1
        
    if(counts_to_prob):
        return {i:cnt[i]/nIter for i in range(mchain.nStates)}
    else:
        return cnt



# if __name__=='__main__':
#     m = MarkovChain(5, 'equal')
#     res = estimate_steady_dist(m, nIter=1_000_000, counts_to_prob=True)
#     print(res)
    