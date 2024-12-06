
import numpy as np
from MDP import MDP


############################
#  MATH HELPER-FUNCTIONS
############################


# the vector difference, V1-V2
# make sure that the vectors are of equal length
def vec_diff(V1: list[float], V2: list[float]):
    return [V1[i] - V2[i] for i in range(len(V1))]


# The l-infinite norm of a vector 'V'
def l_inf(V: list[float]):
    return max((  abs(val) for val in V  ))

 


######################
#  BACKUP OPERATORS
######################


# Compute Q^V(s,a), given mdp and gamma
#   ... The one-step backup of (s,a), based on given state-value-vector 'V'
def Q(mdp: MDP, V: list[float], s: int, a: int, gamma: float) -> float:    
    return sum((   p*(r + gamma*V[s])   for p,s,r in mdp.transitions[s,a]    ))


# The one-step backup of the given state 's' for each possible action, based on state-values 'V'
# Returns a dict 'Q': {a: Q^V(s,a)}, for all possible actions 'a' in state 's'
def Q_allActions(mdp: MDP, V: list[float], s: int, gamma: float) -> dict[int, float]:
    return { a : Q(mdp=mdp, V=V, s=s, a=a, gamma=gamma)   for a in mdp.possibleActions(s)  }

    
# The one-step backup at each state-action pair of the given MDP, based on state-values 'V'
# Returns a list 'Q' such that, for each state 's' Q[s] is a dict {a: Q^V(s,a)}
def Q_allStates(mdp: MDP, V: list[float], gamma: float) -> list[dict[int,float]]:
    ret = []
    for s in range(mdp.nStates):
        ret.append({ a : Q(mdp=mdp, V=V, s=s, a=a, gamma=gamma)   for a in mdp.possibleActions(s)})
    return ret
    

# Get the max one-step backup (over all actions) of the vector 'V' at state s
# Return a tuple (qval, a) where 'qval' 
#   ... is the maximum value over all actions of Q^V(s,a), occuring at action 'a'
def Qmax(mdp: MDP, V: list[float], s, gamma: float) -> [float, int]:
    qvals = Q_allActions(mdp, V, s, gamma)
    bigA = max(qvals, key=qvals.get)
    bigQ = qvals[bigA]
    return bigQ, bigA


# B_pi: The bellman backup operator
# Returns the one-step backup of the given vector, for each state 's' and its action policy[s]
#   'V'        :  the n-element vector to back up
#   'policy'   :  a dict such that policy[s] is the action to be taken in state s
#   'gamma'    :  the discount factor applicable for this computation
def Bpi(mdp: MDP, V: list[float], policy: dict[int,float], gamma: float) -> list[float]:
    return [  Q(mdp=mdp, V=V, s=s, a=policy[s], gamma=gamma)    for s in range(mdp.nStates)  ]


# Computes [B_pi ** k][V], the composition of the B_pi operator with itself 'k' times
#       i.e., return B_pi[B_pi[B_pi[... V]]]
def Bpi_k(mdp: MDP, k: int, V: list[float], policy: dict[int,float], gamma: float) -> list[float]:
    for i in range(k):
        V = Bpi(mdp=mdp, V=V, policy=policy, gamma=gamma)
    return V


# The bellman optimality operator
# Returns the one-step backup of the given vector, for each state 's' and maximized over all its possible actions
def B(mdp: MDP, V: list[float], gamma: float) -> list[float]:
    return [Qmax(mdp, V, s, gamma)[0] for s in range(mdp.nStates)]


# Computes [B ** k][V], the composition of the Bellman optimality operator with itself 'k' times
#       i.e., return B[B[B[... V]]]
def B_k(mdp: MDP, k: int, V: list[float], gamma: float) -> list[float]:
    for i in range(k):
        V = B(mdp, V, gamma)
    return V


# Compute the greedy policy for the state-value vector 'V'
# For each state, identify the action that gets the maximum backup value from V
# To compute the optimal policy 'pi_star', call this method using V=v_star (the optimal state-value vector)
# Returns the policy as a dict: s -> a, mapping each state to its greedy action
def pi_greedy(mdp, V, gamma):
    return {s: Qmax(mdp=mdp, V=V, s=s, gamma=gamma)[1] for s in range(mdp.nStates)}




#################
#  ALGORITHMS
#################


# estimate the state-values of the given mdp for the given policy, under d.f.=gamma,
#   ... by repeatedly applying the 'B_pi' operator on a random vector
# computes B_pi[V] iteratively, upto  'maxIter' times
# stops when adjacent computations differ by less than 10**(-thresh), in all the elements
# set thresh=None to not consider a threshold, running till 'maxIter' is exhausted
def estimate_V_pi(mdp, policy, gamma, thresh=4, maxIter=10_000):
    limit = (10**-thresh) if(thresh is not None) else 0.
    V = [0.] * mdp.nStates
    iterCnt = 0
    while(iterCnt < maxIter):
        Bpi_k(mdp=mdp, k=99, V=V, policy=policy, gamma=gamma)
        VV = Bpi(mdp=mdp, V=V, policy=policy, gamma=gamma)
        # compute the max element of |VV - V|,   a.k.a ||VV - V||_inf
        diff = l_inf(vec_diff(VV,V))
        if(diff < limit):
            return VV, iterCnt
        V = VV
        iterCnt += 100
    
    return V, iterCnt


# estimate the optimal state-values of the given mdp, under d.f.=gamma,
#   ... by repeatedly applying the Bellman Optimality operator (B) on a random vector
# computes B[V] iteratively, upto  'maxIter' times
# stops when adjacent computations differ by less than 10**(-thresh), in all the elements
# set thresh=None to not consider a threshold, running till 'maxIter' is exhausted
def estimate_V_star(mdp, gamma, thresh=4, maxIter=10_000):
    limit = (10**-thresh) if(thresh is not None) else 0.
    V = [0.] * mdp.nStates
    iterCnt = 0
    while(iterCnt < maxIter):
        B_k(mdp=mdp, k=99, V=V, gamma=gamma)
        VV = B(mdp=mdp, V=V, gamma=gamma)
        # compute the max element of |VV - V|,   a.k.a ||VV - V||_inf
        diff = l_inf(vec_diff(VV,V))
        if(diff < limit):
            return VV, iterCnt
        V = VV
        iterCnt += 100
    
    return V, iterCnt






