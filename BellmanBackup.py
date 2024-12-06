
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

 


########################
#  BACKUP OPERATOR/S
########################


# Compute Q^V(s,a), given mdp and gamma
#   ... The one-step backup of (s,a), based on given state-value-vector 'V'
def Q(mdp: MDP, V: list[float], s: int, a: int, gamma: float) -> float:    
    return sum((   p*(r + gamma*V[s])   for p,s,r in mdp.transitions[s,a]    ))


# The one-step backup at each state-action pair of the given MDP, based on state-values 'V'
# Returns a list 'Q' such that, for each state 's' Q[s] is a dict {a: Q^V(s,a)}
def Q_all(mdp: MDP, V: list[float], gamma: float):
    ret = []
    for s in range(mdp.nStates):
        ret.append({ a : Q(mdp=mdp, V=V, s=s, a=a, gamma=gamma)   for a in mdp.possibleActions(s)})
    return ret
    

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




###################
#  ALGORITHMS
###################


# estimate the state-values of the given mdp for the given policy, under d.f.=gamma,
#   ... by repeatedly applying 'B_pi' operator on a random vector
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
            return VV
        V = VV
        iterCnt += 100
    
    return V, iterCnt