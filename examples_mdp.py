
from MDP import MDP

##################
#   Trivial MDP
##################
# w/ values off the top of my head
trivial_mdp = MDP(states=2, actions=2)
trivial_mdp.setStateActionPairs({
                                     0: {0,1},
                                     1: {0}
                                })

trivial_mdp.setTransitions({
                                (0,0) : [(.9,0,0.), (.1,0,10.)],
                                (0,1) : [(.5,0,-10.), (.5,1,10.)],
                                (1,0) : [(.3,0,-10.), (.7,1,10.)]
                           })



##################
#   Simple MDP
##################
# w/ values off the top of my head
simple_mdp = MDP(states=4, actions=3)

simple_mdp.setStateActionPairs({ 
                                 0: {0,1,2},
                                 1: {0,2},
                                 2: {1,2},
                                 3: {0}
                               })

simple_mdp.setTransitions({    
                                (0,0)  :  [(.2,0,0.), (.1,1,2.), (.3,1,.5), (.15,2,1.), (.25,2,0.)],
                                (0,1)  :  [(.2,1,0.), (.2,1,1.), (.2,2,0.), (.2,2,1.), (.2,2,2.) ],
                                (0,2)  :  [(.4,0,0.), (.3,2,0.), (.3,3,0.)],
                                (1,0)  :  [(.5,0,1.), (.5,2,-1.)],
                                (1,2)  :  [(.3,0,-1.), (.4,1,0.), (.3,2,1.)],
                                (2,1)  :  [(.4,0,0.), (.4,1,0.), (.2,3,0.) ],
                                (2,2)  :  [(.4,0,-2.), (.1,1,10.), (.5,3,0.)],
                                (3,0)  :  [(.6,0,8.), (.2,1,4.), (.2,2,4.)]
                          })



#####################
#   HomeWork1 MDP
#####################
# from hw1 of course RL taken in CMI-DS sem3
hw1_mdp = MDP(states=3, actions=2, state_action_pairs='all')
hw1_mdp.setTransitions({
                                (0,0) : [(1.,0,1.)],
                                (0,1) : [(.5,1,2.), (.5,2,2.)],
                                (1,0) : [(1.,1,0.)],
                                (1,1) : [(.3,0,3.), (.7,2,3.)],
                                (2,0) : [(1.,2,1.)],
                                (2,1) : [(.1,0,4.), (.9,1,4.)]
                           })



#################
#   Robot MDP
#################
# from Sutton & Barto - The Soda can collector robot
robot_mdp = MDP(states=["battery-high", "battery-low"], actions=["search", "wait", "recharge"])
robot_mdp.setStateActionPairs({
                                   0: {0,1},
                                   1: {0,1,2}
                              })
robot_mdp.setTransitions({
                                (0,0) : [(.8,0,1.), (.2,1,1.)],
                                (0,1) : [(1.,0,0.1)],
                                (1,0) : [(.3,0,-.2), (.7,1,-.2)],
                                (1,1) : [(1.,1,.1)],
                                (1,2) : [(1.,0,0.)]
                           })




s, t, h, r = simple_mdp, trivial_mdp, hw1_mdp, robot_mdp


