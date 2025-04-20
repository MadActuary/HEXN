# HEXN
Health Experience Node (HEXN). 
The logo is a red hair young witch looking into a crystal ball. 

Given a chain of states. e.g. an "active", "disabled" and "dead" state. And intensities binding these states together. Intensities is the instant probability for jumping from one state to another. This intensity can at the current version include "age", "duration in current state" and "duration since a specific state". The project proposes transition probabilities and expected cash flows for various states given current states and durations. The cashflows are given based on a future monthly scedual. 

The durations are used based on the following: 
Duration since specific state is used to model the dependency to the first claim a policy have claimed. The duration since last transition is used to model some og the danish public social states fx. resourceforløb. The age is the time. 

The arcitecture of the project is a follows:
main()
- Engine
|  |
-  - Model
-  - Payoff

Engine
- getCashflow()

Model
- getTransitionPropabilities()
The method getTrantionPropabilities() simulates n paths and based on them predicts the probability of being in every one of the states in the following months. fx this function answers the question of a policy being in state "diability" in ten time steps in the future given the policy is in the state "active" at time = 0.  
- step()
- getCurrentState()
The method "step()" moves the internal process one step ahead in time. The internal state of the process can be seen by calling getCurrentState().


Payoff
- evaluate()
Payoff is an abstract class. The class carries all the logic related to the custom designed payments for the states of the chain.

