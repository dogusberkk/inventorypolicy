import numpy as np
import matplotlib.pyplot as plt

class Inventory:
    def __init__(self,probability,minLevel,capacity,horizon,
                 price,hcost,bocost,varcost,fixcost,salvage,shortage):

        self.probability = probability
        self.capacity = capacity
        self.minLevel = minLevel
        self.horizon = horizon
        self.price = price
        self.bocost, self.varcost, self.hcost, self.fixedcost = bocost, varcost, hcost, fixcost
        self.salvage, self.shortage = salvage, shortage

        self.S = np.arange(self.minLevel,self.capacity+1)
        self.numStates = np.size(self.S)
        self.numActions = self.capacity - self.minLevel + 1
        self.A = np.array([a for a in range(self.numActions)])
        self.T = np.arange(self.horizon+1,0,-1)

    def oneStepReward(self,s,a):
        demand = np.array(list(self.probability.keys()))
        prob = np.array(list(self.probability.values()))
        expextedDemand = np.sum(demand * prob)
        if a == 0:
            isFixed = 0
        else:
            isFixed = 1

        r1 = (self.price * expextedDemand) - (self.fixedcost * isFixed) - (self.varcost * a)
        x = demand - (s + a)
        keys1 = x[x>0]
        y = (s+a) - demand
        keys2 = y[y>0]
        r2 = self.bocost * sum(keys1 * prob[-np.size(keys1):])
        r2 += self.hcost * sum(keys2 * prob[:np.size(keys2)])

        return r1 - r2

    def oneStepRewardTerminal(self,s):
        if s >= 0:
            return self.salvage * s
        else:
            return self.shortage * s

    def transition(self,j,s,a):
        if s + a < j:
            return 0
        else:
            if s+a-j in self.probability.keys():
                return self.probability[s+a-j]
            else:
                return 0



def bellmannEquaton(env,V,s,t):
    setOfActions = env.A[env.A<=env.capacity-s]
    V_values = np.zeros(np.size(setOfActions))
    for a in setOfActions:
        sum = 0
        for j in env.S:
            sum += env.transition(j,s,a)*V[t,np.where(env.S == j)]

        V_values[a] = env.oneStepReward(s,a) + sum

    return np.max(V_values), setOfActions[np.argmax(V_values)]


prob = {4:0.125,5:0.5,6:0.25,7:0.125}
minLevel = -10
capacity = 5
horizon = 10
price = 8
hcost = 1
bocost = 3
varcost = 2
fixedcost = 10
salvage = 0.5
shortage = 3
env = Inventory(prob,minLevel,capacity,horizon,price,hcost,bocost,varcost,fixedcost,salvage,shortage)


V = np.zeros((np.size(env.T),np.size(env.S)))
optimalActions = np.copy(V)


for index_t,t in enumerate(env.T):
    for index_s,s in enumerate(env.S):
        if index_t == 0:
            V[t-1,index_s] = env.oneStepRewardTerminal(s)
        else:
            V[t-1,index_s],optimalActions[t-1,index_s]  = bellmannEquaton(env,V,s,t)


plt.scatter(env.S,env.S+optimalActions[0,:],)
plt.grid()
plt.yticks(np.arange(np.min(env.S)-2,np.max(env.S)+2,1))
plt.xticks(np.arange(env.minLevel-2,env.capacity+2,1))

index = 0
for a,b in zip(env.S,env.S+optimalActions[0,:]):
    plt.text(a,b,int(V[0,index]),fontsize =7)
    index += 1

plt.xlabel("Inventory Level")
plt.ylabel("Up-to-order Level")
plt.show()












