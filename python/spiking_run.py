import spikinglif as sn
import random
import numpy as np


def create_random(N, p):
    netw = {i:[] for i in range(N)}
    for i in range(N):
        for j in range(N):
            if i==j:
                continue
            else:
                if random.random() < p:
                    netw[j].append(i)
    return netw


nt =create_random(1000,0.05)
wd = {}
for k, v in nt.items():
    for i in v:
        wd[i,k] = 0.2

activities = []

vext = 0.8+0.4*np.random.normal(size=1000)

spn = sn.SpikingNetwork(1000, nt, wd, 8.0)
for i in range(100000):
    output = spn.next_step(vext)
    act = [o for o in output if o > 0.5]
    for a in act:
        activities.append((i,a))

print(len(activities))
