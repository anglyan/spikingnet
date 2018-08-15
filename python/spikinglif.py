"""Implementation of a LIF model
"""

from collections import deque
import numpy as np


class SpikingNeuron:

    def __init__(self, tau, tref=1, dt=0.01):
        self.tau = tau
        self.tref = 1
        self.v = 0
        self.v0 = 1.0
        self.tlast = 0.0
        self.inref = False


    def next_step(self, dt, vext, vneigh):

        if self.inref:
            if self.tlast >= self.tref:
                self.tlast = 0
                self.inref = False
                return 0
            else:
                self.tlast += dt
        else:
            self.v = (self.tau*self.v + vext*dt+vneigh)/(self.tau+dt)
            if self.v >= self.v0:
                self.v = 0
                self.inref = True
                return 1
            else:
                return 0


class SpikingNetwork:

    def __init__(self, N, Wd, Wnn, tau, tref=1, tdelay=1, dt=0.01):
        self.N = N
        self.tau = tau
        self.tref = tref
        self.tdelay = tdelay
        self.dt = dt
        self.time = 0.0
        self.t0 = 0.0
        self.neurons = [SpikingNeuron(self.tau, self.tref, self.dt) for
            i in range(self.N)]
        self.tlist = [deque() for i in range(self.N)]

        self.Wd = Wd
        self.Wnn = Wnn

    def next_step(self, vext):

        spout = np.zeros(self.N)
        spiking = []
        for i in range(self.N):
            vneigh = 0.0
            while len(self.tlist[i]) > 0:
                tnext, w = self.tlist[i][0]
                if tnext <= self.t0:
                    self.tlist[i].popleft()
                    vneigh += w
                else:
                    break
            spout[i] = self.neurons[i].next_step(self.dt, vext[i], vneigh)
            if spout[i] > 0.5:
                spiking.append(i)
        tspike = self.t0 + self.tdelay
        for i in spiking:
            for j in self.Wd[i]:
                self.tlist[j].append((tspike, self.Wnn[j,i]))
        self.t0 += self.dt
        return spout
