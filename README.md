
# Spiking neural networks: Julia vs Python

## Model description

In spiking neural networks, such as those present in biology and
and some neuromorphic hardware, communication between neurons takes
place by means of spikes.

Here I have focused on a simple implementation of a leaky integrate
and fire (LIF) neuron with the following characteristics:

- Neurons spike whenever their potential reaches a certain threshold
  value.

- Neurons are characterized by a refractory time after each spike. During
  that interval, neurons ignore any input.

- Spikes are approximated as impulse or Dirac's delta functions.

- Each connection is characterized by a certain synaptic delay.

A summary with the math will come. Stay tuned.


## A minimal implementation of a LIF network

The code shown in this example is a subset of a larger framework for
doing useful stuff with spiking neurons. However, it is
enough to explore the relative merits of Julia and Python for this
problem.

One frequent criticism that comes up when comparing two programming languages
is that some implementations may not take advantage of the characteristics and
strengths of a particular language. I think that's ok:
here I am coming from a math / theoretical
physics perspective, and focusing on an implementation
that is as close to the math as possible.

In my laptop, the Julia 1.0 test case takes around 5s
to run. The equivalent Python version (Python 3.7) takes around 3m30s. That's
faster than the first Python implementation that incorporated some additional
bookkeeping on the Python side, but Julia is still 40 times faster.

## Julia implementation

One of the things I like about Julia is its expressive type system. Here
are some basic custom types that I've used to implement the LIF network:

A LIF neuron:

```julia
mutable struct SpikingLIF
    tau :: Float64
    tref :: Float64
    v :: Float64
    v0 :: Float64
    tlast :: Float64
    inref :: Bool
end
```
Spikes and spike trains:

```julia
struct Spike
    t :: Float64
    w :: Float64
end

const SpikeTrain = Vector{Spike}
```

The neural network and a helper alias to codify the fan-out of
each neuron:

```julia
const FanOut = Vector{Int}

mutable struct SpNet
    neurons :: Vector{SpikingLIF}
    fanout :: Vector{FanOut}
    W :: Matrix{Float64}
    td :: Float64
end
```

In this simple implementation I haven't bothered to codify the synaptic
weights as a sparse matrix, and I am considering a single propagation delay.
Both generalizations are trivial. Also, bear in mind that
there is no matrix multiplication involved, the matrix is used just
to store the synaptic weights.

We complete our minimal implementation
 with a mutable struct that codifies the relevant data
required to run a simulation:

```julia
mutable struct SpikeSim
    N :: Int
    spikes :: Vector{SpikeTrain}
    time :: Float64
    dt :: Float64
end
```

For every neuron, we keep track of the train of incoming spikes. The evolution
of this network with time comprises three different steps:

1) Determining which spikes are coming
2) Evolving each neuron
3) Determining which spikes are being generated and updating the spike trains

This is implemented in the following function:

```julia
function next_step!(spn::SpNet, spsim::SpikeSim, vin::Vector{Float64})
    spiking = []
    spout = zeros(spsim.N)
    for i=1:spsim.N
        spout[i] = next_step!(spn.neurons[i], spsim.time, spsim.dt, vin[i],
            spsim.spikes[i])
        if spout[i] > 0.5
            push!(spiking, i)
        end
    end
    stime = spsim.time + spn.td
    for isp in spiking
        for j in spn.fanout[isp]
            Wji = spn.W[j,isp]
            push!(spsim.spikes[j], Spike(stime, Wji))
        end
    end
    spsim.time += spsim.dt
    return spout
end
```

This function in turn sends all the information to each neuron:

```julia
function next_step!(sn::SpikingLIF, time::Float64, dt::Float64,
        vext::Float64, spt::SpikeTrain)

    vne = 0.0
    while length(spt) > 0 && spt[1].t < time + dt
        spike = popfirst!(spt)
        vne += spike.w
    end
    return next_step!(sn, dt, vext, vne)
end
```

And the lowest level function takes care of the discretization of the ODE and
determining whether the spiking condition is met:

```julia
function next_step!(sn::SpikingLIF, dt::Float64, vin::Float64, vne::Float64)
    if sn.inref
        if sn.tlast >= sn.tref
            sn.tlast = 0.0
            sn.inref = false
        else
            sn.tlast += dt
        end
        return 0
    else
        sn.v = (sn.tau*sn.v + vin*dt + vne)/(sn.tau + dt)
        if sn.v >= sn.v0
            sn.v = 0
            sn.inref = true
            sn.tlast = 0.0
            return 1
        else
            return 0
        end
    end
end
```

You can simplify this and incorporate all the functionality within a single
method, but I found this way of breaking down the problem easy to understand.

## Python implementation

The implementation in Python is very similar, except that it takes a more
OO-centric approach.

I define two classes:

```python
class SpikingNeuron:

    def __init__(self, tau, tref=1, dt=0.01):
        self.tau = tau
        self.tref = 1
        self.v = 0
        self.v0 = 1.0
        self.tlast = 0.0
        self.inref = False
```

and

```python
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
        self.activity = []
```

Other than that, the implementation mirrors that used in Julia.
