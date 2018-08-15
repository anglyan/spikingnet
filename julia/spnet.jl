mutable struct SpikingLIF
    tau :: Float64
    tref :: Float64
    v :: Float64
    v0 :: Float64
    tlast :: Float64
    inref :: Bool
end

SpikingLIF(tau :: Float64, tref:: Float64, v0 :: Float64) =
    SpikingLIF(tau, tref, 0.0, v0, 0.0, false)

struct Spike
    t :: Float64
    w :: Float64
end

const SpikeTrain = Vector{Spike}

const FanOut = Vector{Int}

mutable struct SpNet
    neurons :: Vector{SpikingLIF}
    fanout :: Vector{FanOut}
    W :: Matrix{Float64}
    td :: Float64
end

mutable struct SpikeSim
    N :: Int
    spikes :: Vector{SpikeTrain}
    time :: Float64
    dt :: Float64
end


SpikeSim(N::Int, dt::Float64) =
    SpikeSim(N, [SpikeTrain() for i=1:N], 0.0, dt)


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


function next_step!(sn::SpikingLIF, time::Float64, dt::Float64,
        vext::Float64, spt::SpikeTrain)

    vne = 0.0
    while length(spt) > 0 && spt[1].t < time + dt
        spike = popfirst!(spt)
        vne += spike.w
    end
    return next_step!(sn, dt, vext, vne)
end


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

function create_random(N::Int, p::Float64)
    flist = [FanOut() for i=1:N]
    for i = 1:N
        for j=1:N
            if i == j
                continue
            else
                if rand() < p
                    push!(flist[i],j)
                end
            end
        end
    end
    return flist
end
