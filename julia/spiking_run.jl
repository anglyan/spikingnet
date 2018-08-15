include("spnet.jl")

cm = create_random(1000, 0.05)
Wnn = zeros(1000,1000)
for i=1:1000
    for j in cm[i]
        Wnn[j,i] = 0.2
    end
end

nlist = [SpikingLIF(8.0, 1.0, 1.0) for i=1:1000]
snet = SpNet(nlist, cm, Wnn, 1.00)
spsim = SpikeSim(1000, 0.01)
vin = 0.8 .+ 0.4*randn(1000)
activity = []
for i=1:10000
    out = next_step!(snet, spsim, vin)
    act = [j for j=1:1000 if out[j] > 0.5]
    for a in act
#        println("$i $a")
        push!(activity, (i,a))
    end
end
println("$(length(activity))")
