function Statistics.mean(states::Vector{DoubleCovidState}, N::Int)
    n_states = length(states)
    sumS = 0
    sumI1 = zeros(Int,length(first(states).I1))
    sumI2 = zeros(Int,length(first(states).I2))
    sumT1 = zeros(Int,size(first(states).T1))
    sumT2 = zeros(Int,size(first(states).T2))
    for s in states
        sumS += s.S
        sumI1 .+= s.I1
        sumI2 .+= s.I2
        sumT1 .+= s.T1
        sumT2 .+= s.T2
    end
    avgS = floor(Int,sumS/n_states)
    avgI1 = floor.(Int, sumI1./n_states)
    avgI2 = floor.(Int, sumI2./n_states)
    avgR = N - (avgS + sum(avgI1) + sum(avgI2))
    @assert avgR ≥ 0
    avgT1 = floor.(Int, sumT1./n_states)
    avgT2 = floor.(Int, sumT2./n_states)

    return DoubleCovidState(
        avgS,
        avgI1,
        avgI2,
        avgR,
        avgT1,
        avgT2,
        CovidPOMDPs.mean_params(states),
        first(states).prev_action,
        )
end
