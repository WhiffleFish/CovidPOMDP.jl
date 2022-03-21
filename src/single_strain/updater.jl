function Statistics.mean(states::Vector{SingleCovidState}, N::Int)
    n_states = length(states)
    sumS = 0
    sumI = zeros(Int,length(first(states).I))
    sumT = zeros(Int,size(first(states).T))
    for s in states
        sumS += s.S
        sumI .+= s.I
        sumT .+= s.T
    end
    avgS = floor(Int,sumS/n_states)
    avgI = floor.(Int, sumI./n_states)
    avgR = N - (avgS + sum(avgI))
    @assert avgR ≥ 0
    avgT = floor.(Int, sumT./n_states)

    return SingleCovidState(
        avgS,
        avgI,
        avgR,
        avgT,
        CovidPOMDPs.mean_params(states),
        first(states).prev_action,
        )
end
