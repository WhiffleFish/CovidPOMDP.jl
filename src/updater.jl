function POMDPs.initialstate(pomdp::CovidPOMDP)
    return ImplicitDistribution() do rng
        CovidState(pomdp, rng)
    end
end

function mean_params(states::Vector{CovidState{D}}) where D
    N = length(states)
    s0 = first(states).params

    test_probs = zero(s0.pos_test_probs)
    symptom_params = zeros(length(params(s0.symptom_dist)))
    Infdistributions = Vector{Gamma{Float64}}(undef, length(s0.Infdistributions))
    asymptomatic_prob = 0.0

    gamma_params = zeros(length(s0.Infdistributions), 2)
    for s in states
        p = s.params
        test_probs .+= p.pos_test_probs
        symptom_params .+= params(p.symptom_dist)
        asymptomatic_prob += p.asymptomatic_prob
        for (i,d) in enumerate(p.Infdistributions)
            gamma_params[i,:] .+= params(d)
        end
    end

    test_probs ./= N
    symptom_params ./= N
    symptom_dist = D(symptom_params...)
    asymptomatic_prob /= N
    gamma_params ./= N

    for i in eachindex(Infdistributions)
        k, θ = @view gamma_params[i,:]
        Infdistributions[i] = Gamma(k,θ)
    end

    return InfParams(test_probs, symptom_dist, asymptomatic_prob, Infdistributions)
end

function initialbelief(pomdp::CovidPOMDP, np::Int)
    ParticleCollection([rand(initialstate(pomdp)) for _ in 1:np])
end

function mean(states::Vector{CovidState}, N::Int)::CovidState
    n_states = length(states)
    sumS = 0
    sumI = zeros(Int,length(first(states).I))
    sumTests = zeros(Int,size(first(states).Tests))
    for s in states
        sumS += s.S
        sumI .+= s.I
        sumTests .+= s.Tests
    end
    avgS = floor(Int,sumS/n_states)
    avgI = floor.(Int, sumI./n_states)
    avgR = N - (avgS + sum(avgI))
    @assert avgR ≥ 0
    avgTests = round.(Int, sumTests./n_states)
    return CovidState(
        avgS,
        avgI,
        avgR,
        avgTests,
        mean_params(states)
        first(states).prev_action,
        )
end

mean(states::Vector{CovidState}) = mean(states, population(first(states)))

mean(pc::ParticleCollection{CovidState}, pomdp::CovidPOMDP) = mean(pc.particles, pomdp.N)
