#=
NOTE: symptomatic_isolation_prob is not averaged
- isolation probability assumed to be known
=#
function mean_params(states::Vector{<:CovidState})
    N = length(states)
    s0 = first(states).params

    test_probs = zero(s0.pos_test_probs)
    symptom_probs = zero(s0.symptom_probs)
    infectiousness = Vector{Gamma{Float64}}(undef, length(s0.infectiousness))
    asymptomatic_prob = 0.0

    gamma_params = zeros(length(s0.infectiousness), 2)
    for s in states
        p = s.params
        test_probs .+= p.pos_test_probs
        symptom_probs .+= p.symptom_probs
        asymptomatic_prob += p.asymptomatic_prob
        for (i,d) in enumerate(p.infectiousness)
            gamma_params[i,:] .+= params(d)
        end
    end

    test_probs ./= N
    symptom_probs ./= N
    asymptomatic_prob /= N
    gamma_params ./= N
    symptomatic_isolation_prob = first(states).params.symptomatic_isolation_prob

    for i in eachindex(infectiousness)
        k, θ = @view gamma_params[i,:]
        infectiousness[i] = Gamma(k,θ)
    end

    return InfParams(
        test_probs,
        symptom_probs,
        asymptomatic_prob,
        symptomatic_isolation_prob,
        infectiousness
    )
end


Statistics.mean(states::Vector{<:CovidState}) = Statistics.mean(states, population(first(states)))

Statistics.mean(pc::ParticleCollection{<:CovidState}) = Statistics.mean(pc.particles)
