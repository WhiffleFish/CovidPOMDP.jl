function incident_infections(params::InfParams, S::Int, I::Vector{Int}, R::Int)
    iszero(S) && return 0
    infSum = 0
    N = S + sum(I) + R
    for (i, inf) in enumerate(I)
        d = params.infectiousness[i]
        k,θ = Distributions.params(d)
        d′ = Gamma(k*S/N, θ) # conversion from R₀ to Rₜ
        infSum += rand(RVsum(d′, inf))
    end

    return min(S, floor(Int, infSum)) # Can't infect more people than are susceptible
end


function symptomatic_isolation(params::InfParams, I::Vector{Int})
    isolating = zero(I)
    for (i, inf) in enumerate(I)
        symptomatic_prob = params.symptom_probs[i]
        isolation_prob = symptomatic_prob*(1-params.asymptomatic_prob)*params.symptomatic_isolation_prob
        isolating[i] = rand(Binomial(inf,isolation_prob))
    end
    return isolating
end


function positive_tests(params::InfParams, I::Vector{Int}, T::Matrix{Int}, a::CovidAction)
    pos_tests = zeros(Int, length(params.pos_test_probs))

    for (i, inf) in enumerate(I)
        num_already_tested = sum(@view T[:,i])
        num_tested = floor(Int,(inf-num_already_tested)*a.testing_prop)
        pos_tests[i] = rand(Binomial(num_tested,params.pos_test_probs[i]))
    end
    return pos_tests
end


# Only record number that have taken the test, the number that return postive is
# Binomial dist, such that s' is stochastic on s.
function update_isolations(params::InfParams, I, R, T, a::CovidAction)

    sympt = symptomatic_isolation(params, I) # Number of people isolating due to symptoms
    pos_tests = positive_tests(params, I, T, a)

    sympt_prop = sympt ./ I # Symptomatic Isolation Proportion
    replace!(sympt_prop, NaN=>0.0)

    R += sum(sympt)
    I .-= sympt

    T[end,:] .= pos_tests

    @. T = floor(Int, (1 - sympt_prop)' * T)

    R += sum(@view T[1,:])
    @views I .-= T[1,:]

    @assert all(≥(0), I)

    CovidPOMDPs.shift_test!(T)

    return I, R, T, pos_tests
end


function CovidPOMDPs.sim_step(pomdp::SingleCovidPOMDP, state::SingleCovidState, a::CovidAction)
    (;S, I, R, T, params, prev_action) = state
    I = copy(I)
    T = copy(T)

    # Update symptomatic and testing-based isolations
    I, R, T, pos_tests = update_isolations(params, I, R, T, a)

    # Incident Infections
    R += I[end]
    I = circshift(I, 1) # shift_inf!(I)
    new_infections = incident_infections(params, S, I, R)
    I[1] = new_infections
    S -= new_infections
    sp = SingleCovidState(S, I, R, T, params, a)

    return sp, new_infections, pos_tests
end
