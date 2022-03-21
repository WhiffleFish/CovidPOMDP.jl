function incident_infections(p1::InfParams, p2::InfParams, S::Int, I1::Vector{Int}, I2::Vector{Int}, R::Int)
    iszero(S) && return 0
    infSum1 = 0
    infSum2 = 0
    N = S + sum(I1) + sum(I2) + R
    susceptible_prop = S/N

    for (i, inf) in enumerate(I1)
        d = p1.infectiousness[i]
        k,θ = Distributions.params(d)
        d′ = Gamma(k*susceptible_prop, θ) # conversion from R₀ to Rₜ
        infSum1 += rand(RVsum(d′, inf))
    end

    for (i, inf) in enumerate(I2)
        d = p2.infectiousness[i]
        k,θ = Distributions.params(d)
        d′ = Gamma(k*susceptible_prop, θ) # conversion from R₀ to Rₜ
        infSum2 += rand(RVsum(d′, inf))
    end

    infSum1 = floor(Int, infSum1)
    infSum2 = floor(Int, infSum2)
    inf_sum = infSum1 + infSum2
    overage = inf_sum - S

    if overage > 0
        half_over = div(overage, 2)
        remove1 = min(infSum1, half_over)
        infSum1 -= remove1
        overage -= remove1
        infSum2 -= overage
    end

    return infSum1, infSum2
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


function CovidPOMDPs.sim_step(pomdp::DoubleCovidPOMDP, state::DoubleCovidState, a::CovidAction)
    (;S, I1, I2, R, T1, T2, params, prev_action) = state
    I1 = copy(I1)
    I2 = copy(I2)
    T1 = copy(T1)
    T2 = copy(T2)

    # Update symptomatic and testing-based isolations
    I1, R, T1, pos_tests1 = update_isolations(pomdp.params, I1, R, T1, a)
    I2, R, T2, pos_tests2 = update_isolations(params, I2, R, T2, a)

    # Incident Infections
    R += I1[end]
    R += I2[end]
    I1 = circshift(I1, 1) # shift_inf!(I1)
    I2 = circshift(I2, 1) # shift_inf!(I2)

    inf1, inf2 = incident_infections(pomdp.params, params, S, I1, I2, R)
    I1[1] = inf1
    I2[1] = inf2

    S -= (inf1 + inf2)
    sp = DoubleCovidState(S, I1, I2, R, T1, T2, params, a)

    return sp, inf1+inf2, pos_tests1 .+ pos_tests2
end
