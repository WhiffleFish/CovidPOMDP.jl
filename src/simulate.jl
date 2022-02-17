"""
Simulation History
# Arguments
- `sus::Vector{Int}` - Susceptible Population History
- `inf::Vector{Int}` - Infected Population History
- `rec::Vector{Int}` - Recovered Population History
- `N::Int` - Total Population
- `T::Int` - Simulation Time
- `incident::Array{Int, 1}` = nothing - Incident Infections need not always be recorded
"""
Base.@kwdef struct SimHist
    sus::Vector{Int} # Susceptible Population History
    inf::Vector{Int} # Infected Population History
    rec::Vector{Int} # Recovered Population History
    N::Int # Total Population
    T::Int # Simulation Time
    pos_test::Vector{Int} = Int[]
    actions::Vector{CovidAction} = CovidAction[]
    rewards::Vector{Float64} = Float64[]
    beliefs::Vector{ParticleCollection{CovidState}} = ParticleCollection{CovidState}[]
end


"""
Convert `simHist` struct to 2-dimentional array - Collapse infected array to sum
"""
function Base.Array(simHist::SimHist)::Array{Int64,2}
    hcat(simHist.sus, simHist.inf, simHist.rec) |> transpose |> Array
end

"""
# Arguments
- `state::CovidState` - Current Sim State
- `pomdp::CovidPOMDP` - Simulation parameters
"""
function incident_infections(params::CovidPOMDP, S::Int, I::Vector{Int}, R::Int)
    infSum = 0
    for (i, inf) in enumerate(I)
        d = params.Infdistributions[i]
        k,θ = Distributions.params(d)
        d′ = Gamma(k*S/params.N, θ) # conversion from R₀ to Rₜ
        infSum += rand(RVsum(d′, inf))
    end

    return min(S, floor(Int, infSum)) # Can't infect more people than are susceptible
end


"""
# Arguments
- `I::Array{Int,1}` - Current infectious population vector (divided by infection age)
- `pomdp::CovidPOMDP` - Simulation parameters
"""
function symptomatic_isolation(params::InfParams, I::Vector{Int})::Vector{Int64}
    isolating = zero(I)
    for (i, inf) in enumerate(I)
        symptomatic_prob = cdf(params.symptom_dist,i) - cdf(params.symptom_dist,i-1)
        isolation_prob = symptomatic_prob*(1-params.asymptomatic_prob)*params.symptomatic_isolation_prob
        isolating[i] = rand(Binomial(inf,isolation_prob))
    end
    return isolating
end


"""
# Arguments
- `state::CovidState` - Current Sim State
- `pomdp::CovidPOMDP` - Simulation parameters
- `action::CovidAction` - Current Sim Action
# Returns
- `pos_tests::Vector{Int}` - Vector of positive tests stratified by infection age
"""
function positive_tests(params::InfParams, I::Vector{Int}, tests::Matrix{Int}, a::CovidAction)
    pos_tests = zeros(Int, length(params.pos_test_probs))

    for (i, inf) in enumerate(I)
        num_already_tested = sum(@view tests[:,i])
        num_tested = floor(Int,(inf-num_already_tested)*a.testing_prop)
        pos_tests[i] = rand(Binomial(num_tested,params.pos_test_probs[i]))
    end
    return pos_tests
end



# Only record number that have taken the test, the number that return postive is
# Binomial dist, such that s' is stochastic on s.
function update_isolations(params::InfParams, I, R, tests, a::CovidAction)

    sympt = symptomatic_isolation(params, I) # Number of people isolating due to symptoms
    pos_tests = positive_tests(params, I, tests, a)

    sympt_prop = sympt ./ I # Symptomatic Isolation Proportion
    replace!(sympt_prop, NaN=>0.0)

    R += sum(sympt)
    I -= sympt

    tests[end,:] .= pos_tests

    @. tests = floor(Int, (1 - sympt_prop)' * tests)

    R += sum(@view tests[1,:])
    @views I .-= tests[1,:]

    @assert all(≥(0), I)

    # Progress testing state forward
    # People k days from receiving test back are now k-1 days from receiving test
    # Tested individuals with infection age t move to infection age t + 1
    tests = circshift(tests,(-1,1))

    # Tests and infection ages do not roll back to beginning; clear last row and first column
    tests[:,1] .= 0
    tests[end,:] .= 0

    return I, R, tests, pos_tests
end


function sim_step(pomdp::CovidPOMDP, state::CovidState, a::CovidAction)
    (;S, I, R, Tests, params, prev_action) = state

    # Update symptomatic and testing-based isolations
    I, R, Tests, pos_tests = update_isolations(pomdp, I, R, Tests, a)

    # Incident Infections
    R += I[end]
    I = circshift(I, 1)
    new_infections = incident_infections(params, S, I, R)
    I[1] = new_infections
    S -= new_infections
    sp = CovidState(S, I, R, Tests, params, a)
    return sp, new_infections, pos_tests
end

function reward(m::CovidPOMDP, s::CovidState, a::CovidAction, sp::CovidState)
    inf_loss = m.inf_loss*sum(sp.I)/m.N
    test_loss = m.test_loss*a.testing_prop
    testrate_loss = m.testrate_loss*abs(a.testing_prop-s.prev_action.testing_prop)
    return -(inf_loss + test_loss + testrate_loss)
end

function continuous_gen(m::CovidPOMDP, s::CovidState, a::CovidAction, rng::AbstractRNG=Random.GLOBAL_RNG)
    sp, new_inf, o = sim_step(m, s, a)
    r = reward(m, s, a, sp)
    o = sum(o)

    return (sp=sp, o=o, r=r)
end

"""
# Arguments
- `T::Int` - Simulation duration (days)
- `state::CovidState` - Current Sim State
- `pomdp::CovidPOMDP` - Simulation parameters
- `action::CovidAction` - Current Sim Action
"""
function simulate(T::Int, state::CovidState, pomdp::CovidPOMDP, action::CovidAction)::SimHist
    susHist = zeros(Int,T)
    infHist = zeros(Int,T)
    recHist = zeros(Int,T)
    testHist = zeros(Int,T)
    actionHist = fill(action, T)
    rewardHist = zeros(Float64, T)

    for day in 1:T
        susHist[day] = state.S
        infHist[day] = sum(state.I)
        recHist[day] = state.R

        sp, new_infections, pos_tests = sim_step(pomdp, state, action)
        r = reward(pomdp, state, action,sp)

        testHist[day] = sum(pos_tests)
        rewardHist[day] = r
        state = sp
    end

    return SimHist(
        susHist,
        infHist,
        recHist,
        pomdp.N,
        T,
        testHist,
        actionHist,
        rewardHist,
        ParticleCollection{CovidState}[]
    )
end


function SimulateEnsemble(T::Int64, trajectories::Int64, pomdp::CovidPOMDP, action::CovidAction)
    [Simulate(T, CovidState(pomdp), pomdp, action) for _ in 1:trajectories]
end

function SimulateEnsemble(T::Int64, trajectories::Int64, pomdp::CovidPOMDP, actions::Vector{CovidAction})
    [Simulate(T, CovidState(pomdp), pomdp, actions[i]) for i in 1:trajectories]
end

function FullArr(state::CovidState, param::CovidPOMDP)::Vector{Float64}
    vcat(state.S,state.I,state.R)./param.N
end

function FullArrToSIR(arr::Array{Float64,2})::Matrix{Float64}
    hcat(
        view(arr,1,:),
        reshape(sum(view(arr,2:15,:),dims=1), size(arr,2)),
        view(arr,16,:)
    )'
end

function SimulateFull(T::Int, state::CovidState, pomdp::CovidPOMDP; action::CovidAction=CovidAction(0.0))::Matrix{Float64}
    StateArr = Array{Float64,2}(undef,16,T)
    StateArr[:,1] = FullArr(s)
    for day in 2:T
        StateArr[:,day] = FullArr(first(sim_step(pomdp, s, action)))
    end
    return StateArr
end


"""
Convert Simulation SimulateEnsemble output to 3D Array
# Arguments
- `histvec::Vector{SimHist}` - Vector of SimHist structs
"""
function Base.Array(histvec::Vector{SimHist})::Array{Int64,3}
    arr = zeros(Int64, 3, histvec[1].T, length(histvec))
    for i in eachindex(histvec)
        arr[:,:,i] .= Array(histvec[i])
    end
    return arr
end
