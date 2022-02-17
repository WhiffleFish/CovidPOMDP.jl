"""
Action input to influence epidemic simulation dynamics
# Arguments
- `testing_prop::Real` - Proportion of population to be tested on one day
    - Simplification of typical "x-days between tests per person"  action strategy due to non agent-based model
"""
struct CovidAction
    testing_prop::Float64
end

Base.zero(CovidAction) = CovidAction(0.0)

struct InfParams{D<:Distribution}
    pos_test_probs::Vector{Float64}
    symptom_dist::D
    asymptomatic_prob::Float64
    Infdistributions::Vector{Gamma{Float64}}
end

"""
# Arguments
- `S::Int` - Current Susceptible Population
- `I::Vector{Int}` - Current Infected Population
- `R::Int` - Current Recovered Population
- `Tests::Matrix{Int}` - Array for which people belonging to array element ``T_{i,j}`` are ``i-1`` days away
    from receiving positive test and have infection age ``j``
"""
struct CovidState{D}
    S::Int # Current Susceptible Population
    I::Vector{Int} # Current Infected Population
    R::Int # Current Recovered Population
    Tests::Matrix{Int} # Rows: Days from receiving test result; Columns: Infection Age
    params::InfParams{D}
    prev_action::CovidAction
end


"""
# Arguments
- `test_delay::Int = 0` - Delay between test being administered and received by subject (days)
- `N::Int = 1_000_000` - Total Population
- `discount::Float64 = 0.95` - POMDP discount factor
"""
Base.@kwdef struct CovidPOMDP <: POMDP{CovidState, CovidAction, Int}
    test_delay::Int
    N::Int
    discount::Float64
    inf_loss::Float64
    test_loss::Float64
    testrate_loss::Float64
    test_period::Int
end

"""
Take given CovidPOMDP obj and return same CovidPOMDP obj only with test_period changed to 1
"""
function unity_test_period(pomdp::CovidPOMDP)::CovidPOMDP
    return CovidPOMDP(
        pomdp.test_delay,
        pomdp.N,
        pomdp.discount,
        pomdp.inf_loss,
        pomdp.test_loss,
        pomdp.testrate_loss,
        1
    )
end

function population(s::CovidState)
    return s.S + sum(s.I) + s.R
end

"""
Convert `CovidState` struct to Vector - Collapse infected array to sum
"""
function Base.Array(state::CovidState)::Vector{Int64}
    [state.S, sum(state.I), state.R]
end

function simplex_sample(N::Int, m::Float64, rng::AbstractRNG=Random.GLOBAL_RNG)
    v = rand(rng, N-1)*m
    push!(v, 0, m)
    sort!(v)
    return (v - circshift(v,1))[2:end]
end


function init_SIRT(pomdp::CovidPOMDP, rng=Random.GLOBAL_RNG)
    N = pomdp.N
    S, inf, R = floor.(Int, simplex_sample(3, Float64(N), rng))

    horizon = INFECTION_HORIZON

    I = floor.(Int,simplex_sample(horizon, Float64(inf), rng))

    leftover = N - (S + sum(I) + R)
    R += leftover

    tests = zeros(Int, pomdp.test_delay+1, horizon)

    return S, I, R, tests
end

"""
Random Initial State using Bayesian Bootstrap / Simplex sampling
# Arguments
- `params::CovidPOMDP` - Simulation parameters
"""
function CovidState(params::CovidPOMDP, rng=Random.GLOBAL_RNG)::CovidState
    S, I, R, tests = init_SIRT(pomdp, rng)
    return CovidState(S, I, R, tests, CovidAction(0.0))
end

function initParams(pomdp::CovidPOMDP, asymptomatic_prob=0.10)
    infection_distributions = similar(INF_DIST)
    for (i,d) in enumerate(INF_DIST)
        k,θ = Distributions.params(d)
        k′ = k*rand()*2
        θ′ = θ*rand()*2
        infection_distributions[i] = Gamma(k′,θ′)
    end

    return InfParams(
        POS_TEST_PROBS,
        SYMPTOM_DIST,
        asymptomatic_prob,
        infection_distributions
        )
end

function rand_initialstate(pomdp::CovidPOMDP)
    S, I, R, T = init_SIRT(pomdp)
    params = initParams(pomdp)
    return CovidState(S, I, R, T, params, CovidAction(0.0))
end
