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

Base.string(a::CovidAction) = string(round(a.testing_prop,sigdigits=2))

struct CovidActionSpace end

Base.rand(rng::AbstractRNG, ::CovidActionSpace) = CovidAction(rand(rng))
Base.rand(A::CovidActionSpace) = rand(Random.GLOBAL_RNG, A)

function CovidActionSpace(n::Int; zero::Bool=false)
    start = zero ? 0 : 1
    return Tuple(CovidAction(i/n) for i in start:n)
end

function CovidActionSpace(v::AbstractVector{<:Real})
    return Tuple(CovidAction(v_i) for v_i in v)
end

struct InfParams
    pos_test_probs::Vector{Float64}
    symptom_probs::Vector{Float64}
    asymptomatic_prob::Float64
    symptomatic_isolation_prob::Float64
    infectiousness::Vector{Gamma{Float64}}
end

const DEFAULT_PARAMS = InfParams(
    POS_TEST_PROBS,
    SYMPTOM_PROBS,
    ASYMPTOMATIC_PROB,
    SYMPTOMATIC_ISOLATION_PROB,
    INF_DIST
)

"""
# Arguments
- `S::Int` - Current Susceptible Population
- `I::Vector{Int}` - Current Infected Population
- `R::Int` - Current Recovered Population
- `Tests::Matrix{Int}` - Array for which people belonging to array element ``T_{i,j}`` are ``i-1`` days away
    from receiving positive test and have infection age ``j``
"""
struct CovidState
    S::Int # Current Susceptible Population
    I::Vector{Int} # Current Infected Population
    R::Int # Current Recovered Population
    Tests::Matrix{Int} # Rows: Days from receiving test result; Columns: Infection Age
    params::InfParams
    prev_action::CovidAction
end

SIR(s::CovidState) = (s.S, sum(s.I), s.R)

Base.@kwdef struct CovidPOMDP{A} <: POMDP{CovidState, CovidAction, Int}
    "Delay (in days) between test being administered and result of test being received `(≥ 0)`"
    test_delay::Int = 1

    "Total population count `(> 0)`"
    N::Int = 10^6

    "POMDP discount factor `(γ ∈ [0,1])`"
    discount::Float64 = 0.95

    "Weight with which to penalize new infections `(≥ 0.0)`"
    inf_loss::Float64 = 1.0

    "Weight with which to penalize testing rate `(≥ 0.0)`"
    test_loss::Float64 = 1.0

    "Weight with which to penalize changes in testing rate `(≥ 0.0)`"
    testrate_loss::Float64 = 1.0

    "Number of days for which a testing policy must be held `(≥ 1)`"
    test_period::Int = 1

    "CovidPOMDP action space (default is continuous [0,1])"
    actions::A = CovidActionSpace()
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
        1,
        pomdp.actions
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


function init_SIRT(pomdp::CovidPOMDP, rng::AbstractRNG=Random.GLOBAL_RNG)
    N = pomdp.N
    S, inf, R = floor.(Int, simplex_sample(3, Float64(N), rng))

    horizon = INFECTION_HORIZON

    I = floor.(Int,simplex_sample(horizon, Float64(inf), rng))

    leftover = N - (S + sum(I) + R)
    R += leftover

    tests = zeros(Int, pomdp.test_delay+1, horizon)

    return S, I, R, tests
end

function init_SIRT(pomdp::CovidPOMDP, inf::Int, rng::AbstractRNG=Random.GLOBAL_RNG)
    N = pomdp.N
    R = 0

    horizon = INFECTION_HORIZON

    I = floor.(Int, simplex_sample(horizon, Float64(inf), rng))

    leftover = N - sum(I)
    S = leftover

    tests = zeros(Int, pomdp.test_delay+1, horizon)

    return S, I, R, tests
end

function CovidState(pomdp::CovidPOMDP, params::InfParams=DEFAULT_PARAMS)
    return CovidState(init_SIRT(pomdp)..., params, CovidAction(0.0))
end

function initParams(pomdp::CovidPOMDP, asymptomatic_prob=0.10)
    infection_distributions = similar(INF_DIST)
    pos_test_probs = copy(POS_TEST_PROBS)
    for (i,d) in enumerate(INF_DIST)
        k,θ = Distributions.params(d)
        # k′ = k*rand()*2
        # θ′ = θ*rand()*2
        k′ = k*(rand() + 0.5)
        θ′ = θ*(rand() + 0.5)
        infection_distributions[i] = Gamma(k′,θ′)

        λ = (k′*θ′) / (k*θ)
        pos_test_probs[i] *= λ
    end

    return InfParams(
        min.(pos_test_probs, 1.0),
        cdf_step(SYMPTOM_DIST, INFECTION_HORIZON),
        asymptomatic_prob,
        SYMPTOMATIC_ISOLATION_PROB,
        infection_distributions
        )
end

function rand_initialstate(pomdp::CovidPOMDP)
    S, I, R, T = init_SIRT(pomdp)
    params = initParams(pomdp)
    return CovidState(S, I, R, T, params, CovidAction(0.0))
end

function rand_initialstate(pomdp::CovidPOMDP, Idist::Distribution)
    inf = clamp(floor(Int, rand(Idist)), 0, pomdp.N)
    S, I, R, T = init_SIRT(pomdp, inf)
    params = initParams(pomdp)
    return CovidState(S, I, R, T, params, CovidAction(0.0))
end

function cdf_step(d::Distribution, N::Int)
    return Float64[cdf(d, i) - cdf(d, i-1) for i in 1:N]
end
