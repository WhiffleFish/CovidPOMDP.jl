"""
# Arguments
- `S::Int` - Current Susceptible Population
- `I::Vector{Int}` - Current Infected Population
- `R::Int` - Current Recovered Population
- `T::Matrix{Int}` - Array for which people belonging to array element ``T_{i,j}`` are ``i-1`` days away
    from receiving positive test and have infection age ``j``
"""
struct SingleCovidState <: CovidPOMDPs.CovidState
    S::Int # Current Susceptible Population
    I::Vector{Int} # Current Infected Population
    R::Int # Current Recovered Population
    T::Matrix{Int} # Rows: Days from receiving test result; Columns: Infection Age
    params::InfParams
    prev_action::CovidAction
end

CovidPOMDPs.CovidState(S,I,R,T,p,a) = SingleCovidState(S,I,R,T,p,a)

CovidPOMDPs.statevars(s::SingleCovidState) = (s.S, s.I, s.R, s.T)

CovidPOMDPs.infected(s::SingleCovidState) = sum(s.I)

Base.@kwdef struct SingleCovidPOMDP{A} <: POMDP{SingleCovidState, CovidAction, Int}
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
    actions::A = CovidPOMDPs.CovidActionSpace()
end

"""
Take given CovidPOMDP obj and return same CovidPOMDP obj only with test_period changed to 1
"""
function CovidPOMDPs.unity_test_period(pomdp::SingleCovidPOMDP)
    return SingleCovidPOMDP(
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

function CovidPOMDPs.init_SIRT(pomdp::SingleCovidPOMDP, rng::AbstractRNG=Random.GLOBAL_RNG)
    N = pomdp.N
    S, inf, R = floor.(Int, simplex_sample(3, Float64(N), rng))

    horizon = INFECTION_HORIZON

    I = floor.(Int, simplex_sample(horizon, Float64(inf), rng))

    leftover = N - (S + sum(I) + R)
    R += leftover

    T = zeros(Int, pomdp.test_delay+1, horizon)

    return S, I, R, T
end

function CovidPOMDPs.init_SIRT(pomdp::SingleCovidPOMDP, inf::Int, rng::AbstractRNG=Random.GLOBAL_RNG)
    N = pomdp.N
    R = 0

    horizon = INFECTION_HORIZON

    I = floor.(Int, simplex_sample(horizon, Float64(inf), rng))

    leftover = N - sum(I)
    S = leftover

    T = zeros(Int, pomdp.test_delay+1, horizon)

    return S, I, R, T
end
