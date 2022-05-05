"""
# Arguments
- `S::Int` - Current Susceptible Population
- `I::Vector{Int}` - Current Infected Population
- `R::Int` - Current Recovered Population
- `Tests::Matrix{Int}` - Array for which people belonging to array element ``T_{i,j}`` are ``i-1`` days away
    from receiving positive test and have infection age ``j``
"""
struct DoubleCovidState <: CovidPOMDPs.CovidState
    S::Int # Current Susceptible Population
    I1::Vector{Int} # Current Infected Population
    I2::Vector{Int} # Current Infected Population
    R::Int # Current Recovered Population
    T1::Matrix{Int} # Rows: Days from receiving test result; Columns: Infection Age
    T2::Matrix{Int} # Rows: Days from receiving test result; Columns: Infection Age
    params::InfParams
    prev_action::CovidAction
end

CovidPOMDPs.CovidState(S,I1,I2,R,T1,T2,p,a) = DoubleCovidState(S,I1,I2,R,T1,T2,p,a)

CovidPOMDPs.statevars(s::DoubleCovidState) = (s.S, s.I1, s.I2, s.R, s.T1, s.T2)

CovidPOMDPs.susceptible(s::DoubleCovidState) = s.S
CovidPOMDPs.infected(s::DoubleCovidState) = sum(s.I1) + sum(s.I2)
CovidPOMDPs.recovered(s::DoubleCovidState) = s.R

Base.@kwdef struct DoubleCovidPOMDP{A} <: POMDP{DoubleCovidState, CovidPOMDPs.CovidAction, Int}
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

    params::CovidPOMDPs.InfParams = DEFAULT_PARAMS
end

"""
Take given CovidPOMDP obj and return same CovidPOMDP obj only with test_period changed to 1
"""
function CovidPOMDPs.unity_test_period(pomdp::DoubleCovidPOMDP)
    return DoubleCovidPOMDP(
        pomdp.test_delay,
        pomdp.N,
        pomdp.discount,
        pomdp.inf_loss,
        pomdp.test_loss,
        pomdp.testrate_loss,
        1,
        pomdp.actions,
        pomdp.params
    )
end

function CovidPOMDPs.init_SIRT(pomdp::DoubleCovidPOMDP, rng::AbstractRNG=Random.GLOBAL_RNG)
    N = pomdp.N
    S, inf, R = floor.(Int, CovidPOMDPs.simplex_sample(3, Float64(N), rng))
    inf1, inf2 = floor.(Int, CovidPOMDPs.simplex_sample(2, Float64(inf), rng))

    horizon = INFECTION_HORIZON

    I1 = floor.(Int, CovidPOMDPs.simplex_sample(horizon, Float64(inf1), rng))
    I2 = floor.(Int, CovidPOMDPs.simplex_sample(horizon, Float64(inf2), rng))

    leftover = N - (S + sum(I1) + sum(I2) + R)
    R += leftover

    T1 = zeros(Int, pomdp.test_delay+1, horizon)
    T2 = zeros(Int, pomdp.test_delay+1, horizon)

    return S, I1, I2, R, T1, T2
end

function CovidPOMDPs.init_SIRT(pomdp::DoubleCovidPOMDP, inf::Int, rng::AbstractRNG=Random.GLOBAL_RNG)
    N = pomdp.N
    R = 0

    horizon = INFECTION_HORIZON
    inf1, inf2 = floor.(Int, simplex_sample(horizon, Float64(inf), rng))
    I1 = floor.(Int, CovidPOMDPs.simplex_sample(horizon, Float64(inf1), rng))
    I2 = floor.(Int, CovidPOMDPs.simplex_sample(horizon, Float64(inf2), rng))

    leftover = N - (sum(I1) + sum(I2))
    S = leftover

    T1 = zeros(Int, pomdp.test_delay+1, horizon)
    T2 = zeros(Int, pomdp.test_delay+1, horizon)

    return S, I1, I2, R, T1, T2
end
