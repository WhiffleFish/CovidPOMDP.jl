abstract type CovidState end

function statevars end

function infected end

function susceptible end

function recovered end

function init_SIRT end

function sim_step end

function unity_test_period end

function population(s::CovidState)
    return s.S + infected(s) + s.R
end

SIR(s::CovidState) = (s.S, infected(s), s.R)

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


function cdf_step(d::Distribution, N::Int)
    return Float64[cdf(d, i) - cdf(d, i-1) for i in 1:N]
end

function init_params(asymptomatic_prob=0.10)
    infection_distributions = similar(INF_DIST)
    pos_test_probs = copy(POS_TEST_PROBS)
    for (i,d) in enumerate(INF_DIST)
        k,θ = Distributions.params(d)
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

const AbstractCovidPOMDP = POMDP{<:CovidState, CovidAction, Int}

function simplex_sample(N::Int, m::Float64, rng::AbstractRNG=Random.GLOBAL_RNG)
    v = rand(rng, N-1)*m
    push!(v, 0, m)
    sort!(v)
    return (v - circshift(v,1))[2:end]
end

function CovidState(pomdp::POMDP{S}, params::InfParams=DEFAULT_PARAMS) where S<:CovidState
    return S(init_SIRT(pomdp)..., params, CovidAction(0.0))
end

function rand_initialstate(pomdp::POMDP{S}) where S<:CovidState
    params = init_params()
    return S(init_SIRT(pomdp)..., params, CovidAction(0.0))
end

function rand_initialstate(pomdp::POMDP{S}, Idist::Distribution) where S<:CovidState
    inf = clamp(floor(Int, rand(Idist)), 0, pomdp.N)
    params = init_params()
    return S(init_SIRT(pomdp, inf)..., params, CovidAction(0.0))
end

function POMDPs.initialstate(pomdp::AbstractCovidPOMDP)
    return ImplicitDistribution() do rng
        rand_initialstate(pomdp)
    end
end

function POMDPs.initialstate(pomdp::AbstractCovidPOMDP, Idist::Distribution)
    return ImplicitDistribution() do rng
        rand_initialstate(pomdp, Idist)
    end
end

struct SimHist{S<:CovidState}
    S::Vector{S}
    N::Int # Total Population
    T::Int # Simulation Time
    pos_test::Vector{Int}
    actions::Vector{CovidAction}
    rewards::Vector{Float64}
    beliefs::Matrix{S}
end

infected(h::SimHist) = infected.(h.S)
susceptible(h::SimHist) = susceptible.(h.S)
recovered(h::SimHist) = recovered.(h.S)
