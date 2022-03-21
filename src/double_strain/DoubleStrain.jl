module DoubleStrain

using Random
using ProgressMeter
using CovidPOMDPs
using ..CovidPOMDPs:
    INFECTION_HORIZON,
    INF_DIST,
    VIRAL_LOADS,
    POS_TEST_PROBS,
    SYMPTOMATIC_ISOLATION_PROB,
    ASYMPTOMATIC_PROB,
    SYMPTOM_PROBS,
    DEFAULT_PARAMS,
    InfParams,
    CovidState,
    CovidAction,
    simplex_sample,
    RVsum

include("typedef.jl")
export DoubleCovidPOMDP, DoubleCovidState

include("simulate.jl")

include("POMDPsInterface.jl")

include("updater.jl")

end
