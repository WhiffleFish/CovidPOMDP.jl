module CovidPOMDPs

using Reexport
using Random
using ProgressMeter
@reexport using POMDPs
@reexport using POMDPModelTools
@reexport using Distributions
@reexport using Statistics
@reexport using ParticleFilters
using DelimitedFiles
using DataFrames: DataFrame
@reexport using CairoMakie
@reexport using BasicPOMCP

include("init.jl")

include("constants.jl")

include("typedef.jl")
export CovidActionSpace, CovidAction

include("simulate.jl")
export SimHist

include("POMDPsInterface.jl")

include("proportional_control.jl")

include("updater.jl")

include("covid_filter.jl")

include("plots.jl")

include(joinpath("single_strain", "SingleStrain.jl"))
using .SingleStrain
export SingleCovidPOMDP, SingleCovidState

include(joinpath("double_strain", "DoubleStrain.jl"))
using .DoubleStrain
export DoubleCovidPOMDP, DoubleCovidState

export susceptible, infected, recovered

end
