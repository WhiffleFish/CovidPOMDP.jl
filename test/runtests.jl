using CovidPOMDPs
using POMCPOW
using Test

macro isinferred(ex)
  quote
    try
      @inferred $ex
      true
    catch err
      println(err)
      false
    end
  end
end

include("simulation.jl")

include("solution.jl")
