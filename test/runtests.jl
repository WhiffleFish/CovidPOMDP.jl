using CovidPOMDPs
using POMCPOW
using Test

macro isinferred(ex)
  quote
    try
      @inferred $(Expr(ex.head, esc.(ex.args)...))
      true
    catch err
      println(err)
      false
    end
  end
end

function validate_history(hist::SimHist)
    S = susceptible(hist)
    I = infected(hist)
    R = recovered(hist)
    N = hist.N
    T = hist.T

    @test all(≥(0), S)
    @test all(≥(0), I)
    @test all(≥(0), R)
    @test all(≥(0), hist.pos_test)

    # Population size preservation
    @test all(==(N), S[i] + I[i] + R[i] for i in 1:T)

    @test all( 0.0 .≤ getfield.(hist.actions,:testing_prop) .≤ 1.0)

    return nothing
end

@testset "Common" begin
    include("common.jl")
end

@testset verbose=true "Single Strain" begin
    include(joinpath("single_strain", "single_strain.jl"))
end

@testset verbose=true "Double Strain" begin
    include(joinpath("double_strain", "double_strain.jl"))
end
