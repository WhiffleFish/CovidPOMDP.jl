using CovidPOMDPs
using POMDPs
using POMCPOW 
using Test

#=
macro testinferred(expr)
    @assert expr.head === :call
    f = expr.args[1]
    args = expr.args[2:end]
    return quote
        try
            @inferred $(f)($((esc(arg) for arg in args)...))
            @test true
        catch e
            if hasproperty(e, :msg) && occursin("does not match inferred return type", e.msg)
                @test false
            else
                throw(e)
            end
        end
    end
end
=#

include("simulation.jl")

include("solution.jl")
