@testset "Initialization" begin
    dists = CovidPOMDPs.FitInfectionDistributions(
        CovidPOMDPs.INF_DATAFRAME,
        CovidPOMDPs.INFECTION_HORIZON,
        50
    )
    @test @isinferred first(dists)

    test_probs = [
        CovidPOMDPs.prop_above_LOD(CovidPOMDPs.VIRAL_LOADS, day, CovidPOMDPs.LIMIT_OF_DETECTION)
        for day in 1:CovidPOMDPs.INFECTION_HORIZON
    ]

    @test @isinferred first(test_probs)
    @test all(≥(0.), test_probs)
end

@testset "shifting" begin
    T = rand(Int, 4, 3)
    T′ = circshift(T, (-1,1))
    T′[:,1] .= 0
    T′[end,:] .= 0
    CovidPOMDPs.shift_test!(T)
    @test all(T .== T′)

    I = rand(Int, 14)
    I′ = circshift(I, 1)
    I′[1] = 0
    CovidPOMDPs.shift_inf!(I)
    @test all(I .== I′)
end
